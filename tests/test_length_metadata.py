"""Tests for the dataset-companion caption-length sidecar
(src/dataset/length_metadata.py and the precompute writer)."""

import json
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset, load_from_disk

from src.dataset.length_metadata import (
    METADATA_VERSION,
    SIDECAR_FILENAME,
    RowLengthMetadata,
    build_from_dataset,
    ensure_sidecar,
    sidecar_path,
)
from src.utils.prompt_contract import (
    DROP_IDX,
    MAX_SEQUENCE_LENGTH,
    PROMPT_TEMPLATE,
    RETAINED_MIN_LENGTH,
    SYSTEM_PROMPT,
)

TOKENIZER_PATH = "dummy/tokenizer"


class CharIndexTokenizer:
    """Deterministic stub: one token id per character, hard cap at max_length.

    Mirrors the call contract of the offline tokenizer path
    (``truncation=True``, ``max_length=MAX_SEQUENCE_LENGTH + DROP_IDX``) so a
    test can re-derive expected lengths from plain string arithmetic.
    """

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        cap = kwargs.get("max_length")
        ids = [list(range(len(text))) for text in texts]
        if cap is not None:
            ids = [row[:cap] for row in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def synthetic_rows():
    """Small corpus: 1-3 captions per row, one empty caption list, a caption
    long enough to hit the 2048 retained-token cap, and multi-byte text."""
    return [
        (1, ["a short caption", "another caption here"]),
        (2, ["single"]),
        (1, []),
        (3, ["中文标题，描述图片内容。"]),
        (2, ["s" * 4000]),
        (1, ["x", "y", "z"]),
    ]


def make_dataset(rows):
    return Dataset.from_dict(
        {
            "latents": [np.zeros((2, 3), dtype=np.float32) for _ in rows],
            "captions": [captions for _, captions in rows],
            "resolution_bucket_id": [resolution for resolution, _ in rows],
        }
    )


def expected_prompt_length(caption):
    """Independent re-derivation of the retained length rule:
    format the complete prompt, cap at MAX+DROP_IDX tokens, drop the first
    DROP_IDX tokens, and floor at RETAINED_MIN_LENGTH."""
    prompt = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=caption)
    full_length = min(len(prompt), MAX_SEQUENCE_LENGTH + DROP_IDX)
    return max(full_length - DROP_IDX, RETAINED_MIN_LENGTH)


def expected_arrays(rows):
    lengths = [
        expected_prompt_length(caption)
        for _, captions in rows
        for caption in captions
    ]
    offsets = [0]
    for _, captions in rows:
        offsets.append(offsets[-1] + len(captions))
    return {
        "prompt_lengths": lengths,
        "caption_offsets": offsets,
        "resolution_ids": [resolution for resolution, _ in rows],
    }


@pytest.fixture
def stub_tokenizer_patch():
    with mock.patch("transformers.AutoTokenizer.from_pretrained") as patched:
        patched.return_value = CharIndexTokenizer()
        yield patched


def test_sidecar_path_and_filename():
    assert SIDECAR_FILENAME == "length_metadata.npz"
    assert sidecar_path("/data/ds") == Path("/data/ds") / SIDECAR_FILENAME
    assert sidecar_path("relative") == Path("relative") / SIDECAR_FILENAME


def test_build_from_dataset_matches_independent_retokenization(tmp_path, stub_tokenizer_patch):
    rows = synthetic_rows()
    out_dir = tmp_path / "ds"
    make_dataset(rows).save_to_disk(str(out_dir))

    metadata = build_from_dataset(str(out_dir), TOKENIZER_PATH)

    expected = expected_arrays(rows)
    assert metadata.prompt_lengths.tolist() == expected["prompt_lengths"]
    assert metadata.caption_offsets.tolist() == expected["caption_offsets"]
    assert metadata.resolution_ids.tolist() == expected["resolution_ids"]
    assert metadata.num_rows == len(rows)
    assert metadata.num_captions == len(expected["prompt_lengths"])
    # The 4000-character caption pins the 2048 retained cap.
    assert int(metadata.prompt_lengths.max()) == MAX_SEQUENCE_LENGTH
    assert int(metadata.prompt_lengths.min()) >= RETAINED_MIN_LENGTH
    # Every value stays inside the contract enforced at load time.
    assert all(RETAINED_MIN_LENGTH <= length <= MAX_SEQUENCE_LENGTH for length in metadata.prompt_lengths)


def test_build_from_dataset_follows_state_json_shard_order(tmp_path, stub_tokenizer_patch):
    """Rows across Arrow shards follow state.json order, not file-name order."""
    rows_a = [(1, ["from-a"]), (1, ["also-a"])]
    rows_b = [(2, ["from-b"])]
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    make_dataset(rows_a).save_to_disk(str(dir_a))
    make_dataset(rows_b).save_to_disk(str(dir_b))

    merged = tmp_path / "merged"
    merged.mkdir()
    arrow_a = next(dir_a.glob("data-*.arrow"))
    arrow_b = next(dir_b.glob("data-*.arrow"))
    shutil.copy(arrow_a, merged / "data-00001-of-00002.arrow")
    shutil.copy(arrow_b, merged / "data-00000-of-00002.arrow")
    # Deliberately list shard 00001 first: state order wins over lexicographic.
    (merged / "state.json").write_text(
        json.dumps(
            {
                "_data_files": [
                    {"filename": "data-00001-of-00002.arrow"},
                    {"filename": "data-00000-of-00002.arrow"},
                ]
            }
        )
    )

    metadata = build_from_dataset(str(merged), TOKENIZER_PATH)
    assert metadata.resolution_ids.tolist() == [1, 1, 2]
    first_row_slice = metadata.row_slice(0)
    assert metadata.prompt_lengths[first_row_slice].tolist() == [
        expected_prompt_length("from-a")
    ]

    # Without state.json the lexicographic glob fallback reverses the order.
    (merged / "state.json").unlink()
    fallback = build_from_dataset(str(merged), TOKENIZER_PATH)
    assert fallback.resolution_ids.tolist() == [2, 1, 1]


def test_ensure_sidecar_builds_then_loads_idempotently(tmp_path, stub_tokenizer_patch):
    rows = synthetic_rows()
    out_dir = tmp_path / "ds"
    make_dataset(rows).save_to_disk(str(out_dir))

    first = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
    assert (out_dir / SIDECAR_FILENAME).is_file()
    stub_tokenizer_patch.assert_called_once_with(TOKENIZER_PATH, local_files_only=True)

    # Second call takes the load path: the tokenizer is never loaded again and
    # the result is identical to the first build.
    with mock.patch("transformers.AutoTokenizer.from_pretrained") as later:
        second = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
        later.assert_not_called()
    for field in ("resolution_ids", "caption_offsets", "prompt_lengths"):
        assert np.array_equal(getattr(first, field), getattr(second, field)), field
    assert first.metadata_info == second.metadata_info

    # The persisted companion describes the dataset on disk.
    first.validate_against_dataset(load_from_disk(str(out_dir)))


def test_ensure_sidecar_rebuilds_corrupt_file(tmp_path, stub_tokenizer_patch):
    rows = synthetic_rows()
    out_dir = tmp_path / "ds"
    make_dataset(rows).save_to_disk(str(out_dir))
    sidecar = out_dir / SIDECAR_FILENAME
    sidecar.write_bytes(b"not a numpy archive")

    metadata = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
    assert metadata.prompt_lengths.tolist() == expected_arrays(rows)["prompt_lengths"]
    reloaded = RowLengthMetadata.load(sidecar)
    assert np.array_equal(reloaded.prompt_lengths, metadata.prompt_lengths)
    metadata.validate_against_dataset(load_from_disk(str(out_dir)))


def test_ensure_sidecar_fails_loudly_without_arrow_files(tmp_path, stub_tokenizer_patch):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no HF Arrow data files"):
        ensure_sidecar(str(empty), TOKENIZER_PATH)


def test_validate_against_dataset_rejects_row_count_mismatch(tmp_path, stub_tokenizer_patch):
    rows = synthetic_rows()
    out_dir = tmp_path / "ds"
    make_dataset(rows).save_to_disk(str(out_dir))
    metadata = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
    metadata.validate_against_dataset(load_from_disk(str(out_dir)))

    # Replace the dataset directory with one of a different row count; the
    # companion file (still on disk) must be rejected by the row check.
    other = make_dataset([(1, ["replacement row"])])
    other.save_to_disk(str(out_dir))
    assert ensure_sidecar(str(out_dir), TOKENIZER_PATH).num_rows == len(other)
    with pytest.raises(ValueError, match="do not match dataset rows"):
        metadata.validate_against_dataset(load_from_disk(str(out_dir)))


def test_sidecar_rebuilds_after_same_size_dataset_rewrite(tmp_path, stub_tokenizer_patch):
    out_dir = tmp_path / "ds"
    make_dataset([(1, ["short"])]).save_to_disk(str(out_dir))
    first = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
    make_dataset([(1, ["long caption " * 100])]).save_to_disk(str(out_dir))
    second = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
    assert first.num_rows == second.num_rows
    assert first.prompt_lengths.tolist() != second.prompt_lengths.tolist()


def test_sidecar_rebuilds_after_tokenizer_change(tmp_path, stub_tokenizer_patch):
    out_dir = tmp_path / "ds"
    make_dataset([(1, ["caption"])]).save_to_disk(str(out_dir))
    first = ensure_sidecar(str(out_dir), TOKENIZER_PATH)
    second = ensure_sidecar(str(out_dir), "different/tokenizer")
    assert first.metadata_info["source_signature"] != second.metadata_info["source_signature"]
    assert stub_tokenizer_patch.call_count == 2


def test_sidecar_rebuilds_after_local_tokenizer_update(tmp_path, stub_tokenizer_patch):
    out_dir = tmp_path / "ds"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_file.write_text('{}')
    make_dataset([(1, ["caption"])]).save_to_disk(str(out_dir))
    first = ensure_sidecar(str(out_dir), str(tokenizer_dir))
    tokenizer_file.write_text('{"changed": true}')
    second = ensure_sidecar(str(out_dir), str(tokenizer_dir))
    assert first.metadata_info["source_signature"] != second.metadata_info["source_signature"]
    assert stub_tokenizer_patch.call_count == 2


def test_precompute_write_length_metadata(tmp_path, stub_tokenizer_patch):
    from src.dataset.precompute import write_length_metadata

    rows = synthetic_rows()
    out_dir = tmp_path / "out"
    dataset = make_dataset(rows)
    dataset.save_to_disk(str(out_dir))

    metadata = write_length_metadata(dataset, out_dir, TOKENIZER_PATH)
    assert (out_dir / SIDECAR_FILENAME).is_file()
    assert metadata.prompt_lengths.tolist() == expected_arrays(rows)["prompt_lengths"]
    assert np.array_equal(
        RowLengthMetadata.load(out_dir / SIDECAR_FILENAME).prompt_lengths,
        metadata.prompt_lengths,
    )
    metadata.validate_against_dataset(load_from_disk(str(out_dir)))


def test_precompute_write_length_metadata_requires_saved_dataset(tmp_path, stub_tokenizer_patch):
    from src.dataset.precompute import write_length_metadata

    dataset = make_dataset(synthetic_rows())
    with pytest.raises(ValueError, match="save_to_disk"):
        write_length_metadata(dataset, tmp_path / "never-saved", TOKENIZER_PATH)
    stub_tokenizer_patch.assert_not_called()


def test_precompute_write_length_metadata_rejects_different_rows(tmp_path, stub_tokenizer_patch):
    from src.dataset.precompute import write_length_metadata

    out_dir = tmp_path / "out"
    make_dataset(synthetic_rows()).save_to_disk(str(out_dir))
    mismatched = make_dataset([(1, ["only one row"])])
    with pytest.raises(ValueError, match="refusing to write"):
        write_length_metadata(mismatched, out_dir, TOKENIZER_PATH)
    # Nothing was written for the mismatched rows.
    assert not (out_dir / SIDECAR_FILENAME).exists()


def test_real_qwen_tokenizer_build_matches_training_rule(tmp_path):
    """End-to-end with the real Qwen3-0.6B tokenizer when it is cached locally.

    build_from_dataset must produce exactly the lengths obtained by
    tokenizing each complete prompt with the same rule the trainer uses
    (truncation at MAX_SEQUENCE_LENGTH + DROP_IDX, DROP_IDX dropped,
    RETAINED_MIN_LENGTH floor).  This test is skipped on machines without the
    tokenizer in the local Hugging Face cache.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen/Qwen3-0.6B tokenizer not cached locally")
    rows = [
        (1, ["a short english caption"]),
        (2, ["一只站在树枝上的猫，背景虚化。"]),
        (1, ["Impressionist oil painting", "landscape by Monet"]),
        (2, ["word " * 1200]),
    ]
    out_dir = tmp_path / "ds"
    make_dataset(rows).save_to_disk(str(out_dir))

    metadata = build_from_dataset(str(out_dir), "Qwen/Qwen3-0.6B")

    expected = []
    for _, captions in rows:
        for caption in captions:
            prompt = PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT, user_prompt=caption
            )
            encoded = tokenizer(
                [prompt],
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH + DROP_IDX,
                padding=False,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            full_length = len(input_ids[0])
            expected.append(max(full_length - DROP_IDX, RETAINED_MIN_LENGTH))

    assert metadata.prompt_lengths.tolist() == expected
    assert metadata.num_rows == len(rows)
    assert all(RETAINED_MIN_LENGTH <= length <= MAX_SEQUENCE_LENGTH for length in metadata.prompt_lengths)


def test_sidecar_rebuilds_when_the_prompt_contract_changes(tmp_path, stub_tokenizer_patch):
    """A contract change must invalidate the sidecar even if data and tokenizer
    are untouched: the stored lengths would otherwise describe a window the
    trainer no longer computes."""
    import src.dataset.length_metadata as length_metadata

    dataset_dir = tmp_path / "ds"
    make_dataset(synthetic_rows()).save_to_disk(str(dataset_dir))

    first = ensure_sidecar(str(dataset_dir), TOKENIZER_PATH)
    assert max(first.prompt_lengths) > MAX_SEQUENCE_LENGTH // 2

    monkeypatched = MAX_SEQUENCE_LENGTH // 2
    with mock.patch.object(length_metadata, "MAX_SEQUENCE_LENGTH", monkeypatched):
        second = ensure_sidecar(str(dataset_dir), TOKENIZER_PATH)
        assert max(second.prompt_lengths) <= monkeypatched

    assert (second.metadata_info["source_signature"]
            != first.metadata_info["source_signature"])


def test_sidecar_rebuilds_when_the_metadata_version_changes(tmp_path, stub_tokenizer_patch):
    """A sidecar written by an older code version is rebuilt: the same field
    names can carry different meaning across versions."""
    dataset_dir = tmp_path / "ds"
    make_dataset(synthetic_rows()).save_to_disk(str(dataset_dir))
    ensure_sidecar(str(dataset_dir), TOKENIZER_PATH)

    stored = RowLengthMetadata.load(sidecar_path(str(dataset_dir)))
    stale = RowLengthMetadata(
        resolution_ids=stored.resolution_ids,
        caption_offsets=stored.caption_offsets,
        prompt_lengths=stored.prompt_lengths,
        metadata_version="row-length-v2",
        metadata_info=stored.metadata_info,
    )
    stale.save(sidecar_path(str(dataset_dir)))

    rebuilt = ensure_sidecar(str(dataset_dir), TOKENIZER_PATH)
    assert rebuilt.metadata_version == METADATA_VERSION
    assert RowLengthMetadata.load(
        sidecar_path(str(dataset_dir))).metadata_version == METADATA_VERSION
