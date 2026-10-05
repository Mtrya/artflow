"""Caption lengths checked against controlled tokens and independent expectations."""

import importlib
import json

import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.dataset import length_metadata as metadata


@pytest.fixture
def tokenizer(tmp_path, monkeypatch):
    # This fixture supplies a three-token drop and four-token retained cap.
    # Its literal prompt contains only the caption; no production template or
    # helper participates in the expected token counts.
    online = importlib.import_module("src.utils.encode_text")
    for module in (metadata, online):
        for name, value in dict(
            DROP_IDX=3,
            MAX_SEQUENCE_LENGTH=4,
            RETAINED_MIN_LENGTH=1,
            PROMPT_TEMPLATE="{user_prompt}",
            SYSTEM_PROMPT="",
        ).items():
            monkeypatch.setattr(module, name, value)
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[PAD]": 1, "word": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    encoder = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    encoder.save_pretrained(tmp_path / "tokenizer")
    return encoder, str(tmp_path / "tokenizer"), online


def test_offline_lengths_follow_manifest_row_order(tmp_path, tokenizer):
    _, tokenizer_path, _ = tokenizer
    path = tmp_path / "dataset"
    Dataset.from_dict(
        {
            "captions": [
                ["word word", "word " * 5],
                ["word " * 10],
                [""],
                ["word " * 4],
            ],
            "resolution_bucket_id": [1, 2, 3, 4],
        }
    ).save_to_disk(str(path), num_shards=2)
    state_path = path / "state.json"
    state = json.loads(state_path.read_text())
    state["_data_files"].reverse()
    state_path.write_text(json.dumps(state))
    actual = metadata.build_from_dataset(str(path), tokenizer_path)
    np.testing.assert_array_equal(actual.resolution_ids, [3, 4, 1, 2])
    np.testing.assert_array_equal(actual.caption_offsets, [0, 1, 2, 4, 5])
    np.testing.assert_array_equal(actual.prompt_lengths, [1, 1, 1, 2, 4])
    assert list(load_from_disk(str(path))["resolution_bucket_id"]) == [3, 4, 1, 2]


@pytest.mark.parametrize("fast_slice", [False, True])
def test_online_retention_has_independent_expected_lengths(tokenizer, fast_slice):
    from types import SimpleNamespace

    encoder, _, online = tokenizer
    model = SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32,
        model=lambda input_ids, attention_mask: SimpleNamespace(
            last_hidden_state=input_ids.unsqueeze(-1).float()),
    )
    _, retained, _ = online.encode_text(
        ["word word", "word " * 5, "word " * 10], model, encoder,
        pooling=False, fast_slice=fast_slice,
    )
    assert retained.sum(dim=1).tolist() == [1, 2, 4]


def test_cached_lengths_follow_replaced_dataset(tmp_path, tokenizer):
    _, tokenizer_path, _ = tokenizer
    path = str(tmp_path / "dataset")

    def save(caption):
        Dataset.from_dict(
            {"captions": [[caption]], "resolution_bucket_id": [1]}
        ).save_to_disk(path)

    save("word word")
    assert metadata.ensure_sidecar(path, tokenizer_path).prompt_lengths.tolist() == [1]
    save("word " * 10)
    assert metadata.ensure_sidecar(path, tokenizer_path).prompt_lengths.tolist() == [4]
    assert metadata.ensure_sidecar(path, tokenizer_path).prompt_lengths.tolist() == [4]


@pytest.mark.parametrize("fault", ["missing", "empty", "missing_shard"])
def test_manifest_is_required_by_metadata_readers(tmp_path, tokenizer, fault):
    _, tokenizer_path, _ = tokenizer
    path = tmp_path / "dataset"
    Dataset.from_dict(
        {"captions": [["word"]], "resolution_bucket_id": [1]}
    ).save_to_disk(str(path))
    state = path / "state.json"
    if fault == "missing":
        state.unlink()
    elif fault == "empty":
        state.write_text('{"_data_files": []}')
    else:
        state.write_text('{"_data_files": [{"filename": "absent.arrow"}]}')
    for read in (
        lambda: metadata._load_text_columns(str(path)),
        lambda: metadata._source_signature(str(path), tokenizer_path),
    ):
        with pytest.raises((ValueError, FileNotFoundError)):
            read()


@pytest.mark.parametrize(
    "fault", ["version", "contract", "zero", "negative", "fractional"]
)
def test_persisted_metadata_rejects_invalid_formats(tmp_path, fault):
    payload = dict(
        resolution_ids=np.array([1]),
        caption_offsets=np.array([0, 1]),
        prompt_lengths=np.array([2]),
        metadata_version=np.asarray("row-length-v3"),
        metadata_info=np.asarray(json.dumps(metadata.prompt_metadata_contract(1))),
    )
    if fault == "version":
        payload["metadata_version"] = np.asarray("row-length-v1")
    elif fault == "contract":
        del payload["metadata_info"]
    else:
        payload["prompt_lengths"] = np.array(
            [{"zero": 0, "negative": -5, "fractional": 1.5}[fault]]
        )
    path = tmp_path / "lengths.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        metadata.RowLengthMetadata.load(path)
