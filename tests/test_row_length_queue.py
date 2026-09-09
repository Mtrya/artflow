import random

import numpy as np
import pytest
import torch
from datasets import Dataset

from src.dataset.captions import (
    caption_probabilities_from_token_counts,
    sample_caption,
    sample_caption_index_from_token_counts,
)
from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.sampler import (
    BucketPlan,
    LenBucket,
    RowDescriptorDataset,
    RowLengthQueueBatchSampler,
    RowRef,
    pad_text_to_hi,
    row_length_collate_fn,
)


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        cap = kwargs.get("max_length")
        ids = [list(range(min(len(text), cap) if cap else len(text))) for text in texts]
        return {"input_ids": ids}


def metadata(resolutions, caption_lengths, prompt_lengths=None):
    offsets = [0]
    curriculum = []
    for captions in caption_lengths:
        curriculum.extend(captions)
        offsets.append(offsets[-1] + len(captions))
    if prompt_lengths is None:
        prompt_lengths = curriculum
    return RowLengthMetadata(
        resolution_ids=np.asarray(resolutions),
        caption_offsets=np.asarray(offsets),
        curriculum_lengths=np.asarray(curriculum),
        prompt_lengths=np.asarray(prompt_lengths),
    )


def dataset(resolutions, captions):
    return Dataset.from_dict(
        {
            "latents": [np.full((1, 2, 2), i, dtype=np.float32) for i in range(len(captions))],
            "captions": captions,
            "resolution_bucket_id": resolutions,
        }
    )


def sampler_for(meta, buckets=((4, 2), (2048, 2)), weights=None, seed=4):
    return RowLengthQueueBatchSampler(
        [meta], BucketPlan.uniform([1], [LenBucket(*bucket) for bucket in buckets]),
        dataset_weights=weights, shuffle=False, seed=seed,
    )


def test_row_preservation_and_caption_stage_is_row_inner():
    meta = metadata([1, 1], [[1, 100], [100, 1]], [1, 100, 100, 1])
    sampler = sampler_for(meta, buckets=((100, 1), (2048, 1)), seed=7)
    iterator = iter(sampler)
    refs = [next(iterator)[0] for _ in range(50)]
    assert {ref.row_idx for ref in refs} == {0, 1}
    assert all(ref.caption_idx in (0, 1) for ref in refs)
    assert all(ref.row_idx in (0, 1) for ref in refs)

    sampler.set_stage(0.0)
    short_refs = [next(iter(sampler))[0] for _ in range(10)]
    assert all(ref.caption_idx in (0, 1) for ref in short_refs)


def test_mixed_dataset_batch_and_dataset_not_in_bucket_key():
    meta0 = metadata([1, 1], [[2], [2]])
    meta1 = metadata([1, 1], [[2], [2]])
    sampler = RowLengthQueueBatchSampler(
        [meta0, meta1], BucketPlan.uniform([1], [LenBucket(4, 4)]),
        dataset_weights=[1, 1], shuffle=False, seed=1,
    )
    batch = next(iter(sampler))
    assert len(batch) == 4
    assert {ref.resolution_id for ref in batch} == {1}
    assert {ref.len_bucket_idx for ref in batch} == {0}
    assert {ref.dataset_id for ref in batch} == {0, 1}


def test_no_cross_resolution_or_length_bucket_mixing():
    meta = metadata([1, 2, 1, 2], [[2], [2], [8], [8]])
    plan = BucketPlan({1: [LenBucket(4, 1), LenBucket(2048, 1)], 2: [LenBucket(4, 1), LenBucket(2048, 1)]})
    sampler = RowLengthQueueBatchSampler([meta], plan, shuffle=False, seed=1)
    for _ in range(30):
        batch = next(iter(sampler))
        assert len({(ref.resolution_id, ref.len_bucket_idx) for ref in batch}) == 1


def test_partial_queue_persists_until_a_later_row_fills_it():
    meta = metadata([1, 1], [[2], [2]])
    sampler = sampler_for(meta, buckets=((4, 2), (2048, 2)), seed=1)
    iterator = iter(sampler)
    first = next(iterator)
    state = sampler.state_dict()
    assert first[0].batch_id == first[1].batch_id

    meta2 = metadata([1, 1, 1], [[2], [2], [2]])
    sampler2 = sampler_for(meta2, buckets=((4, 2), (2048, 2)), seed=1)
    # A direct draw that does not complete a bucket remains serialized.
    sampler3 = RowLengthQueueBatchSampler(
        [metadata([1], [[2]])], BucketPlan.uniform([1], [LenBucket(4, 2)]),
        shuffle=False, seed=1,
    )
    sampler3._enqueue_draw()
    assert sum(len(q) for q in sampler3.state_dict()["queues"].values()) == 1
    restored = sampler_for(metadata([1], [[2]]), buckets=((4, 2),), seed=1)
    restored.load_state_dict(sampler3.state_dict())
    assert sum(len(q) for q in restored._queues.values()) == 1
    del sampler2


def test_bucket_boundary_is_inclusive_and_2048_is_valid():
    plan = BucketPlan({1: [LenBucket(64, 1), LenBucket(128, 1), LenBucket(2048, 1)]})
    assert plan.bucket_for(1, 64)[0] == 0
    assert plan.bucket_for(1, 128)[0] == 1
    assert plan.bucket_for(1, 2048)[0] == 2
    assert BucketPlan({1: [LenBucket(64, 1)]}).bucket_for(1, 64)[0] == 0


def test_collate_strict_metadata_and_row_resolution():
    base = {
        "latents": torch.zeros(1, 2, 2), "captions": "caption", "dataset_id": 0,
        "row_idx": 0, "caption_idx": 0, "resolution_bucket_id": 1,
        "retained_length": 2, "len_bucket_idx": 0, "bucket_hi": 4, "batch_id": 3,
    }
    output = row_length_collate_fn([base, dict(base)])
    assert output["latents"].shape == (2, 1, 2, 2)
    for key in ("resolution_bucket_id", "bucket_hi", "batch_id"):
        bad = dict(base)
        bad[key] = base[key] + 1
        with pytest.raises(ValueError, match="mixed"):
            row_length_collate_fn([base, bad])


def test_pad_text_overflow_raises():
    txt = torch.zeros(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds"):
        pad_text_to_hi(txt, mask, 4)


def test_state_roundtrip_replays_ready_queue_and_rng():
    meta = metadata([1, 1, 1], [[2], [2], [2]])
    sampler = sampler_for(meta, buckets=((4, 2),), seed=11)
    iterator = iter(sampler)
    first = next(iterator)
    state = sampler.state_dict()

    restored = sampler_for(meta, buckets=((4, 2),), seed=999)
    restored.load_state_dict(state)
    replayed = next(iter(restored))
    assert [ref.to_state() for ref in replayed] == [ref.to_state() for ref in first]
    assert first[0].batch_id == first[1].batch_id


def test_descriptor_dataset_and_metadata_save_load(tmp_path):
    ds = dataset([1], [["a", "long caption"]])
    meta = RowLengthMetadata.from_hf_dataset(ds, StubTokenizer())
    meta_path = tmp_path / "row_lengths.bin"
    meta.save(meta_path)
    loaded = RowLengthMetadata.load(meta_path)
    assert loaded.metadata_version == meta.metadata_version
    assert np.array_equal(loaded.prompt_lengths, meta.prompt_lengths)
    assert loaded.metadata_info == meta.metadata_info
    meta.validate_against_dataset(ds)
    assert meta.num_rows == 1
    assert meta.num_captions == 2
    assert meta.prompt_lengths.tolist() == [
        max(
            len(
                "<|im_start|>system\nDescribe the image, focusing on its content, artistic style, composition, lighting, color, texture, and the spatial relationships between objects and the background:<|im_end|>\n<|im_start|>user\n"
                + caption
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            - 38,
            1,
        )
        for caption in ["a", "long caption"]
    ]

    ref = RowRef(0, 0, 1, 1, int(meta.prompt_lengths[1]), 0, 2048, 0)
    sample = RowDescriptorDataset([ds])[ref]
    assert sample["captions"] == "long caption"


def test_caption_helpers_match_sample_caption_distribution_shape():
    counts = [1, 4, 10]
    probs = caption_probabilities_from_token_counts(counts, 0.5)
    assert np.isclose(sum(probs), 1)
    random.seed(2)
    idx = sample_caption_index_from_token_counts(counts, 0.5)
    random.seed(2)
    assert sample_caption(["a", "bbbb", "cccccccccc"], 0.5) in {"a", "bbbb", "cccccccccc"}
    assert idx in range(3)
