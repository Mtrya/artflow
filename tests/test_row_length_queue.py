"""Row sampling checked against source probabilities, input rows and uninterrupted draws."""

from collections import Counter

import numpy as np
import pytest
from datasets import Dataset

from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.sampler import BucketPlan, LenBucket, RowDescriptorDataset, RowLengthQueueBatchSampler, row_length_collate_fn
from src.pretrain.train import set_caption_curriculum


def metadata(resolutions, lengths):
    return RowLengthMetadata(resolution_ids=np.asarray(resolutions),
        caption_offsets=np.arange(len(lengths) + 1), prompt_lengths=np.asarray(lengths))


def test_source_probability_is_independent_of_dataset_size():
    entries = [metadata([1] * n, [2] * n) for n in (2, 100)]
    sampler = RowLengthQueueBatchSampler(entries, BucketPlan.uniform([1], [LenBucket(8, 1)]),
        dataset_weights=[3, 1], seed=47)
    iterator = iter(sampler)
    counts = Counter(next(iterator)[0].dataset_id for _ in range(6000))
    assert counts[0] / 6000 == pytest.approx(0.75, abs=0.025)


def test_batches_agree_with_source_images_and_caption_lengths():
    resolutions, lengths = [1, 2, 1, 2], [2, 2, 9, 9]
    ds = Dataset.from_dict({'latents': [np.full((1, r + 1, 2), i, dtype=np.float32)
                                      for i, r in enumerate(resolutions)],
                           'captions': [['x' * n] for n in lengths],
                           'resolution_bucket_id': resolutions})
    meta = metadata(resolutions, lengths)
    plan = BucketPlan.uniform([1, 2], [LenBucket(4, 2), LenBucket(16, 2)])
    sampler = RowLengthQueueBatchSampler([meta], plan, seed=1)
    rows = RowDescriptorDataset([ds])
    seen = set()
    iterator = iter(sampler)
    for _ in range(20):
        refs = next(iterator)
        batch = row_length_collate_fn([rows[ref] for ref in refs])
        for index, caption in enumerate(batch['captions']):
            row_id = int(batch['latents'][index, 0, 0, 0])
            seen.add(row_id)
            assert caption == ds[row_id]['captions'][0]
            assert len(caption) <= batch['bucket_hi']
            assert batch['latents'].shape[2] == resolutions[row_id] + 1
    assert seen == set(range(4))


def make_sampler(seed=17, stage=0.0):
    meta = RowLengthMetadata(resolution_ids=np.array([1, 1]),
        caption_offsets=np.array([0, 2, 4]), prompt_lengths=np.array([100, 1000, 100, 1000]))
    return RowLengthQueueBatchSampler([meta],
        BucketPlan.uniform([1], [LenBucket(256, 3), LenBucket(2048, 2)]),
        seed=seed, initial_stage=stage)


def test_resume_replays_prefetch_then_matches_uninterrupted_draws():
    original = make_sampler(stage=0.75)
    iterator = iter(original)
    for _ in range(10):
        batch = next(iterator)
        original.ack_batch(batch[0].batch_id)
    pending = [next(iterator) for _ in range(3)]
    saved = original.state_dict()
    expected = pending + [next(iterator) for _ in range(20)]
    resumed = make_sampler(seed=999)
    resumed.load_state_dict(saved)
    actual = iter(resumed)
    assert [next(actual) for _ in range(len(expected))] == expected


@pytest.mark.parametrize('progress', [0.0, 0.75, 0.95])
def test_restored_global_progress_matches_sampler_started_at_that_progress(progress):
    restored = make_sampler()
    set_caption_curriculum(restored, global_step=int(progress * 1000), max_steps=1000,
                           curriculum_start=0.0, curriculum_end=1.0)
    actual, expected = iter(restored), iter(make_sampler(stage=progress))
    assert [next(actual) for _ in range(50)] == [next(expected) for _ in range(50)]
