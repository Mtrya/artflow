"""Row sampling checked against source probabilities, input rows and uninterrupted draws."""

from collections import Counter

import numpy as np
import pytest
from datasets import Dataset

from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.captions import CaptionPolicy
from src.dataset.sampler import BucketPlan, LenBucket, RowDescriptorDataset, RowLengthQueueBatchSampler, row_length_collate_fn
from src.pretrain.train import set_caption_curriculum


def metadata(resolutions, lengths):
    return RowLengthMetadata(resolution_ids=np.asarray(resolutions),
        caption_offsets=np.arange(len(lengths) + 1), prompt_lengths=np.asarray(lengths))


def test_source_probability_is_independent_of_dataset_size():
    entries = [metadata([1] * n, [2] * n) for n in (2, 100)]
    sampler = RowLengthQueueBatchSampler(entries, BucketPlan.uniform([1], [LenBucket(8, 1)]),
        dataset_weights=[3, 1], seed=47, caption_policy=CaptionPolicy(beta_start=-0.5, beta_end=1.5, short_reserve=0.1, short_threshold=200))
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
    sampler = RowLengthQueueBatchSampler([meta], plan, seed=1, caption_policy=CaptionPolicy(beta_start=-0.5, beta_end=1.5, short_reserve=0.1, short_threshold=200))
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
        seed=seed, initial_stage=stage, caption_policy=CaptionPolicy(beta_start=-0.5, beta_end=1.5, short_reserve=0.1, short_threshold=200))


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


def test_resume_rejects_appended_dataset():
    entries = [metadata([1] * 8, [2] * 8), metadata([1] * 8, [2] * 8)]
    plan = BucketPlan.uniform([1], [LenBucket(8, 1)])
    original = RowLengthQueueBatchSampler(entries, plan, dataset_weights=[1, 1], seed=47, caption_policy=CaptionPolicy(beta_start=-0.5, beta_end=1.5, short_reserve=0.1, short_threshold=200))
    iterator = iter(original)
    for _ in range(5):
        batch = next(iterator)
        original.ack_batch(batch[0].batch_id)
    [next(iterator) for _ in range(2)]
    saved = original.state_dict()

    resumed = RowLengthQueueBatchSampler(entries + [metadata([1] * 8, [2] * 8)], plan,
        dataset_weights=[1, 1, 1], seed=999, caption_policy=CaptionPolicy(beta_start=-0.5, beta_end=1.5, short_reserve=0.1, short_threshold=200))
    with pytest.raises(ValueError, match="state metadata entries"):
        resumed.load_state_dict(saved)


def test_load_state_dict_rejects_state_with_more_datasets_than_sampler():
    saved = make_sampler().state_dict()
    shrunk = RowLengthQueueBatchSampler([make_sampler().metadata[0]],
        BucketPlan.uniform([1], [LenBucket(256, 3), LenBucket(2048, 2)]), seed=1, caption_policy=CaptionPolicy(beta_start=-0.5, beta_end=1.5, short_reserve=0.1, short_threshold=200))
    # A saved state listing more cycles than the sampler has datasets can only
    # come from a removed/reordered mixture, which is not a supported amendment.
    with pytest.raises(ValueError, match="state metadata entries"):
        shrunk.load_state_dict({**saved, "cycles": saved["cycles"] + [[]],
                                "cursors": saved["cursors"] + [0]})


@pytest.mark.parametrize('field', ['inflight', 'replay', 'ready_batches'])
def test_saved_state_requires_pending_work_fields(field):
    sampler = make_sampler()
    state = sampler.state_dict()
    del state[field]
    with pytest.raises(ValueError, match='missing fields'):
        sampler.load_state_dict(state)


def test_stored_row_requires_batch_id_but_unbatched_queue_rows_keep_sentinel():
    from src.dataset.sampler import RowRef
    ref = RowRef(0, 0, 0, 1, 3, 0, 8)
    assert RowRef.from_state((0, 0, 0, 1, 3, 0, 8, -1)) == ref
    with pytest.raises(ValueError, match='eight fields'):
        RowRef.from_state((0, 0, 0, 1, 3, 0, 8))


@pytest.mark.parametrize("progress", [-0.1, 1.1, float("nan")])
def test_resume_rejects_invalid_progress_instead_of_clamping(progress):
    sampler = make_sampler()
    state = sampler.state_dict()
    state["stage"] = progress
    with pytest.raises(ValueError, match="invalid saved sampler progress"):
        sampler.load_state_dict(state)
