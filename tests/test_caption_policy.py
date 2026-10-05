"""Caption draws checked against probabilities; distributed telemetry against serial recording."""


import numpy as np
import pytest

from src.dataset.captions import CaptionPolicy
from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.sampler import BucketPlan, LenBucket, RowLengthQueueBatchSampler
from src.pretrain.caption_telemetry import CaptionTelemetry


def test_sampling_matches_probabilities():
    # beta=1 gives weights 2:6; a 1/5 reserve for the shorter caption
    # leaves the longer caption with (4/5)*(6/8) = 3/5 probability.
    metadata = RowLengthMetadata(np.array([1]), np.array([0, 2]), np.array([2, 6]))
    sampler = RowLengthQueueBatchSampler(
        [metadata], BucketPlan.uniform([1], [LenBucket(8, 1)]), seed=0,
        caption_policy=CaptionPolicy(beta_start=1, beta_end=1,
                                     short_reserve=0.2, short_threshold=4))
    iterator = iter(sampler)
    draws = []
    for _ in range(20000):
        batch = next(iterator)
        draws.append(batch[0].caption_idx)
        sampler.ack_batch(batch[0].batch_id)
    assert np.mean(draws) == pytest.approx(0.6, abs=0.02)


def test_distributed_telemetry_matches_serial_recording():
    rank_a, rank_b, serial = CaptionTelemetry(), CaptionTelemetry(), CaptionTelemetry()
    batches = [([100] * 10, [True] + [False] * 9, 256, [(0, row) for row in range(10)]),
               ([900] * 30, [True] * 15 + [False] * 15, 1024, [(1, row) for row in range(30)])]
    for rank, (lengths, dropped, width, rows) in zip((rank_a, rank_b), batches):
        rank.record(lengths, dropped, width, row_positions=rows)
        serial.record(lengths, dropped, width, row_positions=rows)
    rank_a.merge_window_counts(rank_b.window_counts())
    assert rank_a.snapshot() == pytest.approx(serial.snapshot())


def test_reduce_preserves_fractional_sums_and_unequal_rank_counts(monkeypatch):
    import torch
    from src.pretrain.caption_telemetry import PolicyState

    local = CaptionTelemetry()
    local.record([2], [False], 8, policy=PolicyState(.25, .125, -.5), loss_weights=[1.25])
    dtypes = []
    # Supply the peer's independent counters at the collective boundary.
    # Peer: three captions of length 6, weight .75, policy (.5, .25, -.25).
    def all_reduce(tensor, op):
        dtypes.append(tensor.dtype)
        if tensor.dtype == torch.int64:
            tensor[:10] += torch.tensor([3, 0, 18, 24, 1, 3, 0, 0, 0, 3])
            tensor[10 + 6] += 3
        else:
            assert tensor.dtype == torch.float32
            tensor += torch.tensor([1.5, .75, -.75, 2.25, 13.5])
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)
    local.reduce(world_size=2)
    assert dtypes == [torch.int64, torch.float32]
    counts = local.window_counts()
    assert counts[:6] == [4, 0, 20, 32, 2, 4]
    assert counts[10:15] == pytest.approx([1.75, .875, -1.25, 3.5, 16])
    metrics = local.snapshot()
    assert metrics['caption/selected_mean_tokens'] == 5



def _reduce_on_rank(rank, init_method):
    from datetime import timedelta
    import torch.distributed as dist
    from src.pretrain.caption_telemetry import PolicyState

    dist.init_process_group("gloo", init_method=init_method, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        counters = CaptionTelemetry()
        if rank == 0:
            counters.record([2], [False], 8, policy=PolicyState(.25, .125, -.5),
                            loss_weights=[1.25])
        else:
            counters.record([6, 6, 6], [False] * 3, 8,
                            policy=PolicyState(.5, .25, -.25), loss_weights=[.75] * 3)
        counters.reduce(world_size=2)
        values = counters.window_counts()
        assert values[:6] == [4, 0, 20, 32, 2, 4]
        assert values[10:15] == pytest.approx([1.75, .875, -1.25, 3.5, 16])
    finally:
        dist.destroy_process_group()


def test_fractional_telemetry_across_real_processes(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_reduce_on_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
