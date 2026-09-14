"""Whole-run caption progress survives resolution changes and crash resume."""

import ast
import inspect

import numpy as np
import pytest

from src.dataset.captions import CaptionPolicy
from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.sampler import BucketPlan, LenBucket, RowLengthQueueBatchSampler
from src.train import train


def make_sampler(stage=0.0):
    metadata = RowLengthMetadata(
        resolution_ids=np.array([1, 1]),
        caption_offsets=np.array([0, 2, 4]),
        prompt_lengths=np.array([100, 1000, 100, 1000]),
    )
    return RowLengthQueueBatchSampler(
        [metadata], BucketPlan.uniform([1], [LenBucket(256, 3), LenBucket(2048, 2)]),
        seed=17, initial_stage=stage, caption_policy=CaptionPolicy(kind="beta"),
    )


def set_progress(sampler, step, total, start=0.0, end=1.0):
    train.set_caption_curriculum(
        sampler, global_step=step, max_steps=total,
        curriculum_start=start, curriculum_end=end,
    )


@pytest.mark.parametrize("total", [400_000, 420_000])
@pytest.mark.parametrize("fraction,beta", [(0.0, -1.0), (0.75, 0.5), (0.95, 0.9)])
def test_fresh_sampler_uses_global_curriculum_from_first_draw(total, fraction, beta):
    resumed = make_sampler()
    set_progress(resumed, int(fraction * total), total)
    assert resumed.stage == pytest.approx(fraction)
    assert resumed.caption_policy.beta(resumed.stage) == pytest.approx(beta)
    reference = make_sampler(stage=fraction)
    actual, expected = iter(resumed), iter(reference)
    assert [next(actual) for _ in range(50)] == [next(expected) for _ in range(50)]


def test_same_stage_resume_preserves_queues_replay_and_rng():
    original = make_sampler(stage=0.75)
    iterator = iter(original)
    for _ in range(10):
        batch = next(iterator)
        original.ack_batch(batch[0].batch_id)
    next(iterator)  # A prefetched, unacknowledged batch must be replayed.
    saved = original.state_dict()
    resumed = make_sampler()
    resumed.load_state_dict(saved)
    before = resumed.state_dict()
    set_progress(resumed, 300_000, 400_000)
    assert resumed.state_dict() == before
    reference = make_sampler()
    reference.load_state_dict(saved)
    actual, expected = iter(resumed), iter(reference)
    assert [next(actual) for _ in range(50)] == [next(expected) for _ in range(50)]


def test_custom_curriculum_endpoints_and_zero_step_run():
    sampler = make_sampler()
    set_progress(sampler, 75, 100, start=0.2, end=0.8)
    assert sampler.stage == pytest.approx(0.65)
    set_progress(sampler, 0, 0, start=0.2, end=0.8)
    assert sampler.stage == pytest.approx(0.2)


def test_training_initializes_curriculum_before_telemetry_and_prefetch():
    """Guard the actual wiring: DataLoader iteration can draw ahead immediately."""
    tree = ast.parse(inspect.getsource(train))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    updates = sorted(node.lineno for node in calls
                     if isinstance(node.func, ast.Name)
                     and node.func.id == "set_caption_curriculum")
    telemetry = min(node.lineno for node in calls
                    if isinstance(node.func, ast.Name)
                    and node.func.id == "current_policy_state")
    prefetch = min(node.lineno for node in calls
                   if isinstance(node.func, ast.Name) and node.func.id == "iter"
                   and node.args and isinstance(node.args[0], ast.Name)
                   and node.args[0].id == "dataloader")
    restored_step = next(node.lineno for node in ast.walk(tree)
                         if isinstance(node, ast.Assign)
                         and isinstance(node.value, ast.Name)
                         and node.value.id == "resumed_step")
    assert len(updates) == 2  # Initialization and ongoing optimizer-step updates.
    assert restored_step < updates[0] < telemetry < prefetch < updates[1]


@pytest.mark.parametrize("saved,current", [(1, 1), (2, 2), (1, 2), (2, 1), (0, 1)])
def test_resume_requires_exact_sampler_rank_set(tmp_path, saved, current):
    for rank in range(saved):
        (tmp_path / f"sampler_state_rank_{rank:05d}.pt").touch()
    (tmp_path / "sampler_state_rank_00007.pt.tmp").touch()
    paths = train.matching_sampler_sidecars(str(tmp_path), current)
    assert len(paths) == (current if saved == current else 0)
