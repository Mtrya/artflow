"""Retention decisions checked against complete and damaged checkpoint fixtures."""

import pytest

from src.pretrain.checkpoint_retention import prune_checkpoints
from src.pretrain.stage_control import CHECKPOINT_RECORD, write_checkpoint_record


CONTRACT = dict(max_steps=600000, scheduler_count=2, use_ema=True, world_size=4)


def saved(root, step, *, complete=True, horizon=600000):
    path = root / f"checkpoint_step_{step:06d}"
    path.mkdir(parents=True)
    files = ["model.safetensors", "ema_weights.pt", "optimizer.bin", "optimizer_1.bin",
             "scheduler.bin", "scheduler_1.bin"]
    files += [f"random_states_{rank}.pkl" for rank in range(4)]
    files += [f"sampler_state_rank_{rank:05d}.pt" for rank in range(4)]
    for name in files:
        (path / name).write_bytes(b"saved-state")
    if complete:
        write_checkpoint_record(path, step=step, **dict(CONTRACT, max_steps=horizon))
    return path


def prune(path, keep_last=3):
    return prune_checkpoints(path, keep_last=keep_last, **CONTRACT)


def test_keep_latest_three_complete_without_touching_incomplete_or_future(tmp_path):
    paths = [saved(tmp_path, step) for step in (2000, 4000, 6000, 8000, 10000)]
    incomplete = saved(tmp_path, 7000, complete=False)
    future = saved(tmp_path, 12000)
    assert prune(paths[-1]) == ["checkpoint_step_002000", "checkpoint_step_004000"]
    assert all(p.exists() for p in paths[-3:])
    assert incomplete.exists() and future.exists()


@pytest.mark.parametrize("fault", ["missing_record", "truncated", "wrong_horizon"])
def test_unusable_replacement_never_prunes_previous_checkpoints(tmp_path, fault):
    old = [saved(tmp_path, step) for step in (2000, 4000, 6000)]
    current = saved(tmp_path, 8000, horizon=480000 if fault == "wrong_horizon" else 600000)
    if fault == "missing_record":
        (current / CHECKPOINT_RECORD).unlink()
    elif fault == "truncated":
        (current / "optimizer.bin").write_bytes(b"x")
    with pytest.raises(ValueError):
        prune(current)
    assert all(p.exists() for p in old)


def test_other_stage_endpoints_and_symlinks_are_untouched(tmp_path):
    predecessor = saved(tmp_path / "256p", 450000)
    other = saved(tmp_path / "other", 1000)
    current_root = tmp_path / "640p"
    paths = [saved(current_root, step) for step in (452000, 454000, 456000, 458000)]
    link = current_root / "checkpoint_step_001000"
    link.symlink_to(other, target_is_directory=True)
    prune(paths[-1])
    assert predecessor.exists() and other.exists() and link.is_symlink()
    assert not paths[0].exists()


def test_incompatible_old_checkpoint_does_not_count_as_a_recovery_copy(tmp_path):
    old = saved(tmp_path, 2000)
    incompatible = saved(tmp_path, 4000, horizon=480000)
    current = saved(tmp_path, 6000)
    assert prune(current, keep_last=2) == []
    assert old.exists() and incompatible.exists()
