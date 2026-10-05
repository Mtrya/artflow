"""Checkpoint validation and recovery checked against saved state and uninterrupted schedules."""

import copy
import json
import random
import numpy as np

import pytest
import torch

from src.pretrain import train
from src.pretrain.stage_control import CHECKPOINT_RECORD, validate_checkpoint, write_checkpoint_record, verify_restored_rng


def checkpoint(tmp_path, step=300000, total=400000):
    root = tmp_path / f"checkpoint_step_{step:06d}"
    root.mkdir()
    for name in ("model.safetensors", "optimizer.bin", "optimizer_1.bin", "scheduler.bin",
                 "scheduler_1.bin", "ema_weights.pt", "sampler_state_rank_00000.pt"):
        (root / name).write_bytes(b"test-artifact")
    for rank in range(8):
        for name in (f"random_states_{rank}.pkl", f"sampler_state_rank_{rank:05d}.pt"):
            (root / name).write_bytes(b"test-rank-state")
    write_checkpoint_record(root, step=step, max_steps=total, scheduler_count=2,
                            use_ema=True, world_size=8)
    return root


def validate(root, **kwargs):
    return validate_checkpoint(root, max_steps=400000, stop_at_step=380000,
                               scheduler_count=2, use_ema=True,
                               world_size=8, **kwargs)


@pytest.mark.parametrize("fault", ["record", "horizon", "step", "scheduler", "truncated", "ema", "world"])
def test_incomplete_or_incompatible_checkpoints_rejected(tmp_path, fault):
    root = checkpoint(tmp_path)
    assert validate(root) == 300000
    record = root / CHECKPOINT_RECORD
    if fault == "record":
        record.unlink()
    elif fault in ("scheduler", "ema"):
        (root / ("scheduler_1.bin" if fault == "scheduler" else "ema_weights.pt")).unlink()
    elif fault == "truncated":
        (root / "optimizer.bin").write_bytes(b"x")
    else:
        data = json.loads(record.read_text())
        data[{"horizon": "max_steps", "step": "global_step", "world": "world_size"}[fault]] += 1
        record.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate(root)


@pytest.mark.parametrize("fault", [None, "python", "numpy", "torch", "corrupt"])
def test_strict_rng_verification_detects_silent_load_failure(tmp_path, fault):
    saved = dict(random_state=random.getstate(), numpy_random_seed=np.random.get_state(),
                 torch_manual_seed=torch.get_rng_state())
    path = tmp_path / "random_states_0.pkl"
    torch.save(saved, path)
    try:
        if fault == "python":
            random.random()
        elif fault == "numpy":
            np.random.random()
        elif fault == "torch":
            torch.rand(1)
        elif fault == "corrupt":
            path.write_bytes(b"invalid checkpoint")
        if fault:
            with pytest.raises(ValueError, match="RNG continuation failed"):
                verify_restored_rng(tmp_path, process_index=0, device="cpu")
        else:
            verify_restored_rng(tmp_path, process_index=0, device="cpu")
    finally:
        random.setstate(saved["random_state"])
        np.random.set_state(saved["numpy_random_seed"])
        torch.set_rng_state(saved["torch_manual_seed"])


def test_scheduler_roundtrip_preserves_learning_rates():
    def setup():
        param = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([param], lr=.02)
        sch = train.build_linear_cosine_scheduler(opt, num_warmup_steps=5,
                num_training_steps=100, min_learning_rate=.001,
                base_learning_rate=.02, start_learning_rate=.0001)
        return opt, sch
    opt, sch = setup()
    for _ in range(75):
        opt.step()
        sch.step()
    state, optim_state = copy.deepcopy(sch.state_dict()), copy.deepcopy(opt.state_dict())
    resumed_opt, resumed_sch = setup()
    resumed_opt.load_state_dict(optim_state)
    resumed_sch.load_state_dict(state)
    for _ in range(20):
        opt.step(); sch.step()
        resumed_opt.step(); resumed_sch.step()
        assert resumed_sch.get_last_lr() == sch.get_last_lr()
    assert resumed_sch.last_epoch == 95
    assert sch.get_last_lr()[0] > .001  # The schedule has not ended at 95%.
