import json

import pytest
import torch

from scripts.bench.infra_resume_smoke import check_saved_state, check_updates


def test_update_interval_requires_every_rank_and_finite_loss(tmp_path):
    (tmp_path / "infra").mkdir()
    for rank in range(2):
        (tmp_path / "infra" / f"rank-{rank}.jsonl").write_text("\n".join(
            json.dumps(dict(step=step, rank=rank, loss=0.5)) for step in [65, 66]))
    check_updates(tmp_path, start=64, end=66, ranks=2)
    path = tmp_path / "infra" / "rank-1.jsonl"
    path.write_text(json.dumps(dict(step=65, rank=1, loss=0.5)))
    with pytest.raises(ValueError, match="interval"):
        check_updates(tmp_path, start=64, end=66, ranks=2)
    path.write_text("\n".join(json.dumps(dict(step=s, rank=1, loss=float("nan")))
                              for s in [65, 66]))
    with pytest.raises(ValueError, match="nonfinite"):
        check_updates(tmp_path, start=64, end=66, ranks=2)


@pytest.mark.parametrize("fault", [None, "scheduler", "optimizer", "ema", "step"])
def test_saved_state_checks_real_serialized_optimizer_scheduler_and_ema(tmp_path, fault):
    for index in range(2):
        suffix = "" if index == 0 else "_1"
        torch.save(dict(last_epoch=65 if fault == "scheduler" else 66),
                   tmp_path / f"scheduler{suffix}.bin")
        state = dict(momentum_buffer=torch.ones(3)) if index == 0 else dict(
            step=torch.tensor(65 if fault == "step" else 66), exp_avg=torch.ones(3))
        if fault == "optimizer":
            state["bad"] = torch.tensor(float("inf"))
        torch.save(dict(state={0: state}), tmp_path / f"optimizer{suffix}.bin")
    torch.save(dict(weight=torch.tensor(float("nan") if fault == "ema" else 1.0)),
               tmp_path / "ema_weights.pt")
    if fault:
        with pytest.raises(ValueError):
            check_saved_state(tmp_path, 66)
    else:
        check_saved_state(tmp_path, 66)
