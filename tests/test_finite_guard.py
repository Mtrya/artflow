"""Nonfinite state must stop every rank before optimizer mutation."""

from datetime import timedelta
import ast
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.train.finite_guard import require_finite_update


def test_production_guard_precedes_optimizer_updates():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/train/train.py").read_text())
    boundary = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "should_optimizer_step"
                    and "require_finite_update" in ast.unparse(n))
    statements = [ast.unparse(n) for n in boundary.body]
    guard = next(i for i, text in enumerate(statements) if text.startswith("require_finite_update("))
    clip = next(i for i, text in enumerate(statements) if "accelerator.clip_grad_norm_(" in text)
    optimizer = next(i for i, text in enumerate(statements) if text.startswith("for opt in optimizers:"))
    assert clip < guard < optimizer


@pytest.mark.parametrize("loss,norm", [(1., 0.), (0., 5.), (1., 1e10)])
def test_finite_update_allowed(loss, norm):
    require_finite_update(loss, torch.tensor(norm))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["loss", "norm"])
def test_nonfinite_update_rejected(bad, field):
    with pytest.raises(FloatingPointError, match="before optimizer update"):
        require_finite_update(bad if field == "loss" else 1.,
                              torch.tensor(bad if field == "norm" else 1.))


def _rank_guard(rank, init_method, fault):
    dist.init_process_group("gloo", init_method=init_method, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        parameter.grad = torch.tensor([float("inf") if rank == 1 and fault == "grad" else 1.])
        before = parameter.detach().clone()
        norm = torch.nn.utils.clip_grad_norm_([parameter], 1.)
        loss = float("nan") if rank == 1 and fault == "loss" else 1.
        caught = False
        try:
            require_finite_update(loss, norm)
            optimizer.step()
        except FloatingPointError:
            caught = True
        assert caught == (fault != "none")
        if caught:
            assert torch.equal(parameter, before)
            assert not optimizer.state
        else:
            assert not torch.equal(parameter, before)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fault", ["none", "loss", "grad"])
def test_rank_local_failure_stops_all_optimizers(tmp_path, fault):
    mp.spawn(_rank_guard, args=((tmp_path / "rendezvous").as_uri(), fault),
             nprocs=2, join=True)
