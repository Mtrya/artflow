"""A nonfinite loss or gradient on one rank must prevent updates on every rank."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.pretrain.finite_guard import require_finite_update


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
