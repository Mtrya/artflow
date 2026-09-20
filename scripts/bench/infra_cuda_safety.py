"""Bounded torchrun CUDA/NCCL smoke for infra primitives, not full model readiness.

Each rank injects the same test sequence; only the last rank sees a nonfinite
input. Every rank must refuse the update. Also compare opt-in pointwise update
primitives against their reference on actual CUDA tensors.
"""

import copy
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist

from src.train.finite_guard import require_finite_update
from src.train.update_ops import divide_gradients, update_ema


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    device = torch.device("cuda", local_rank)
    rank, world = dist.get_rank(), dist.get_world_size()
    checks = {}
    try:
        for fault in ("none", "loss", "gradient"):
            param = torch.nn.Parameter(torch.ones(8, device=device))
            optimizer = torch.optim.SGD([param], lr=.1)
            param.grad = torch.ones_like(param)
            loss = torch.tensor(1., device=device)
            norm = torch.tensor(1., device=device)
            if rank == world - 1 and fault != "none":
                (loss if fault == "loss" else norm).fill_(float("nan"))
            stopped = False
            try:
                require_finite_update(loss, norm)
                optimizer.step()
            except FloatingPointError:
                stopped = True
            unchanged = torch.equal(param, torch.ones_like(param))
            checks[f"finite_guard_{fault}"] = (
                not stopped and not unchanged if fault == "none" else stopped and unchanged)

        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(42)
            reference = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Linear(64, 8))
            reference.to(device=device, dtype=dtype)
            candidate = copy.deepcopy(reference)
            ema_ref, ema_candidate = copy.deepcopy(reference), copy.deepcopy(candidate)
            for _ in range(5):
                with torch.no_grad():
                    for p, q in zip(reference.parameters(), candidate.parameters()):
                        p.add_(torch.randn_like(p) * .01)
                        q.copy_(p)
                        p.grad = torch.randn_like(p)
                        q.grad = p.grad.clone()
                divide_gradients(reference.parameters(), 37.5)
                divide_gradients(candidate.parameters(), 37.5, foreach=True)
                update_ema(ema_ref, reference, .9999)
                update_ema(ema_candidate, candidate, .9999, foreach=True)
            checks[f"gradients_exact_{dtype}"] = all(
                torch.equal(p.grad, q.grad) for p, q in zip(reference.parameters(), candidate.parameters()))
            checks[f"ema_exact_{dtype}"] = all(
                torch.equal(p, q) for p, q in zip(ema_ref.parameters(), ema_candidate.parameters()))
        failed = torch.tensor(int(not all(checks.values())), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        print(json.dumps(dict(rank=rank, world_size=world, checks=checks,
                              all_ranks_pass=not bool(failed.item()))), flush=True)
        if failed.item():
            raise RuntimeError("infra CUDA primitive smoke failed on at least one rank")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
