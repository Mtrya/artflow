"""Coordinated failure before any optimizer can consume nonfinite state."""

import torch
import torch.distributed as dist


@torch.no_grad()
def require_finite_update(loss: float | torch.Tensor, grad_norm: torch.Tensor) -> None:
    """All ranks raise if any rank has nonfinite loss or pre-clip gradient norm.

    Call at the shared optimizer boundary, after gradient clipping returns its
    pre-clip norm and before *any* optimizer, scheduler or EMA update. Checking
    that norm catches NaN/Inf gradients and norm overflow without scanning every
    parameter again. The collective is unconditional so a rank-local failure
    cannot leave peers updating weights or waiting in a later collective.
    """
    loss_tensor = torch.as_tensor(loss, device=grad_norm.device)
    failed = (~(torch.isfinite(loss_tensor) & torch.isfinite(grad_norm))).to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise FloatingPointError(
            "Nonfinite loss or gradient norm on at least one rank; "
            "stopping all ranks before optimizer update. Resume only from a "
            "previous verified checkpoint after reviewing the failure."
        )
