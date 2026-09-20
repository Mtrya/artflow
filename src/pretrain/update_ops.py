"""Opt-in multi-tensor equivalents for pointwise training-state updates."""

from collections import defaultdict

import torch


@torch.no_grad()
def divide_gradients(parameters, divisor, *, foreach=False):
    groups = defaultdict(list)
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        if not foreach or grad.is_sparse:
            grad.div_(divisor)
        else:
            groups[(grad.device, grad.dtype)].append(grad)
    for tensors in groups.values():
        torch._foreach_div_(tensors, divisor)


@torch.no_grad()
def update_ema(ema_model, model, decay, *, foreach=False):
    groups = defaultdict(lambda: ([], []))
    for ema, live in zip(ema_model.parameters(), model.parameters()):
        if not foreach or ema.device != live.device or ema.dtype != live.dtype:
            ema.mul_(decay).add_(live, alpha=1.0 - decay)
        else:
            targets, sources = groups[(ema.device, ema.dtype)]
            targets.append(ema)
            sources.append(live)
    for targets, sources in groups.values():
        # Preserve the reference's two operations and their rounding order;
        # lerp or a fused multiply-add would be a different numerical path.
        torch._foreach_mul_(targets, decay)
        torch._foreach_add_(targets, sources, alpha=1.0 - decay)
    for ema, live in zip(ema_model.buffers(), model.buffers()):
        ema.copy_(live)


def clear_local_cuda_cache(device):
    """Empty only this rank's cache; never create contexts on peer GPUs."""
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
