"""Model-internal health metrics, read inline every `health_interval` steps.

Three signals, all passive reads of tensors the training loop already owns:

1. update/weight RMS ratio per optimizer group. Changes can reflect the LR
   schedule, weight scale, or optimization dynamics; drift alone does not
   establish a normalization failure or predict a loss spike.
2. attention QK gain. The blocks apply RMSNorm with learned affine gains to
   q and k. Gain growth can increase logit scale, although alignment of the
   normalized features also changes logits. No extra forward pass is needed.
3. EMA vs live relative weight distance. The two copies separating quickly
   is an early divergence sign, and the EMA copy exists anyway.

These diagnostics do not change optimizer math, but CPU snapshots and device
synchronization add overhead that must be included in infrastructure profiling.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

# (parameter, CPU fp32 copy) taken before the optimizer step.
WeightSnapshot = List[Tuple[torch.nn.Parameter, torch.Tensor]]


def snapshot_weights(optimizers: Sequence[torch.optim.Optimizer]) -> List[WeightSnapshot]:
    """CPU copies of every trainable parameter, grouped per optimizer.

    Called right before ``optimizer.step()``; pairing the copies with the
    post-step weights gives the step's actual update. CPU copies keep the
    probe's transient GPU memory at one parameter at a time.
    """
    snapshots: List[WeightSnapshot] = []
    for optimizer in optimizers:
        group: WeightSnapshot = []
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                group.append(
                    (param, param.detach().to("cpu", torch.float32, copy=True))
                )
        snapshots.append(group)
    return snapshots


def update_weight_ratios(snapshots: Sequence[WeightSnapshot]) -> List[float]:
    """||ΔW||_F / ||W||_F per optimizer group, from a pre-step snapshot."""
    ratios: List[float] = []
    for group in snapshots:
        delta_sq = 0.0
        weight_sq = 0.0
        for param, old in group:
            new = param.detach().to(torch.float32)
            delta_sq += float((new - old.to(new.device)).pow(2).sum())
            weight_sq += float(old.pow(2).sum())
        ratios.append(delta_sq**0.5 / (weight_sq**0.5 + 1e-12))
    return ratios


def qk_gain_stats(model: torch.nn.Module) -> Optional[Tuple[float, float]]:
    """(max, mean) of the learned q/k RMSNorm gains across attention blocks."""
    values: List[torch.Tensor] = []
    for module in model.modules():
        for attr in ("q_norm", "k_norm", "q_norm_img", "k_norm_img",
                     "q_norm_txt", "k_norm_txt"):
            norm = getattr(module, attr, None)
            if isinstance(norm, torch.nn.RMSNorm) and norm.weight is not None:
                values.append(norm.weight.detach().float())
    if not values:
        return None
    flat = torch.cat([v.reshape(-1) for v in values])
    return float(flat.max()), float(flat.mean())


def ema_rel_distance(
    ema_model: torch.nn.Module, model: torch.nn.Module
) -> float:
    """||ema − live||_F / ||live||_F over all parameters."""
    delta_sq = 0.0
    weight_sq = 0.0
    for ema_param, live_param in zip(ema_model.parameters(), model.parameters()):
        live = live_param.detach().to(torch.float32)
        delta_sq += float((ema_param.detach().to(torch.float32) - live).pow(2).sum())
        weight_sq += float(live.pow(2).sum())
    return delta_sq**0.5 / (weight_sq**0.5 + 1e-12)
