"""
Muon optimizer (MomentUm Orthogonalized by Newton-Schulz) with chunked
orthogonalization for fused matrices.

Adapted from Keller Jordan's reference implementation
(https://github.com/KellerJordan/Muon) with two changes:

1. Chunked orthogonalization (CMuon, arXiv:2608.02502): DiTs fuse functionally
   distinct weights into single tensors (fused QKV, 6xdim AdaLN modulation,
   gated-FFN up projections). Orthogonalizing the fused tensor couples the
   subspaces and causes a late-stage convergence plateau. Groups can carry a
   `chunks` hint; the momentum/grad is split into that many row-chunks and each
   chunk is orthogonalized independently.

2. Update scaling follows Moonlight (arXiv:2502.16982): the orthogonalized
   update has RMS 1/sqrt(max(m, n)); scaling by 0.2*sqrt(max(m, n)) matches the
   typical AdamW update RMS (~0.2), so Muon LRs in the 0.01-0.05 range behave
   intuitively. Decoupled weight decay uses the base LR.

Param routing convention (see build_param_groups):
- Muon: 2D hidden weights (attention projections, FFN, modulation/QKV with
  chunk hints).
- AdamW (separate optimizer): embeddings, patch conv, final layer, norms,
  biases, timestep/conditioning MLPs, and anything with ndim != 2.

DDP-safe: gradients are identical across ranks after all-reduce and
Newton-Schulz is deterministic, so all ranks compute identical updates.
"""

import math
from typing import List, Optional

import torch
from torch import nn


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G via quintic Newton-Schulz iteration (bf16 internally)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    # Normalize so the spectral norm is <= 1 before iterating.
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon for 2D hidden-layer weights.

    Param group fields beyond the standard ones:
        chunks (int): split the [m, n] matrix into this many row chunks and
            orthogonalize each independently (1 = no chunking).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            chunks=1,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            chunks = group["chunks"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - momentum)
                g = g.lerp_(buf, momentum) if group["nesterov"] else buf

                m, n = p.shape
                if chunks > 1:
                    assert m % chunks == 0, f"chunks={chunks} does not divide {m}"
                    gs = g.reshape(chunks, m // chunks, n)
                    updated = torch.cat(
                        [
                            _zeropower_via_newtonschulz5(gs[i], group["ns_steps"])
                            for i in range(chunks)
                        ],
                        dim=0,
                    )
                    scale = 0.2 * math.sqrt(max(m // chunks, n))
                else:
                    updated = _zeropower_via_newtonschulz5(g, group["ns_steps"])
                    scale = 0.2 * math.sqrt(max(m, n))

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(updated.to(p.dtype), alpha=-lr * scale)

        return loss


def _chunk_hint(name: str, shape: torch.Size) -> int:
    """Row-chunk count for fused matrices (CMuon). 1 = treat as a single matrix."""
    if "qkv" in name and shape[0] == 3 * shape[1]:
        return 3
    if "modulation" in name and shape[0] % shape[1] == 0 and shape[0] > shape[1]:
        return shape[0] // shape[1]  # 6xdim -> 6, 3xdim -> 3
    if "up_proj" in name and shape[0] == 2 * ((shape[0]) // 2) and shape[0] > shape[1]:
        # GatedFeedForward fused gate|linear projection -> 2 chunks
        return 2
    return 1


def build_param_groups(
    model: nn.Module,
    muon_lr: float,
    muon_wd: float = 0.01,
    adam_lr: float = 3e-4,
    adam_wd: float = 0.01,
    adam_betas=(0.9, 0.95),
    muon_momentum: float = 0.95,
) -> List[torch.optim.Optimizer]:
    """
    Split model parameters into Muon (2D hidden) and AdamW (everything else)
    groups and return [muon_optimizer, adamw_optimizer].

    AdamW-routed: embeddings (x/txt), patch conv, final layer, timestep and
    conditioning MLPs (t_embedder has no params; c_mlp/txt_pooled_proj are
    small conditioning heads), all norms and biases, and the MSRoPE buffers
    never appear here (no grad).
    """
    adam_name_patterns = (
        "x_embedder",
        "txt_embedder",
        "txt_pooled_proj",
        "c_mlp",
        "final_layer",
    )

    muon_groups: dict[int, dict] = {}
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_adam = (
            p.ndim != 2
            or any(pat in name for pat in adam_name_patterns)
        )
        if is_adam:
            adam_params.append(p)
            continue
        chunks = _chunk_hint(name, p.shape)
        if chunks not in muon_groups:
            muon_groups[chunks] = {"params": [], "chunks": chunks}
        muon_groups[chunks]["params"].append(p)

    optimizers: List[torch.optim.Optimizer] = []
    if muon_groups:
        optimizers.append(
            Muon(
                [muon_groups[c] for c in sorted(muon_groups)],
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_wd,
            )
        )
    if adam_params:
        optimizers.append(
            torch.optim.AdamW(
                adam_params, lr=adam_lr, weight_decay=adam_wd, betas=adam_betas
            )
        )
    return optimizers
