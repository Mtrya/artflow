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

2. Update scaling follows Moonlight (arXiv:2502.16982), matching PyTorch Muon's
   `match_rms_adamw` convention. An ideal full-rank orthogonalized update has
   RMS 1/sqrt(max(m, n)); scaling each chunk by 0.2*sqrt(max(m, n)) targets
   update RMS ~0.2 before LR multiplication. Finite NS iterations make this
   approximate. This convention does not prescribe the optimal base LR or
   guarantee architecture-independent behavior. Decay uses the base LR.

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


@torch.no_grad()
def _zeropower_via_newtonschulz5_batched(G: torch.Tensor, steps: int = 5):
    """Batched Newton-Schulz over a stack [N, r, c] of same-shaped matrices.

    Same iteration as the 2D version, but the per-matrix GEMMs become one bmm
    so the GPU is not left waiting on a chain of small kernels. Returns a list
    of [r, c] tensors.
    """
    assert G.ndim == 3
    X = G.to(torch.bfloat16)
    transposed = X.shape[1] > X.shape[2]
    if transposed:
        X = X.mT
    norms = X.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1)
    X = X / norms
    X = _newtonschulz5_batched_iterations(X, steps)
    if transposed:
        X = X.mT
    return list(X.unbind(0))


def _newtonschulz5_batched_iterations(X: torch.Tensor, steps: int = 5):
    """Iteration-only compiler boundary; preserve eager norm reduction order."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
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
        compile_square_ns: bool = False,
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
        # Execution policy, not optimizer state: checkpoint loading must not
        # silently replace the launch's measured kernel selection.
        self.compile_square_ns = compile_square_ns
        self._compiled_square_ns = None

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

            # Pass 1: momentum/nesterov (elementwise) and collect the matrices
            # that need orthogonalization. Weight decay is applied here because
            # it is independent of the NS result and must run once per param,
            # not once per chunk.
            entries = []
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

                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                m, n = p.shape
                if chunks > 1:
                    assert m % chunks == 0, f"chunks={chunks} does not divide {m}"
                    rows = m // chunks
                    gs = g.reshape(chunks, rows, n)
                    scale = 0.2 * math.sqrt(max(rows, n))
                    for index in range(chunks):
                        entries.append(
                            (p.narrow(0, index * rows, rows), gs[index], scale)
                        )
                else:
                    entries.append((p, g, 0.2 * math.sqrt(max(m, n))))

            if group.get("batched_ns", True) and len(entries) > 1:
                # Pass 2: one bmm chain per distinct matrix shape. Same NS math
                # per matrix, without a serial chain of tiny GEMM launches.
                by_shape = {}
                for entry in entries:
                    by_shape.setdefault(tuple(entry[1].shape), []).append(entry)
                for items in by_shape.values():
                    stacked = torch.stack([item[1] for item in items])
                    if self.compile_square_ns and stacked.shape[-2] == stacked.shape[-1]:
                        # The target-runtime probe found exact, faster square
                        # batches initially; a full-update probe subsequently
                        # found rare drift. Keep norm reduction eager, since
                        # cast emulation does not fix reduction ordering.
                        # Rectangular batches stay entirely eager.
                        if self._compiled_square_ns is None:
                            self._compiled_square_ns = torch.compile(
                                _newtonschulz5_batched_iterations,
                                fullgraph=True, dynamic=False,
                                options={"emulate_precision_casts": True,
                                         "shape_padding": False},
                            )
                        X = stacked.to(torch.bfloat16)
                        norms = X.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1)
                        updates = list(self._compiled_square_ns(
                            X / norms, group["ns_steps"]
                        ).unbind(0))
                    else:
                        updates = _zeropower_via_newtonschulz5_batched(
                            stacked, group["ns_steps"]
                        )
                    for (dest, _, scale), updated in zip(items, updates):
                        dest.add_(updated.to(dest.dtype), alpha=-lr * scale)
                continue

            for dest, matrix, scale in entries:
                updated = _zeropower_via_newtonschulz5(matrix, group["ns_steps"])
                dest.add_(updated.to(dest.dtype), alpha=-lr * scale)

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
    batched_ns: bool = True,
    compile_square_ns: bool = False,
    fused_adamw: bool = False,
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
            muon_groups[chunks] = {
                "params": [],
                "chunks": chunks,
                "batched_ns": batched_ns,
            }
        muon_groups[chunks]["params"].append(p)

    optimizers: List[torch.optim.Optimizer] = []
    if muon_groups:
        optimizers.append(
            Muon(
                [muon_groups[c] for c in sorted(muon_groups)],
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_wd,
                compile_square_ns=compile_square_ns,
            )
        )
    if adam_params:
        adamw_cls = torch.optim.AdamW
        if fused_adamw:
            # torch_npu's fused AdamW runs the whole update as one kernel
            # instead of ~6 small ops per parameter — a large host-dispatch
            # saving on NPU, where Python dispatch dominates. Same AdamW
            # math and per-param state layout as torch.optim.AdamW.
            from torch_npu.optim import NpuFusedAdamW

            adamw_cls = NpuFusedAdamW
        optimizers.append(
            adamw_cls(
                adam_params, lr=adam_lr, weight_decay=adam_wd, betas=adam_betas
            )
        )
    return optimizers
