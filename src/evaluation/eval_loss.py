"""
Fixed held-out eval-loss probe for ablation comparability.

Replaces the old rotating-subset approach: the same samples, captions, noise,
and timesteps at every probe and in every arm, so eval/loss curves are directly
comparable across runs. Forward-only; runs on every rank (deterministic, acts
as a natural sync point) and is logged by the main process.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..flow.paths import shift_timesteps
from ..utils.encode_text import encode_text


def _autocast_ctx(device: torch.device):
    """Match training-time numerics (accelerate bf16 mixed precision)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


class EvalLossProbe:
    """
    Fixed-set flow-matching loss probe.

    Holds a fixed slice of the eval dataset: normalized latents, pre-encoded
    text (deterministic caption choice, no dropout), per-sample fixed noise,
    and a fixed timestep grid.
    """

    def __init__(
        self,
        dataset_path: str,
        text_encoder: Any,
        tokenizer: Any,
        pooling: bool,
        exit_layer: Optional[int],
        vae_mean: torch.Tensor,
        vae_std: torch.Tensor,
        num_samples: int = 512,
        batch_size: int = 64,
        t_grid: Tuple[float, ...] = (0.15, 0.4, 0.65, 0.9),
        seed: int = 123,
        device: Optional[torch.device] = None,
    ):
        from datasets import load_from_disk

        if device is None:
            device = next(text_encoder.parameters()).device
        self.device = device
        self.batch_size = batch_size
        self.t_grid = tuple(t_grid)

        dataset = load_from_disk(dataset_path).shuffle(seed=seed)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        captions: List[str] = []
        latents: List[torch.Tensor] = []
        for item in dataset:
            caps = item["captions"]
            captions.append(caps[1] if len(caps) > 1 else caps[0])
            latents.append(torch.as_tensor(item["latents"]).float())

        # Group by latent shape: buckets have different HxW and cannot be stacked
        # into one batch. Groups preserve the fixed caption/noise alignment.
        shape_groups: Dict[Tuple[int, int], List[int]] = {}
        for idx, z in enumerate(latents):
            shape_groups.setdefault((z.shape[-2], z.shape[-1]), []).append(idx)
        self.groups = list(shape_groups.values())

        # Normalize latents exactly like training: z = (z - mean) / std
        mean = vae_mean.float().cpu().squeeze()
        std = vae_std.float().cpu().squeeze()
        self.latents = [(z - mean.view(-1, 1, 1)) / std.view(-1, 1, 1) for z in latents]

        # Pre-encode text once (fixed captions; identical across probes/arms)
        txt, txt_mask, txt_pooled = encode_text(
            captions, text_encoder, tokenizer, pooling, exit_layer=exit_layer
        )
        self.txt = txt.cpu()
        self.txt_mask = txt_mask.cpu()
        self.txt_pooled = txt_pooled.cpu() if txt_pooled is not None else None

        # Fixed noise per sample (same across probes and arms)
        gen = torch.Generator().manual_seed(seed)
        self.noise = [torch.randn(z.shape, generator=gen) for z in self.latents]

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> Dict[str, float]:
        was_training = model.training
        model.eval()

        per_t_losses: List[List[float]] = [[] for _ in self.t_grid]

        for group in self.groups:
            for gstart in range(0, len(group), self.batch_size):
                idxs = group[gstart : gstart + self.batch_size]
                z1 = torch.stack([self.latents[i] for i in idxs]).to(
                    self.device, torch.bfloat16
                )
                z0 = torch.stack([self.noise[i] for i in idxs]).to(
                    self.device, torch.bfloat16
                )
                txt = self.txt[idxs].to(self.device)
                txt_mask = self.txt_mask[idxs].to(self.device)
                txt_pooled = (
                    self.txt_pooled[idxs].to(self.device)
                    if self.txt_pooled is not None
                    else None
                )
                bs = z1.shape[0]

                for ti, t_val in enumerate(self.t_grid):
                    t = torch.full((bs,), t_val, device=self.device)
                    t = shift_timesteps(t, z1)
                    z_t = (1.0 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z1
                    with _autocast_ctx(self.device):
                        out = model(
                            z_t, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask
                        )
                    loss = torch.nn.functional.mse_loss(
                        out.float(), (z1 - z0).float(), reduction="mean"
                    )
                    per_t_losses[ti].append(loss.item())

        if was_training:
            model.train()

        metrics: Dict[str, float] = {}
        all_losses = [v for losses in per_t_losses for v in losses]
        metrics["eval/loss"] = sum(all_losses) / max(len(all_losses), 1)
        for ti, t_val in enumerate(self.t_grid):
            losses = per_t_losses[ti]
            tag = f"{t_val:.2f}".replace(".", "").ljust(3, "0")
            metrics[f"eval/loss_t{tag}"] = sum(losses) / max(len(losses), 1)
        return metrics
