"""
Flow matching and score matching algorithms for generative modeling

This module implements three training algorithms for ablation:
1. ScoreMatchingDiffusion - Baseline (predict score, VP-SDE diffusion path)
2. FlowMatchingDiffusion - Flow matching with diffusion path (predict velocity)
3. FlowMatchingOT - Flow matching with optimal transport path (predict velocity)

Each algorithm encapsulates both the probability path and loss computation.
"""

import math

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

# SD3-style resolution-dependent time shift.
# SD3 (Esser et al., 2024, Eq. 23) derives the shift from matching the
# uncertainty about the clean image across resolutions:
#     alpha = sqrt(m / n),
# where n and m are the pixel counts of the reference and target resolutions.
# With a fixed VAE and patch size, token count is proportional to pixel count,
# so the same ratio applies to token counts. Reference: 256 tokens (256×256
# image → 32×32 latent, patch_size=2 → 16×16 patches) gets shift 1.0.
# SD3 empirically found little quality difference among shift values above
# 1.5 and used 3.0 at 1024×1024, where the formula gives 4.0 — so the formula
# values used here sit inside the range they found acceptable.
_SHIFT_BASE_TOKENS = 256      # 256×256 → 32×32 latent → 16×16 patches


def resolution_time_shift(z: torch.Tensor, patch_size: int = 2) -> float:
    """Compute SD3-style time shift from a noise/latent tensor [B, C, H, W]."""
    _, _, h, w = z.shape
    n_tokens = (h * w) / (patch_size ** 2)
    if n_tokens <= _SHIFT_BASE_TOKENS:
        return 1.0
    return math.sqrt(n_tokens / _SHIFT_BASE_TOKENS)


def apply_time_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the SD3 shift transform in this repo's timestep convention.

    SD3 (Esser et al., 2024, Eq. 23) uses t=0 for data and t=1 for noise and
    pushes timesteps toward the noisy end at higher resolutions:
        u' = (s * u) / (1 + (s - 1) * u).
    This repo uses the opposite convention (t=0 noise, t=1 data), so the same
    transform applied to u = 1 - t yields:
        t' = t / (s - (s - 1) * t),
    which decreases t (moves toward noise) when shift > 1.
    """
    return t / (shift - (shift - 1) * t)


def shift_timesteps(
    t: torch.Tensor,
    z_like: torch.Tensor,
    *,
    patch_size: int = 2,
    time_shift: float | None = None,
) -> torch.Tensor:
    """Shift timesteps using the repo's resolution-dependent convention.

    This is the shared entry point for both training and inference to ensure
    the model always sees the same "t_used" that is implied by the resolution.

    Args:
        t: Base timesteps in [0, 1], shape [B] or scalar tensor.
        z_like: A latent/noise tensor with shape [B, C, H, W] used to infer
            the resolution-dependent shift when time_shift is None.
        patch_size: Patch size used to convert latent HxW into token count.
        time_shift: Optional explicit shift scalar to override auto-compute.
    """
    if time_shift is None:
        time_shift = resolution_time_shift(z_like, patch_size=patch_size)
    return apply_time_shift(t, time_shift)


def sample_weighted_mse(
    error: torch.Tensor, sample_weights: torch.Tensor
) -> torch.Tensor:
    """Weighted mean of the per-sample mean squared errors of ``error``.

    The samples are combined as ``sum(w_i * L_i) / sum(w_i)``: a weighted mean,
    not a weighted sum.  A sum would scale a step's gradient with the weights
    themselves, which is a learning-rate change wearing a reweighting's
    clothes, and two runs would no longer be comparable at equal step counts.

    The reduction runs in float32 whatever the compute dtype is, because the
    weight sum is a normalization: a denominator rounded to the compute dtype
    would bias every loss the weights touch.
    """
    if sample_weights.shape != (error.shape[0],):
        raise ValueError(
            "sample weights must hold one multiplier per sample, got "
            f"{tuple(sample_weights.shape)} for a batch of {error.shape[0]}"
        )
    weights = sample_weights.float()
    per_sample = (error.float() ** 2).flatten(1).mean(dim=1)
    return (per_sample * weights).sum() / weights.sum()


class BaseAlgorithm(ABC):
    @abstractmethod
    def sample_zt(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample z_t from the probability path p_t(z_t | z0, z1).
        Args:
            z0: Source samples (noise), [B, C, H, W]
            z1: Target samples (data), [B, C, H, W]
            t: Timesteps, [B] or [B, 1, 1, 1]
        Returns:
            Interpolated samples z_t, [B, C, H, W]
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model_output: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the loss for the algorithm.
        Args:
            model_output: Model predictions, [B, C, H, W]
            z0: Source samples (noise), [B, C, H, W]
            z1: Target samples (data), [B, C, H, W]
            t: Timesteps, [B] or [B, 1, 1, 1]
            sample_weights: Optional per-sample multipliers, one per row of the
                batch. With them the returned scalar is the weighted mean of
                the per-sample losses instead of their plain mean, which is how
                a run raises the share of its gradient that long captions
                receive; without them the computation is unchanged.
        Returns:
            Scalar loss value
        """
        pass


class FlowMatchingOT(BaseAlgorithm):
    """
    Flow matching with Optimal Transport path.
    """

    def sample_zt(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        z_t = (1.0 - t) * z0 + t * z1
        return z_t

    def compute_loss(
        self,
        model_output: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        velocity_target = z1 - z0
        if sample_weights is not None:
            return sample_weighted_mse(model_output - velocity_target, sample_weights)
        return F.mse_loss(model_output, velocity_target)
