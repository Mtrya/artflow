"""
Flow matching and score matching algorithms for generative modeling

This module implements three training algorithms for ablation:
1. ScoreMatchingDiffusion - Baseline (predict score, VP-SDE diffusion path)
2. FlowMatchingDiffusion - Flow matching with diffusion path (predict velocity)
3. FlowMatchingOT - Flow matching with optimal transport path (predict velocity)

Each algorithm encapsulates both the probability path and loss computation.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    @abstractmethod
    def sample_zt(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, shift: float=1.0
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
    ) -> torch.Tensor:
        """
        Compute the loss for the algorithm.
        Args:
            model_output: Model predictions, [B, C, H, W]
            z0: Source samples (noise), [B, C, H, W]
            z1: Target samples (data), [B, C, H, W]
            t: Timesteps, [B] or [B, 1, 1, 1]
        Returns:
            Scalar loss value
        """
        pass


class ScoreMatchingDiffusion(BaseAlgorithm):
    """
    Score matching with VP-SDE diffusion path.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _marginal_prob_std(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return std

    def sample_zt(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, shift: float=1.0
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        z_t = mean_coeff * z1 + std * z0
        return z_t

    def compute_loss(
        self,
        model_output: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        std = self._marginal_prob_std(t)

        # Stable formulation:
        # loss = || s(x, t) - s_target ||^2 * std^2
        #      = || s(x, t) - (-z0/std) ||^2 * std^2
        #      = || s(x, t) * std + z0 ||^2
        weighted_loss = ((model_output * std + z0) ** 2).mean()

        return weighted_loss


class FlowMatchingDiffusion(BaseAlgorithm):
    """
    Flow matching with VP-SDE diffusion path
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _marginal_prob_std(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return std

    def _velocity_target(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)

        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

        # v_t = d(mean)/dt * z1 + d(std)/dt * z0
        # d(mean)/dt = mean * (-0.5 * beta_t)
        # d(std)/dt = beta_t * (1 - std^2) / (2 * std)  <-- derived from std^2 = 1 - mean^2

        dmean_dt = -0.5 * beta_t * mean_coeff
        dstd_dt = beta_t * (1.0 - std**2) / (2.0 * std + 1e-6)

        velocity = dmean_dt * z1 + dstd_dt * z0
        return velocity

    def sample_zt(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, shift: float = 1.0,
    ) -> torch.Tensor:
        t = (shift * t) / (1 + (shift - 1) * t)

        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

        return mean_coeff * z1 + std * z0

    def compute_loss(
        self,
        model_output: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        velocity_target = self._velocity_target(z0, z1, t)
        return F.mse_loss(model_output, velocity_target)


class FlowMatchingOT(BaseAlgorithm):
    """
    Flow matching with Optimal Transport path.
    """

    def sample_zt(
        self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, shift: float=1.0
    ) -> torch.Tensor:
        t = (shift * t) / (1 + (shift - 1) * t)
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
    ) -> torch.Tensor:
        velocity_target = z1 - z0
        return F.mse_loss(model_output, velocity_target)
