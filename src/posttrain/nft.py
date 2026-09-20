"""DiffusionNFT (arXiv:2509.16117) loss in this repo's flow convention.

DiffusionNFT contrasts positive and negative generations directly on the
forward process. For each sampled image with optimality probability r, the
objective is

    L = r * ||v+ - v||^2 + (1 - r) * ||v- - v||^2,

with the implicit positive/negative policies parameterized through the old
(EMA data-collection) policy v_old and the trainable policy v_theta:

    v+ = (1 - beta) * v_old + beta * v_theta
    v- = (1 + beta) * v_old - beta * v_theta

Minimizing L drives v_theta to v_old + (2r - 1) * (v - v_old) / beta: toward
the target velocity for preferred samples (r -> 1) and away from it for
dispreferred ones (r -> 0). No likelihoods, no trajectories — only the final
images and their rewards are needed, so rollouts may use any solver and any
step count (the paper's ablations use 10-step rollouts).
"""

import torch
import torch.nn.functional as F


def implicit_velocities(v_old: torch.Tensor, v_theta: torch.Tensor,
                        beta: float):
    """Return the implicit positive and negative policy velocities."""
    v_pos = (1.0 - beta) * v_old + beta * v_theta
    v_neg = (1.0 + beta) * v_old - beta * v_theta
    return v_pos, v_neg


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).flatten(1).mean(dim=1)


def nft_loss(v_theta: torch.Tensor, v_old: torch.Tensor,
             v_target: torch.Tensor, r: torch.Tensor, beta: float, *,
             x_t: torch.Tensor | None = None, t: torch.Tensor | None = None,
             adaptive_x0_weight: bool = False) -> torch.Tensor:
    """DiffusionNFT objective, Eq. 5 of the paper.

    Args:
        v_theta: trainable policy velocity at (x_t, t).
        v_old: old data-collection policy velocity at the same point
            (detached by the caller or here).
        v_target: flow-matching velocity target x0 - eps of the generated
            image.
        r: per-sample optimality probability in [0, 1], shape (N,).
        beta: guidance-strength hyperparameter of the implicit policies.
        x_t, t: the noised sample and timesteps; required when
            adaptive_x0_weight is set.
        adaptive_x0_weight: replace the plain velocity MSE with the paper's
            self-normalized x0 regression, ||x0_pred - x0||^2 /
            sg(mean(|x0_pred - x0|)), per branch.

    v_old is detached internally; the old policy must never receive gradients.
    """
    v_old = v_old.detach()
    r = r.to(v_theta.dtype).view(-1)
    v_pos, v_neg = implicit_velocities(v_old, v_theta, beta)
    err_pos = _per_sample_mse(v_pos, v_target)
    err_neg = _per_sample_mse(v_neg, v_target)
    if adaptive_x0_weight:
        if x_t is None or t is None:
            raise ValueError("adaptive_x0_weight requires x_t and t")
        t_b = t.view(-1, *([1] * (v_theta.dim() - 1))).to(v_theta.dtype)
        # x0_pred = x_t + (1 - t) * v under this repo's path convention.
        x0_err = ((1.0 - t_b) * (v_theta - v_target)).flatten(1).abs().mean(dim=1)
        norm = x0_err.detach().mean().clamp_min(1e-8)
        err_pos = err_pos * (1.0 - t) ** 2 / norm
        err_neg = err_neg * (1.0 - t) ** 2 / norm
    return (r * err_pos + (1.0 - r) * err_neg).mean()
