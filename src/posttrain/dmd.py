"""DMD2-style distribution-matching distillation primitives.

Implements the pieces of DMD2 (arXiv:2405.14867) that the Stage-6 joint
DMD+RL loop (DMDR, arXiv:2511.13649) builds on. Everything here is written in
this repo's timestep convention (t=0 noise, t=1 data; x_t = (1-t)*eps + t*x0;
velocity target v = x0 - eps).

Derivation of the DMD gradient in velocity space. The DMD update descends

    E_t[ (s_fake - s_real)(x_t) . d x_t/d theta ],

where s is the marginal score. For the rectified-flow path, Tweedie gives

    s(x_t) = (t * v - x_t) / (1 - t)

with v the velocity prediction, and x_t depends on the generated x0 through
x_t = (1-t)*eps + t*x0, so d x_t/d x0 = t. The score difference collapses to
(v_fake - v_real) * t/(1-t), and the full weight is t^2/(1-t) up to the
learning rate. `dmd_surrogate_loss` implements exactly this direction.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


def backward_simulate(model_fn, z: torch.Tensor, grid: torch.Tensor,
                      start_idx: int) -> torch.Tensor:
    """DMD2 backward simulation: match training inputs to inference inputs.

    The student runs without gradients from pure noise along its inference
    grid up to `start_idx`, then takes the remaining step(s) with gradients
    enabled. The no-grad prefix reproduces the intermediate states the student
    actually visits at inference, instead of training on off-distribution
    re-noised states.

    Args:
        model_fn: closure (x, t_scalar) -> velocity, wrapping the student with
            its conditioning fixed.
        z: initial noise, (N, C, H, W).
        grid: 1D ascending tensor of K+1 timesteps from noise (0) to data (1),
            the same grid used at inference (post-shift).
        start_idx: index into `grid`; steps before it run under no_grad.

    Returns:
        The generated clean latent x0. Requires grad iff start_idx < K.
    """
    if not 0 <= start_idx < len(grid) - 1:
        raise ValueError(f"start_idx {start_idx} outside [0, {len(grid) - 1})")
    x = z
    with torch.no_grad():
        for i in range(start_idx):
            x = x + (grid[i + 1] - grid[i]) * model_fn(x, grid[i])
    for i in range(start_idx, len(grid) - 1):
        x = x + (grid[i + 1] - grid[i]) * model_fn(x, grid[i])
    return x


def dmd_surrogate_loss(x0_gen: torch.Tensor, v_fake: torch.Tensor,
                       v_real: torch.Tensor, t: torch.Tensor, *,
                       x_t: torch.Tensor | None = None,
                       normalize: bool = True) -> torch.Tensor:
    """Surrogate loss whose gradient is the DMD update direction.

    Gradient descent on the returned value moves x0_gen along
    -(t^2/(1-t)) * (v_fake - v_real), i.e. toward the real distribution's
    score. `v_fake`/`v_real` are the fake- and real-score networks' velocity
    predictions at the re-noised x0_gen; both are detached here.

    `normalize` applies DMD2's gradient normalization: each sample's
    gradient is divided by that sample's teacher reconstruction residual
    sg(mean(|x0_gen - x0_real|)), where x0_real = x_t + (1 - t) * v_real
    (DMD2 sd_guidance.py divides by abs(p_real) per sample). This keeps the
    update proportional to the *relative* score discrepancy instead of
    amplifying tiny absolute differences; `x_t` is required when enabled.
    """
    t = t.view(-1, *([1] * (x0_gen.dim() - 1)))
    weight = (t ** 2 / (1.0 - t)).clamp(max=1e4)
    grad = (v_fake.detach() - v_real.detach()) * weight
    if normalize:
        if x_t is None:
            raise ValueError("normalize=True requires x_t (the re-noised "
                             "x0_gen the scores were evaluated at)")
        x0_real = x_t.detach() + (1.0 - t) * v_real.detach()
        resid = ((x0_gen.detach() - x0_real).double().abs()
                 .flatten(1).mean(dim=1).clamp_min(1e-8))
        grad = grad / resid.view(-1, *([1] * (grad.dim() - 1))).to(grad.dtype)
        grad = torch.nan_to_num(grad)
    return (x0_gen * grad).mean()


def noise_sample(x0: torch.Tensor, t: torch.Tensor,
                 generator: torch.Generator | None = None):
    """Draw eps and form x_t = (1-t)*eps + t*x0 for the given timesteps."""
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype,
                      generator=generator)
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - t) * eps + t * x0, eps


def fake_score_loss(v_pred: torch.Tensor, x0_fake: torch.Tensor,
                    eps: torch.Tensor) -> torch.Tensor:
    """Ordinary flow-matching loss training the fake-score network.

    `x0_fake` must already be detached from the student. The velocity target
    is x0 - eps, the same convention as pretraining.
    """
    return F.mse_loss(v_pred, x0_fake - eps)


class BottleneckCapture:
    """Capture the input of `model.final_layer` (post-blocks features).

    This is DMD2's discriminator attachment point: the classification branch
    reads the fake-score backbone's bottleneck features. A forward hook keeps
    the model itself untouched; the captured tensor stays in the autograd
    graph, so the discriminator loss also trains the upstream features, as in
    DMD2.
    """

    def __init__(self, model: nn.Module):
        if not hasattr(model, "final_layer"):
            raise ValueError("model must expose a final_layer module")
        self.features: torch.Tensor | None = None
        self._handle = model.final_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.features = inputs[0]

    def clear(self):
        self.features = None

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DiscriminatorHead(nn.Module):
    """Classification branch on bottleneck token features.

    Masked mean-pools tokens and maps them to a single logit, following
    DMD2's minimalist design. For a DiT there is no encoder/decoder split;
    mean pooling over the final block features is the bottleneck equivalent.
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim // 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            pooled = features.mean(dim=1)
        else:
            w = mask.unsqueeze(-1).to(features.dtype)
            pooled = (features * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return self.net(pooled).squeeze(-1)


def discriminator_loss(real_logit: torch.Tensor,
                       fake_logit: torch.Tensor) -> torch.Tensor:
    """Non-saturating GAN discriminator loss (DMD2 Eq. 4, softplus form)."""
    return (F.softplus(-real_logit) + F.softplus(fake_logit)).mean()


def generator_adv_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Non-saturating GAN generator loss."""
    return F.softplus(-fake_logit).mean()
