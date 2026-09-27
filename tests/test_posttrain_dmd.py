"""DMD behavior checked against the flow solver and analytic residuals."""

import torch
from torch import nn

from src.posttrain.dmd import backward_simulate, dmd_surrogate_loss, fake_score_loss, noise_sample
from src.flow.solvers import sample_ode


class LinearVelocity(nn.Module):
    """Minimal stand-in velocity model with the model_fn(x, t) convention."""

    def __init__(self, channels=2, size=8):
        super().__init__()
        self.net = nn.Conv2d(channels, channels, 1)

    def forward(self, x, t):
        return self.net(x) * (1.0 + t)


def test_backward_simulate_matches_plain_euler_without_grad():
    torch.manual_seed(0)
    model = LinearVelocity()
    z = torch.randn(2, 2, 8, 8)
    grid = torch.linspace(0.0, 1.0, 9)
    with torch.no_grad():
        out = backward_simulate(model, z, grid, start_idx=0)
    reference = sample_ode(lambda x, t: model(x, t[:, None, None, None]),
                           z, steps=8, solver="euler", time_shift=1)
    torch.testing.assert_close(out, reference)


def test_dmd_normalization_is_per_sample_teacher_residual():
    # DMD2 (sd_guidance.py) divides each sample's gradient by that sample's
    # |x0_gen - x0_real|, NOT by a batch-wide magnitude of the score
    # difference. Samples with small score differences must keep small
    # gradients.
    x0 = torch.zeros(2, 1, requires_grad=True)
    v_real = torch.zeros(2, 1)
    v_fake = torch.tensor([[1.0], [1e-6]])
    t = torch.tensor([0.5, 0.5])
    # x0_real = x_t + (1 - t) * v_real = x_t; per-sample residuals 2 and 1.
    x_t = torch.tensor([[-2.0], [-1.0]])
    dmd_surrogate_loss(x0, v_fake, v_real, t, x_t=x_t).backward()
    # Analytic gradients for residuals 2 and 1, with a two-sample mean.
    expected = torch.tensor([[0.125], [0.00000025]])
    torch.testing.assert_close(x0.grad, expected)
    assert x0.grad[0].abs() > 1e4 * x0.grad[1].abs()


def test_fake_score_loss_uses_flow_target():
    x0 = torch.randn(2, 2, 4, 4)
    eps = torch.randn(2, 2, 4, 4)
    v_pred = (x0 - eps) + 0.5
    assert torch.isclose(fake_score_loss(v_pred, x0, eps), torch.tensor(0.25))


def test_noise_sample_endpoints():
    x0 = torch.randn(2, 2, 4, 4)
    x_t, eps = noise_sample(x0, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(x_t[0], eps[0])  # t=0 is pure noise
    torch.testing.assert_close(x_t[1], x0[1])   # t=1 is clean data
