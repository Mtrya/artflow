import torch
from torch import nn

from src.models.artflow import ArtFlow
from src.posttrain.dmd import (
    BottleneckCapture,
    DiscriminatorHead,
    backward_simulate,
    dmd_surrogate_loss,
    discriminator_loss,
    fake_score_loss,
    generator_adv_loss,
    noise_sample,
)


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
    x = z.clone()
    for i in range(8):
        x = x + (grid[i + 1] - grid[i]) * model(x, grid[i])
    torch.testing.assert_close(out, x)


def test_backward_simulate_grad_flows_only_through_suffix():
    torch.manual_seed(0)
    model = LinearVelocity()
    z = torch.randn(2, 2, 8, 8)
    grid = torch.linspace(0.0, 1.0, 9)
    out = backward_simulate(model, z, grid, start_idx=6)
    assert out.requires_grad
    out.sum().backward()
    assert model.net.weight.grad is not None
    # Rebuild the no-grad prefix input independently; the grad-bearing suffix
    # must start from exactly those values.
    with torch.no_grad():
        x = z.clone()
        for i in range(6):
            x = x + (grid[i + 1] - grid[i]) * model(x, grid[i])
        expected = x
        for i in range(6, 8):
            expected = expected + (grid[i + 1] - grid[i]) * model(expected, grid[i])
    torch.testing.assert_close(out.detach(), expected)


def test_backward_simulate_rejects_bad_start():
    model = LinearVelocity()
    z = torch.randn(1, 2, 8, 8)
    grid = torch.linspace(0.0, 1.0, 5)
    import pytest
    with pytest.raises(ValueError):
        backward_simulate(model, z, grid, start_idx=4)


def test_dmd_surrogate_loss_gradient_direction():
    torch.manual_seed(1)
    x0 = torch.randn(3, 2, 4, 4, requires_grad=True)
    v_fake = torch.randn(3, 2, 4, 4)
    v_real = torch.randn(3, 2, 4, 4)
    t = torch.tensor([0.2, 0.5, 0.8])
    loss = dmd_surrogate_loss(x0, v_fake, v_real, t, normalize=False)
    loss.backward()
    w = (t ** 2 / (1 - t)).view(-1, 1, 1, 1)
    expected = (v_fake - v_real) * w / x0.numel()
    torch.testing.assert_close(x0.grad, expected)


def test_dmd_surrogate_moves_toward_real_score():
    # If fake velocity systematically exceeds real velocity, one descent step
    # must move x0 opposite to that difference.
    torch.manual_seed(2)
    x0 = torch.zeros(2, 2, 4, 4, requires_grad=True)
    v_real = torch.zeros(2, 2, 4, 4)
    v_fake = torch.ones(2, 2, 4, 4)
    t = torch.tensor([0.5, 0.5])
    dmd_surrogate_loss(x0, v_fake, v_real, t, normalize=False).backward()
    assert (x0.grad > 0).all()  # descent step subtracts a positive gradient


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


def test_gan_losses_non_saturating_direction():
    real_good, real_bad = torch.tensor(3.0), torch.tensor(-3.0)
    fake = torch.tensor(0.5)
    assert discriminator_loss(real_good, fake) < discriminator_loss(real_bad, fake)
    assert generator_adv_loss(torch.tensor(2.0)) < generator_adv_loss(torch.tensor(-2.0))


def _tiny_model():
    torch.manual_seed(3)
    return ArtFlow(in_channels=4, txt_in_features=16, hidden_size=32,
                   num_heads=4, double_stream_depth=2, single_stream_depth=1,
                   mlp_ratio=2)


def test_bottleneck_capture_and_head_on_tiny_model():
    model = _tiny_model()
    head = DiscriminatorHead(dim=32)
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.3, 0.7])
    txt = torch.randn(2, 5, 16)
    with BottleneckCapture(model) as cap:
        out = model(x, t, txt)
        feats = cap.features
    assert feats is not None and feats.shape == (2, 16, 32)
    assert out.shape == x.shape
    logit = head(feats)
    assert logit.shape == (2,)
    # The discriminator loss must reach both the head and the backbone.
    discriminator_loss(logit[:1], logit[1:]).backward()
    assert head.net[0].weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.blocks[-1].parameters())


def test_bottleneck_capture_hook_removal():
    model = _tiny_model()
    with BottleneckCapture(model) as cap:
        model(torch.randn(1, 4, 8, 8), torch.tensor([0.5]), torch.randn(1, 5, 16))
        assert cap.features is not None
    cap.features = None
    model(torch.randn(1, 4, 8, 8), torch.tensor([0.5]), torch.randn(1, 5, 16))
    assert cap.features is None  # hook no longer fires
