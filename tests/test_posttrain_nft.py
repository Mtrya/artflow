import torch
import torch.nn.functional as F

from src.posttrain.nft import implicit_velocities, nft_loss


def test_balanced_reward_has_zero_gradient_at_old_policy():
    # r = 0.5 makes the two implicit branches exert equal and opposite pull;
    # the objective is stationary at v_theta = v_old.
    torch.manual_seed(0)
    v_theta = torch.randn(4, 2, 4, 4, requires_grad=True)
    v_old = v_theta.detach().clone()
    v_target = torch.randn(4, 2, 4, 4)
    loss = nft_loss(v_theta, v_old, v_target, torch.full((4,), 0.5), beta=0.5)
    loss.backward()
    torch.testing.assert_close(v_theta.grad, torch.zeros_like(v_theta.grad))


def test_implicit_velocities():
    v_old = torch.ones(2, 3)
    v_theta = torch.zeros(2, 3)
    v_pos, v_neg = implicit_velocities(v_old, v_theta, beta=0.25)
    torch.testing.assert_close(v_pos, torch.full((2, 3), 0.75))
    torch.testing.assert_close(v_neg, torch.full((2, 3), 1.25))


def test_positive_reward_pulls_toward_target_negative_pushes_away():
    torch.manual_seed(1)
    v_old = torch.zeros(1, 8)
    v_target = torch.ones(1, 8)
    # The optimum overshoots the target by design (v* = v_old + (2r-1)(v -
    # v_old)/beta), so assert on direction, not on reaching the target.
    for r, expected_sign in ((torch.tensor([1.0]), 1.0), (torch.tensor([0.0]), -1.0)):
        v_theta = torch.zeros(1, 8, requires_grad=True)
        opt = torch.optim.SGD([v_theta], lr=0.2)
        for _ in range(200):
            opt.zero_grad()
            nft_loss(v_theta, v_old, v_target, r, beta=0.5).backward()
            opt.step()
        assert v_theta.detach().mean().item() * expected_sign > 0.5


def test_closed_form_optimum():
    # Minimizing Eq. 5 over v_theta gives v* = v_old + (2r-1)(v - v_old)/beta.
    torch.manual_seed(2)
    v_old = torch.randn(1, 16)
    v_target = torch.randn(1, 16)
    beta = 0.3
    for r_val in (0.0, 0.25, 0.75, 1.0):
        r = torch.tensor([r_val])
        v_theta = v_old.clone().requires_grad_(True)
        opt = torch.optim.Adam([v_theta], lr=0.05)
        for _ in range(2000):
            opt.zero_grad()
            nft_loss(v_theta, v_old, v_target, r, beta).backward()
            opt.step()
        expected = v_old + (2 * r_val - 1) * (v_target - v_old) / beta
        torch.testing.assert_close(v_theta.detach(), expected, rtol=1e-2, atol=1e-2)


def test_old_policy_receives_no_gradient():
    v_old = torch.randn(2, 4, requires_grad=True)
    v_theta = torch.randn(2, 4, requires_grad=True)
    v_target = torch.randn(2, 4)
    nft_loss(v_theta, v_old, v_target, torch.tensor([0.5, 0.5]), 0.5).backward()
    assert v_old.grad is None
    assert v_theta.grad is not None


def test_adaptive_x0_weighting_is_finite_and_scales():
    torch.manual_seed(3)
    x0 = torch.randn(3, 2, 4, 4)
    eps = torch.randn(3, 2, 4, 4)
    t = torch.tensor([0.2, 0.5, 0.9])
    tb = t.view(-1, 1, 1, 1)
    x_t = (1 - tb) * eps + tb * x0
    v_target = x0 - eps
    v_theta = (v_target + 0.1).requires_grad_(True)
    v_old = v_target + 0.2
    r = torch.tensor([0.9, 0.5, 0.1])
    loss = nft_loss(v_theta, v_old, v_target, r, 0.5,
                    x_t=x_t, t=t, adaptive_x0_weight=True)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(v_theta.grad).all()


def test_adaptive_weighting_requires_xt_and_t():
    v = torch.randn(2, 4)
    import pytest
    with pytest.raises(ValueError):
        nft_loss(v, v, v, torch.tensor([0.5, 0.5]), 0.5,
                 adaptive_x0_weight=True)
