"""NFT optimization checked against its analytic minimum and converged-policy gradient."""

import torch

from src.posttrain.nft import nft_loss


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


def test_adaptive_weighting_does_not_explode_when_policy_converges():
    # Reference scenario (NVlabs/DiffusionNFT train_nft_sd3.py): each branch
    # is normalized by its OWN per-sample residual. When v_theta has already
    # converged to the target but v_old has not, a shared v_theta residual
    # would be ~0 and blow the loss up; per-branch normalization keeps the
    # gradient at the documented magnitude.
    v_target = torch.zeros(1, 1)
    v_theta = v_target.clone().requires_grad_(True)
    v_old = torch.full((1, 1), 10.0)
    t = torch.tensor([0.5])
    x_t = torch.zeros(1, 1)  # any fixed point; only (1 - t) matters here
    loss = nft_loss(v_theta, v_old, v_target, torch.tensor([1.0]), 0.5,
                    x_t=x_t, t=t, adaptive_x0_weight=True)
    loss.backward()
    # Single element, r=1: loss reduces to |resid_pos|, whose gradient
    # magnitude is 2 * (1 - t) * beta = 0.5.
    torch.testing.assert_close(v_theta.grad, torch.full((1, 1), 0.5))
