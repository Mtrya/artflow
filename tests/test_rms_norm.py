"""Normalization outputs and gradients checked against torch.nn.RMSNorm."""

import pytest
import torch

from src.models.dit_blocks import RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("width", [72, 1152])
def test_rms_norm_matches_reference_with_learned_gains(dtype, device, width):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("requires an Ascend device")
    torch.manual_seed(42)
    reference = torch.nn.RMSNorm(width, eps=1e-6).to(device)
    with torch.no_grad():
        reference.weight.uniform_(0.5, 1.5)
    norm = RMSNorm(width, eps=1e-6).to(device)
    norm.load_state_dict(reference.state_dict(), strict=True)

    # Q/K inputs have noncontiguous sequence/head axes.
    x = torch.randn(2, 11, 4, width, device=device, dtype=dtype).transpose(1, 2)
    x = x.detach().requires_grad_()
    other = x.detach().requires_grad_()
    dy = torch.randn_like(x)
    expected, actual = reference(x), norm(other)
    ref_grads = torch.autograd.grad(expected, (x, reference.weight), dy)
    grads = torch.autograd.grad(actual, (other, norm.weight), dy)
    assert actual.dtype == expected.dtype == dtype
    assert grads[1].dtype == torch.float32
    if device == "cpu":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(grads, ref_grads):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        # Reduction ordering can cross a BF16 rounding boundary. Relative L2
        # avoids an arbitrary pointwise relative error near a zero gradient.
        limit = 0.005 if dtype == torch.bfloat16 else 1e-5
        for a, b in zip((actual, *grads), (expected, *ref_grads)):
            assert torch.isfinite(a).all()
            error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
            assert error < limit
