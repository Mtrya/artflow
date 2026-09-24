"""FFN fusion retains outputs, master-parameter gradients and checkpoint layout."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from src.models.dit_blocks import GatedFeedForward


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ffn_forward_and_gradients_match_unfused(device, dtype):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("requires an Ascend device")
    torch.manual_seed(42)
    model = GatedFeedForward(32, 96).to(device)
    reference = deepcopy(model)
    reference.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 13, 32, device=device, dtype=dtype).requires_grad_()
    other = x.detach().clone().requires_grad_()
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        actual = model(x)
        gate, linear = reference.up_proj(other).chunk(2, -1)
        expected = reference.down_proj(F.silu(gate) * linear)
    dy = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, *model.parameters()), dy)
    ref_grads = torch.autograd.grad(expected, (other, *reference.parameters()), dy)
    assert actual.dtype == expected.dtype == dtype
    assert all(g.dtype == torch.float32 for g in grads[1:])
    for a, b in zip((actual, *grads), (expected, *ref_grads)):
        assert torch.isfinite(a).all()
        if device == "cpu":
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            # Fused activation omits BF16 rounding of the intermediate SiLU.
            limit = 0.006 if dtype == torch.bfloat16 else 1e-5
            relative_l2 = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
            assert relative_l2 < limit
