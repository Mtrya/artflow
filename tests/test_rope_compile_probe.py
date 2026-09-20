import pytest
import torch
import copy

from src.models.dit_blocks import (
    apply_rotary_emb, apply_rotary_emb_real, set_real_rope,
    DoubleStreamAttention, SingleStreamAttention, UnconditionalAttention,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_real_rope_preserves_rotation_and_input_gradient(dtype, strided):
    torch.manual_seed(42)
    # Actual head dimension and a fused-QKV slice, with non-unit frequency
    # magnitude so the test does not accidentally assume rotation orthogonality.
    packed = torch.randn(2, 11, 3, 4, 72, dtype=dtype)
    value = packed[:, :, 0] if strided else packed[:, :, 0].contiguous()
    inputs = [value.detach().requires_grad_(), value.detach().requires_grad_()]
    freqs = torch.randn(11, 36, dtype=torch.complex64)
    gradient = torch.randn_like(value)
    outputs = [fn(x, freqs) for fn, x in zip((apply_rotary_emb, apply_rotary_emb_real), inputs)]
    gradients = [torch.autograd.grad(out, x, gradient)[0] for out, x in zip(outputs, inputs)]
    for reference, candidate in (outputs, gradients):
        assert candidate.dtype == dtype
        assert torch.isfinite(candidate).all()
        torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["double", "single_split", "single_hoisted", "unconditional"])
def test_real_rope_policy_is_instance_local_and_preserves_attention(kind):
    torch.manual_seed(9)
    cls = (DoubleStreamAttention if kind == "double" else
           UnconditionalAttention if kind == "unconditional" else SingleStreamAttention)
    reference = cls(32, 4, rope_axes_dim=[4, 4])
    candidate = copy.deepcopy(reference)
    set_real_rope(candidate, True)
    assert candidate.real_rope and not reference.real_rope
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    candidate.load_state_dict(reference.state_dict())
    assert candidate.real_rope  # Loading weights must not overwrite execution policy.
    image, text = torch.randn(2, 6, 32), torch.randn(2, 5, 32)

    def forward(model):
        if kind == "double":
            return model(image, text, (2, 3), 5)
        if kind == "unconditional":
            return (model(image, (2, 3)),)
        freqs = model.rope.prepare_freqs((2, 3), 5, image.device) if kind.endswith("hoisted") else None
        return (model(torch.cat((image, text), 1), (2, 3), 5, rope_freqs=freqs),)

    outputs = [forward(model) for model in (reference, candidate)]
    for expected, actual in zip(*outputs):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for result in outputs:
        sum(out.square().mean() for out in result).backward()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-7)
