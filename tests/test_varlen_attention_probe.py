import pytest
import torch
import torch.nn.functional as F

from src.models.varlen_attention import mask_metadata, packed_attention


def reference_varlen(q, k, v, cu_q, cu_k, max_q, max_k):
    outputs = []
    for row in range(len(cu_q) - 1):
        qi = q[cu_q[row]:cu_q[row + 1]].transpose(0, 1).unsqueeze(0)
        ki = k[cu_k[row]:cu_k[row + 1]].transpose(0, 1).unsqueeze(0)
        vi = v[cu_k[row]:cu_k[row + 1]].transpose(0, 1).unsqueeze(0)
        outputs.append(F.scaled_dot_product_attention(qi, ki, vi)[0].transpose(0, 1))
    return torch.cat(outputs)


def test_kv_packing_preserves_all_queries_and_masked_gradients():
    torch.manual_seed(4)
    q, k, v = [torch.randn(3, 2, 7, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    keep = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0],
                         [1, 0, 1, 0, 1, 0, 1]], dtype=torch.bool)
    metadata = mask_metadata(keep)
    assert metadata[1].tolist() == [0, 7, 14, 21]
    assert metadata[2].tolist() == [0, 7, 9, 13]
    assert metadata[1].dtype == metadata[2].dtype == torch.int32
    actual = packed_attention(q, k, v, metadata, kernel=reference_varlen)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(q)
    ga = torch.autograd.grad(actual, (q, k, v), upstream)
    ge = torch.autograd.grad(expected, (q, k, v), upstream)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def test_reject_empty_key_sequence():
    with pytest.raises(ValueError):
        mask_metadata(torch.zeros(2, 4, dtype=torch.bool))


@pytest.mark.parametrize("mask_kind", ["none", "irregular", "empty_text"])
def test_model_hoists_metadata_and_preserves_both_streams(monkeypatch, mask_kind):
    """CPU reference kernel checks integration, not CUDA accuracy/performance."""
    from src.models.artflow import ArtFlow
    from src.models import dit_blocks, varlen_attention

    monkeypatch.setattr(varlen_attention, "validate_runtime", lambda: None)
    monkeypatch.setattr(varlen_attention.NativeFlashVarlen, "apply", staticmethod(reference_varlen))
    metadata_ids = []

    def capture(q, k, v, metadata):
        metadata_ids.append(id(metadata))
        return packed_attention(q, k, v, metadata)

    monkeypatch.setattr(dit_blocks, "packed_attention", capture)
    torch.manual_seed(52)
    model = ArtFlow(hidden_size=64, num_heads=4, double_stream_depth=1,
                    single_stream_depth=2, mlp_ratio=2.,
                    conditioning_scheme="fused",
                    double_stream_modulation="layer", single_stream_modulation="layer")
    torch.nn.init.normal_(model.final_layer[1].weight, std=.05)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.05)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    text = torch.randn(2, 7, 1024, requires_grad=True)
    pooled = torch.randn(2, 1024, requires_grad=True)
    t = torch.rand(2)
    mask = None
    if mask_kind == "irregular":
        mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]])
    elif mask_kind == "empty_text":
        mask = torch.zeros(2, 7, dtype=torch.long)
    parameters = tuple(model.parameters())
    ref = model(x, t, txt=text, txt_pooled=pooled, txt_mask=mask, fast_attn=True)
    gradients = torch.autograd.grad(ref.square().mean(), (x, text, pooled, *parameters))
    actual = model(x, t, txt=text, txt_pooled=pooled, txt_mask=mask,
                   fast_attn=True, native_flash_varlen=True)
    actual_gradients = torch.autograd.grad(actual.square().mean(), (x, text, pooled, *parameters))
    assert len(metadata_ids) == 3 and len(set(metadata_ids)) == 1
    torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-5)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-5)
