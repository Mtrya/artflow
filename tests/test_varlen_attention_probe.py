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
