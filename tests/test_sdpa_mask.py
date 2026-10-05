"""Masked attention and its gradients checked against unpadded PyTorch attention."""

import pytest
import torch
import torch.nn.functional as F

from src.models.dit_blocks import sdpa_with_pad_mask


@pytest.mark.parametrize('keep', [[True] * 7, [True, False, True, False, False, True, False],
                                  [True, False, False, False, False, False, False]])
def test_masked_attention_matches_removing_padded_keys(keep):
    torch.manual_seed(0)
    q, k, v = [torch.randn(2, 3, 7, 8, requires_grad=True) for _ in range(3)]
    mask = torch.tensor(keep)
    actual = sdpa_with_pad_mask(q, k, v, mask[None, None, None])
    expected = F.scaled_dot_product_attention(q, k[:, :, mask], v[:, :, mask])
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    for output, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
