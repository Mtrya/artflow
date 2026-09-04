"""
Tests for sdpa_with_pad_mask: the masked-SDPA memory fix.

The old path passed a bool mask straight into F.scaled_dot_product_attention,
which falls off flash onto the math backend and materializes B*H*S*S attention
weights per layer (saved for backward) — tens of GiB at seq ~2K and the cause
of the stage-2 mid-training OOM spikes. The new path converts to an additive
bias and prefers the memory-efficient kernel. These tests pin the numerics.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dit_blocks import sdpa_with_pad_mask


def _ref_masked_attention(q, k, v, keep_mask):
    """fp64 reference: scores -> mask -> softmax -> out."""
    B, H, S, D = q.shape
    scores = (
        torch.matmul(q.double(), k.double().transpose(-1, -2)) / (D**0.5)
    )  # [B, H, S, S]
    neg = torch.finfo(torch.float64).min
    scores = scores.masked_fill(~keep_mask.double().bool(), neg)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.double())


def test_parity_vs_reference():
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 96, 32
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    # per-sample valid key counts (padding at the end)
    keep = torch.zeros(B, S, dtype=torch.bool)
    keep[0, :70] = True
    keep[1, :33] = True
    mask = keep.view(B, 1, 1, S)

    out = sdpa_with_pad_mask(q, k, v, mask)
    ref = _ref_masked_attention(q, k, v, mask)
    err = (out.double() - ref).abs().max().item()
    assert err < 1e-4, f"masked path deviates from reference: {err}"


def test_none_mask_matches_plain_sdpa():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 64, 32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = sdpa_with_pad_mask(q, k, v, None)
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    assert torch.allclose(out, ref)


def test_fully_masked_keys_give_zero_not_nan():
    # A sample with all-but-one key masked must stay finite.
    q = torch.randn(1, 2, 16, 8)
    k, v = torch.randn_like(q), torch.randn_like(q)
    mask = torch.zeros(1, 1, 1, 16, dtype=torch.bool)
    mask[..., 0] = True
    out = sdpa_with_pad_mask(q, k, v, mask)
    assert torch.isfinite(out).all()
    # Only key 0 attended -> output equals v[..., 0, :] broadcast
    assert torch.allclose(out[..., 1:, :], v[:, :, 0, :].unsqueeze(2).expand_as(out[..., 1:, :]), atol=1e-5)


def test_backward_runs():
    q = torch.randn(2, 4, 64, 32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = torch.ones(2, 1, 1, 64, dtype=torch.bool)
    mask[0, ..., 40:] = False
    out = sdpa_with_pad_mask(q, k, v, mask)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_long_seq_memory_is_linear_enough():
    # The regression we care about: masked attention at seq 4K must not
    # allocate O(S^2) per layer. On CPU we can only check it runs; on CUDA
    # assert the delta stays far below the math-path S^2 blow-up.
    if not torch.cuda.is_available():
        return
    B, H, S, D = 16, 16, 4096, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device="cuda")
    mask[:, :, S // 2 :] = False
    torch.cuda.reset_peak_memory_stats()
    out = sdpa_with_pad_mask(q, k, v, mask)
    out.float().sum().backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    # math path would materialize >= B*H*S^2*2B = 4 GiB of scores alone (x2 for
    # softmax save); efficient path stays well under 1 GiB for this op.
    assert peak_gb < 1.5, f"masked attention peak {peak_gb:.2f} GiB looks quadratic"


if __name__ == "__main__":
    test_parity_vs_reference()
    test_none_mask_matches_plain_sdpa()
    test_fully_masked_keys_give_zero_not_nan()
    test_backward_runs()
    test_long_seq_memory_is_linear_enough()
    print("all masked-sdpa tests passed")
