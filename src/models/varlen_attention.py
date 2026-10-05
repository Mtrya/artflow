"""Opt-in native variable-length attention for the pinned torch 2.9.1 runtime.

Keep every query and pack only valid K/V tokens. Padded-query outputs remain
defined, matching the existing key-padding mask (not a query-padding mask).
Metadata is built once outside compiled blocks and shared by every layer.
"""

import torch


def validate_runtime():
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("native FlashAttention varlen requires the audited torch 2.9.1 runtime")


class NativeFlashVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k):
        out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            q, k, v, cu_q, cu_k, max_q, max_k, 0., False, False)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, rng, unused)
        ctx.max_q, ctx.max_k = max_q, max_k
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q, k, v, out, lse, cu_q, cu_k, rng, unused = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad, q, k, v, out, lse, cu_q, cu_k, ctx.max_q, ctx.max_k,
            0., False, rng, unused)
        return dq, dk, dv, None, None, None, None


def mask_metadata(keep):
    if keep.ndim != 2 or keep.dtype != torch.bool:
        raise ValueError("keep must be a boolean [batch, sequence] tensor")
    batch, sequence = keep.shape
    lengths = keep.sum(1, dtype=torch.int32)
    if batch < 1 or sequence < 1 or bool((lengths == 0).any()):
        raise ValueError("each sequence must have at least one valid key")
    indices = keep.flatten().nonzero().flatten()
    cu_q = torch.arange(batch + 1, device=keep.device, dtype=torch.int32) * sequence
    cu_k = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
    return indices, cu_q, cu_k, sequence, int(lengths.max())


def packed_attention(q, k, v, metadata, kernel=None):
    indices, cu_q, cu_k, max_q, max_k = metadata
    batch, heads, sequence, dim = q.shape
    query = q.transpose(1, 2).reshape(batch * sequence, heads, dim)
    key = k.transpose(1, 2).reshape(batch * sequence, heads, dim).index_select(0, indices)
    value = v.transpose(1, 2).reshape(batch * sequence, heads, dim).index_select(0, indices)
    if kernel is None:
        kernel = NativeFlashVarlen.apply
    output = kernel(query, key, value, cu_q, cu_k, max_q, max_k)
    return output.reshape(batch, sequence, heads, dim).transpose(1, 2)
