"""Compare masked SDPA kernels at layouts observed in the 640p Kineto trace.

This is a component diagnostic, not a training throughput or accuracy gate.
Q/K are BHSD contiguous; V is a view into fused B,S,3,H,D storage. The upstream
gradient is BSHD contiguous before transposition, matching the compiled trace.
Synthetic caption masks preserve all image keys and vary retained text length.
Force one backend at a time: unsupported kernels must fail, never fall back.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def make_inputs(batch, sequence, image_tokens, *, device="cuda", dtype=torch.bfloat16):
    if batch < 1 or image_tokens < 1 or sequence <= image_tokens:
        raise ValueError("require positive batch/image tokens and sequence > image tokens")
    heads, dim = 16, 72
    q = torch.randn(batch, heads, sequence, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    packed = torch.randn(batch, sequence, 3, heads, dim, device=device, dtype=dtype)
    v = packed[:, :, 2].transpose(1, 2)
    grad = torch.randn(batch, sequence, heads, dim, device=device, dtype=dtype).transpose(1, 2)
    # The traced bias has last-dimension storage rounded to a multiple of eight.
    padded = (sequence + 7) // 8 * 8
    storage = torch.zeros(batch, 1, 1, padded, device=device, dtype=dtype)
    bias = storage[..., :sequence]
    lengths = torch.linspace(0, sequence - image_tokens, batch, device=device).long()
    positions = torch.arange(sequence, device=device)
    bias.masked_fill_(positions[None, None, None, :] >=
                      (image_tokens + lengths)[:, None, None, None], -float("inf"))
    return tuple(t.detach().requires_grad_() for t in (q, k, v)), bias, grad


def errors(actual, reference):
    delta = actual.float() - reference.float()
    return dict(finite=bool(torch.isfinite(actual).all()),
                max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / reference.float().norm().clamp_min(1e-12)))


def evaluate(inputs, bias, upstream, backend):
    with sdpa_kernel([backend]):
        output = F.scaled_dot_product_attention(*inputs, attn_mask=bias, dropout_p=0.)
        gradients = torch.autograd.grad(output, inputs, upstream)
    return (output.detach(), *gradients)


def measure(inputs, bias, upstream, backend, iterations):
    for _ in range(4):
        evaluate(inputs, bias, upstream, backend)
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    forward_ms, backward_ms, total_ms = [], [], []
    with sdpa_kernel([backend]):
        for _ in range(iterations):
            start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            start.record()
            output = F.scaled_dot_product_attention(*inputs, attn_mask=bias, dropout_p=0.)
            middle.record()
            gradients = torch.autograd.grad(output, inputs, upstream)
            end.record()
            end.synchronize()
            forward_ms.append(start.elapsed_time(middle))
            backward_ms.append(middle.elapsed_time(end))
            total_ms.append(start.elapsed_time(end))
            del output, gradients
    return dict(forward_median_ms=statistics.median(forward_ms),
                backward_median_ms=statistics.median(backward_ms),
                total_median_ms=statistics.median(total_ms),
                total_samples_ms=total_ms,
                incremental_peak_bytes=torch.cuda.max_memory_allocated() - baseline_memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=1649)
    parser.add_argument("--image-tokens", type=int, default=1600)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    torch.manual_seed(args.seed)
    inputs, bias, upstream = make_inputs(args.batch, args.sequence, args.image_tokens)
    print(json.dumps(dict(torch_version=torch.__version__, cudnn_version=torch.backends.cudnn.version(),
                          gpu=torch.cuda.get_device_name(), seed=args.seed,
                          shape=list(inputs[0].shape), strides=[list(t.stride()) for t in inputs],
                          upstream_stride=list(upstream.stride()), bias_stride=list(bias.stride()),
                          mask="synthetic linear caption lengths including text-dropped row",
                          component_only=True)), flush=True)
    reference = evaluate(inputs, bias, upstream, SDPBackend.EFFICIENT_ATTENTION)
    for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            torch.cuda.synchronize()
            start = time.monotonic()
            actual = evaluate(inputs, bias, upstream, backend)
            torch.cuda.synchronize()
            cold_seconds = time.monotonic() - start
            agreement = {name: errors(a, r) for name, a, r in
                         zip(("output", "dq", "dk", "dv"), actual, reference)}
            del actual
            timings = measure(inputs, bias, upstream, backend, args.iterations)
            print(json.dumps(dict(backend=backend.name, cold_seconds=cold_seconds,
                                  agreement=agreement, **timings)), flush=True)
        except (RuntimeError, NotImplementedError) as error:
            print(json.dumps(dict(backend=backend.name, error=str(error))), flush=True)


if __name__ == "__main__":
    main()
