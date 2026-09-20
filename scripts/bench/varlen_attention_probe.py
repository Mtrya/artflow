"""Experimental native FlashAttention varlen probe; never selected by training.

Keep every query, pack only valid K/V tokens, and retain independent sequence
boundaries. This exactly represents key-padding semantics, including outputs
at padded query positions. Uses private PyTorch 2.9.1 CUDA operators; deployment
requires pinned-version, full-block, compile/DDP and gradient validation.

Packing/layout conversion and their gradients are included in timings. Mask
metadata is constructed once outside the repeated attention calls; report that
cost separately rather than implying it is free in a complete model forward.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.bench.attention_backend_probe import make_inputs, errors
from src.models.varlen_attention import mask_metadata, packed_attention, validate_runtime


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
    # This private ABI was inspected against precisely this target version.
    validate_runtime()
    torch.manual_seed(args.seed)
    inputs, bias, upstream = make_inputs(args.batch, args.sequence, args.image_tokens)
    torch.cuda.synchronize()
    started = time.monotonic()
    metadata = mask_metadata(bias[:, 0, 0] == 0)
    torch.cuda.synchronize()
    print(json.dumps(dict(torch_version=torch.__version__, seed=args.seed,
                          batch=args.batch, sequence=args.sequence,
                          packed_keys=metadata[0].numel(), metadata_seconds=time.monotonic()-started,
                          private_operator=True, component_only=True)), flush=True)

    def efficient():
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(*inputs, attn_mask=bias, dropout_p=0.)

    reference_out = efficient()
    reference_grad = torch.autograd.grad(reference_out, inputs, upstream)
    reference = (reference_out.detach(), *reference_grad)
    del reference_out, reference_grad
    for name, function in (("efficient", efficient),
                           ("native_flash_varlen_with_packing", lambda: packed_attention(*inputs, metadata))):
        out = function()
        grad = torch.autograd.grad(out, inputs, upstream)
        agreement = {key: errors(actual, expected) for key, actual, expected in
                     zip(("output", "dq", "dk", "dv"), (out, *grad), reference)}
        del out, grad
        for _ in range(4):
            out = function()
            torch.autograd.grad(out, inputs, upstream)
            del out
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iterations):
            start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            start.record()
            out = function()
            middle.record()
            grad = torch.autograd.grad(out, inputs, upstream)
            end.record()
            end.synchronize()
            times.append([start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)])
            del out, grad
        print(json.dumps(dict(variant=name, agreement=agreement,
                              forward_backward_total_medians_ms=[statistics.median(x) for x in zip(*times)],
                              samples_ms=times,
                              incremental_peak_bytes=torch.cuda.max_memory_allocated()-baseline)), flush=True)


if __name__ == "__main__":
    main()
