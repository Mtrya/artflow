"""Measure repeated-backward determinism of the actual native attention path.

This isolates kernel numerics, not checkpoint restoration or training quality.
No production deterministic setting is changed by running this separate process.
"""
import argparse
import json
import time

import torch

from src.models.varlen_attention import mask_metadata, packed_attention, validate_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats required")
    validate_runtime()
    torch.manual_seed(42)
    for batch, image_tokens, text_tokens in ((70, 252, 154), (14, 1610, 171)):
        sequence = image_tokens + text_tokens
        shape = (batch, 16, sequence, 72)
        values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        output_gradient = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        keep = torch.ones(batch, sequence, device="cuda", dtype=torch.bool)
        lengths = torch.arange(batch, device="cuda") % text_tokens + 1
        keep[:, image_tokens:] = torch.arange(text_tokens, device="cuda")[None, :] < lengths[:, None]
        metadata = mask_metadata(keep)
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            for compiled in (False, True):
                def attention(q, k, v):
                    return packed_attention(q, k, v, metadata)
                function = torch.compile(attention, fullgraph=True, dynamic=True) if compiled else attention
                first = None
                report = dict(shape=shape, deterministic=deterministic, compiled=compiled,
                              repeats=args.repeats, exact=True, max_absolute_differences=[0., 0., 0.],
                              max_relative_l2_differences=[0., 0., 0.], finite=True)
                start = time.monotonic()
                try:
                    for index in range(args.repeats):
                        q, k, v = [value.detach().requires_grad_(True) for value in values]
                        output = function(q, k, v)
                        gradients = torch.autograd.grad(output, (q, k, v), output_gradient)
                        report["finite"] &= all(bool(torch.isfinite(g).all()) for g in gradients)
                        if first is None:
                            first = [g.detach().clone() for g in gradients]
                            continue
                        for position, (a, b) in enumerate(zip(first, gradients)):
                            report["exact"] &= torch.equal(a, b)
                            difference = a.float() - b.float()
                            report["max_absolute_differences"][position] = max(
                                report["max_absolute_differences"][position], float(difference.abs().max()))
                            report["max_relative_l2_differences"][position] = max(
                                report["max_relative_l2_differences"][position],
                                float(difference.norm() / a.float().norm().clamp_min(1e-30)))
                except RuntimeError as error:
                    report["error"] = str(error)
                    report["exact"] = False
                report["elapsed_seconds"] = time.monotonic() - start
                print(json.dumps(report), flush=True)
                del first, function
    torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    main()
