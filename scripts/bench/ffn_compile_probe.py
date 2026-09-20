"""Test compiler padding of the unchanged 1152 -> 3075 hero FFN.

This is a component diagnostic, not end-to-end throughput. No parameter
dimensions or model weights are changed by the experiment.
"""

import argparse
import copy
import json
import statistics
import time

import torch

from src.models.dit_blocks import GatedFeedForward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.tokens < 1 or args.iterations < 1:
        parser.error("tokens and iterations must be positive")
    torch.manual_seed(42)
    model = GatedFeedForward(1152, int(1152 * 2.67)).cuda().bfloat16()
    x = torch.randn(args.tokens, 1152, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    reference = None
    for force_pad in (False, True):
        live = copy.deepcopy(model)
        compiled = torch.compile(live, dynamic=False, options={"force_shape_pad": force_pad})

        def step():
            live.zero_grad(set_to_none=True)
            x.grad = None
            output = compiled(x)
            output.backward(dy)
            return output

        torch.cuda.synchronize()
        start = time.monotonic()
        y = step()
        torch.cuda.synchronize()
        cold_seconds = time.monotonic() - start
        values = [y.detach().float(), x.grad.detach().float(),
                  *[p.grad.detach().float() for p in live.parameters()]]
        errors = None
        if reference is None:
            reference = [value.cpu() for value in values]
        else:
            errors = [float((value - ref.to(value.device)).norm() /
                            ref.norm().clamp_min(1e-12).to(value.device))
                      for value, ref in zip(values, reference)]
        del values, y
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iterations):
            start = time.monotonic()
            step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
        print(json.dumps(dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                              force_shape_pad=force_pad, tokens=args.tokens,
                              cold_seconds=cold_seconds, mean_seconds=statistics.mean(times),
                              median_seconds=statistics.median(times),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              relative_l2_output_inputgrad_paramgrads=errors)), flush=True)
        del compiled, live


if __name__ == "__main__":
    main()
