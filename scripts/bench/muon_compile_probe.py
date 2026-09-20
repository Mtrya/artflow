"""Bounded component probe of compilation for the unchanged Muon NS iteration.

Derive matrix batches from the actual model routing, without allocating model
weights. This does not modify the training optimizer or qualify hero throughput.
Precision-cast emulation preserves explicit bf16 rounding boundaries, but the
measured numerical differences must still be reviewed before any adoption.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import statistics
import time

import torch

from src.models.artflow import ArtFlow
from src.train.config import load_config
from src.train.muon import Muon, build_param_groups, _zeropower_via_newtonschulz5_batched


def batch_shapes(model):
    """Match the per-chunk-group, then per-matrix-shape grouping in Muon.step."""
    result = []
    for optimizer in build_param_groups(model, muon_lr=.02):
        if not isinstance(optimizer, Muon):
            continue
        for group in optimizer.param_groups:
            chunks = group["chunks"]
            counts = Counter()
            for param in group["params"]:
                rows, cols = param.shape
                if rows % chunks:
                    raise ValueError("invalid optimizer chunk routing")
                counts[(rows // chunks, cols)] += chunks
            for (rows, cols), count in counts.items():
                # The trainer uses the scalar path for a one-entry group.
                if sum(counts.values()) <= 1:
                    continue
                result.append((count, rows, cols))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    config = load_config(args.config)
    with torch.device("meta"):
        model = ArtFlow(**asdict(config.model))
    shapes = batch_shapes(model)
    print(json.dumps(dict(model_parameters=sum(p.numel() for p in model.parameters()),
                          batch_shapes=shapes, component_only=True)), flush=True)
    if args.list_only:
        return
    torch.manual_seed(args.seed)
    eager = _zeropower_via_newtonschulz5_batched
    compiled = torch.compile(eager, fullgraph=True, dynamic=False,
                             options={"emulate_precision_casts": True,
                                      "shape_padding": False})
    for shape in shapes:
        g = torch.randn(shape, device="cuda", dtype=torch.float32)
        reference = torch.stack(eager(g))
        for label, function in (("eager", eager), ("compiled", compiled)):
            torch.cuda.synchronize()
            start = time.monotonic()
            actual = torch.stack(function(g))
            torch.cuda.synchronize()
            cold = time.monotonic() - start
            difference = actual.float() - reference.float()
            errors = dict(exact=torch.equal(actual, reference),
                          max_abs=float(difference.abs().max()),
                          relative_l2=float(difference.norm() /
                                            reference.float().norm().clamp_min(1e-12)))
            del actual, difference
            for _ in range(3):
                function(g)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(args.iterations):
                start = time.monotonic()
                output = function(g)
                torch.cuda.synchronize()
                timings.append(time.monotonic() - start)
                del output
            print(json.dumps(dict(variant=label, shape=shape, seed=args.seed,
                                  torch_version=torch.__version__,
                                  gpu=torch.cuda.get_device_name(), cold_seconds=cold,
                                  median_seconds=statistics.median(timings),
                                  mean_seconds=statistics.mean(timings), errors=errors,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated())),
                  flush=True)
        del g, reference


if __name__ == "__main__":
    main()
