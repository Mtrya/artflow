"""Isolate gradient-division and EMA batching on the actual hero parameter shapes.

Full-training traces show foreach EMA slower despite fewer launches. This probe
compares existing implementations, without changing arithmetic or adding kernels.
No model forward, optimizer, data or DDP is timed; no end-to-end rate is claimed.
"""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import torch

from src.models.artflow import ArtFlow
from src.train.config import load_config
from src.train.update_ops import divide_gradients, update_ema


def measure_health(model, config, args):
    """Measure the existing periodic diagnostics, not an optimizer-step proxy."""
    from src.train.health import (
        snapshot_weights, update_weight_ratios, qk_gain_stats, ema_rel_distance,
    )
    from src.train.muon import build_param_groups
    optimizers = build_param_groups(model, muon_lr=config.optim.muon_lr,
                                   muon_wd=config.optim.muon_wd,
                                   adam_lr=config.optim.learning_rate,
                                   muon_momentum=config.optim.muon_momentum)
    ema = copy.deepcopy(model)

    def once(device="cpu"):
        samples = {}
        def timed(name, fn):
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.profiler.record_function(name):
                value = fn()
            torch.cuda.synchronize()
            samples[name] = (time.monotonic() - start) * 1000
            return value
        snapshots = timed("health_snapshot", lambda: snapshot_weights(optimizers, device=device))
        # Synthetic nonzero update is deliberately outside diagnostic timings.
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1e-4)
        ratios = timed("health_update_ratios", lambda: update_weight_ratios(snapshots))
        del snapshots
        gains = timed("health_qk_gains", lambda: qk_gain_stats(model))
        distance = timed("health_ema_distance", lambda: ema_rel_distance(ema, model))
        if gains is None or not all(math.isfinite(x) for x in [*ratios, *gains, distance]):
            raise RuntimeError("missing or nonfinite health diagnostics")
        samples["total_ms"] = sum(samples.values())
        return dict(samples_ms=samples, ratios=ratios, gains=gains, ema_distance=distance)

    devices = ("cpu", "cuda") if args.compare_health_snapshot else ("cpu",)
    if args.compare_health_snapshot:
        reference = snapshot_weights(optimizers)
        candidate = snapshot_weights(optimizers, device="cuda")
        for ref_group, candidate_group in zip(reference, candidate):
            if not all(torch.equal(a[1], b[1].cpu()) for a, b in zip(ref_group, candidate_group)):
                raise RuntimeError("device snapshot values differ")
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1e-4)
        expected, actual = update_weight_ratios(reference), update_weight_ratios(candidate)
        if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-10)
                   for a, b in zip(expected, actual)):
            raise RuntimeError("device snapshot changes health ratios beyond tolerance")
        print(json.dumps(dict(exact_snapshot_values=True, ratio_reference=expected,
                              ratio_candidate=actual, ratio_rtol=1e-6, ratio_atol=1e-10)), flush=True)
        del reference, candidate, ref_group, candidate_group
    for device in devices:
        once(device)  # Warm up outside the measured samples.
    for round_index in range(args.rounds if args.compare_health_snapshot else 1):
        for device in (devices if round_index % 2 == 0 else devices[::-1]):
            torch.cuda.reset_peak_memory_stats()
            rows = [once(device) for _ in range(args.iterations)]
            medians = {key: statistics.median(row["samples_ms"][key] for row in rows)
                       for key in rows[0]["samples_ms"]}
            interval = config.telemetry.health_interval
            print(json.dumps(dict(operation="health", component_only=True,
                                  snapshot_device=device, round=round_index,
                                  parameters=sum(p.numel() for p in model.parameters()),
                                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                                  threads=torch.get_num_threads(), health_interval=interval,
                                  medians_ms=medians, samples=rows,
                                  amortized_median_ms=medians["total_ms"] / interval if interval else None,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  allocated_after_diagnostics_bytes=torch.cuda.memory_allocated(),
                                  caveat="Synthetic update; no training, DDP or full-pipeline contention. "
                                         "Metric definitions/cadence unchanged; weight-norm reduction "
                                         "runs on the selected snapshot device.")), flush=True)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
        for device in devices:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                once(device)
            prof.export_chrome_trace(str(args.trace_dir / f"health-{device}.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--compare-health-snapshot", action="store_true")
    parser.add_argument("--trace-dir", type=Path,
                        help="Separate untimed health diagnostic trace")
    args = parser.parse_args()
    if min(args.iterations, args.rounds) < 1:
        parser.error("positive iteration/round counts required")
    if args.compare_health_snapshot and not args.health_only:
        parser.error("--compare-health-snapshot requires --health-only")
    torch.manual_seed(42)
    config = load_config(args.config)
    model = ArtFlow(**asdict(config.model)).cuda()
    if args.health_only:
        measure_health(model, config, args)
        return
    models = [model, copy.deepcopy(model)]
    emas = [copy.deepcopy(model), copy.deepcopy(model)]
    params = [list(m.parameters()) for m in models]
    with torch.no_grad():
        for ema in emas:
            for p in ema.parameters():
                p.zero_()
    divisor = torch.tensor(123.5, device="cuda", dtype=torch.float32)
    decay = config.train.ema_decay
    print(json.dumps(dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                          parameters=sum(p.numel() for p in model.parameters()),
                          dtype=str(next(model.parameters()).dtype), decay=decay,
                          component_only=True)), flush=True)

    # Independent random gradients; exactness checks are outside timed regions.
    for index in range(3):
        for p, q in zip(*params):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        for variant in (0, 1):
            divide_gradients(params[variant], divisor, foreach=bool(variant))
            update_ema(emas[variant], models[variant], decay, foreach=bool(variant))
        for p, q in zip(*params):
            if not torch.equal(p.grad, q.grad) or not bool(torch.isfinite(q.grad).all()):
                raise RuntimeError("gradient division differs or is nonfinite")
        for p, q in zip(emas[0].parameters(), emas[1].parameters()):
            if not torch.equal(p, q) or not bool(torch.isfinite(q).all()):
                raise RuntimeError("EMA differs or is nonfinite")
        print(json.dumps(dict(check=index, exact_finite=True)), flush=True)

    def operation(kind, variant):
        if kind == "gradient_division":
            divide_gradients(params[variant], divisor, foreach=bool(variant))
        else:
            update_ema(emas[variant], models[variant], decay, foreach=bool(variant))

    for kind in ("gradient_division", "ema"):
        for round_index in range(args.rounds):
            for variant in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                for _ in range(10):
                    operation(kind, variant)
                torch.cuda.synchronize()
                wall_samples, gpu_samples = [], []
                for _ in range(args.iterations):
                    # Reset outside timing to avoid underflow/zero-gradient benchmarking.
                    if kind == "gradient_division":
                        for p in params[variant]:
                            p.grad.fill_(1.25)
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    before = time.monotonic()
                    start.record()
                    operation(kind, variant)
                    end.record()
                    end.synchronize()
                    wall_samples.append((time.monotonic() - before) * 1000)
                    gpu_samples.append(start.elapsed_time(end))
                print(json.dumps(dict(operation=kind, foreach=bool(variant), round=round_index,
                                      wall_median_ms=statistics.median(wall_samples),
                                      gpu_median_ms=statistics.median(gpu_samples),
                                      wall_samples_ms=wall_samples, gpu_samples_ms=gpu_samples)),
                      flush=True)


if __name__ == "__main__":
    main()
