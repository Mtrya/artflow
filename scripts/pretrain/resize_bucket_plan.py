"""Draft fixed-boundary batch/accumulation candidates for device validation.

Use the existing beta-policy metadata weighting over the stage interval. Integer
sizes are scaled proportionally toward the requested harmonic-mean global batch,
rounded up, then trimmed without going below that target. This is a screening
proposal, not a memory model or throughput optimizer. Realized finite-queue
exposure, memory tails and measured speed still require device validation.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from scripts.pretrain.plan_buckets import (
    load_sidecar_lengths, mean_emitted_batch, resolution_lengths,
)
from src.dataset.captions import CaptionPolicy
from src.dataset.mix import parse_dataset_mix
from src.pretrain.config import load_config, flatten


def resize(sizes, shares, *, old_accumulation, new_accumulation, ranks, target):
    for value in (*sizes, old_accumulation, new_accumulation, ranks):
        if type(value) is not int or value < 1:
            raise ValueError("sizes, accumulations and ranks must be positive integers")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    if (len(sizes) != len(shares) or not sizes or
            any(not math.isfinite(p) or p < 0 for p in shares) or sum(shares) <= 0):
        raise ValueError("need matching sizes and finite nonnegative shares with positive mass")
    factor = ranks * new_accumulation
    reference_micro = mean_emitted_batch(shares, sizes)
    scale = min(old_accumulation / new_accumulation, target / (factor * reference_micro))
    # Remove the reference plan's overshoot uniformly, rather than concentrating
    # the entire difference in whichever single bucket has the largest mass.
    proposed = [min(math.ceil(b * scale),
                    (b * old_accumulation + new_accumulation - 1) // new_accumulation)
                for b in sizes]
    initial = mean_emitted_batch(shares, proposed) * factor
    if initial < target and not math.isclose(initial, target, rel_tol=1e-12):
        raise ValueError("scaled proposal cannot meet target; do not silently enlarge it")
    # Each accepted decrement approaches the target without crossing below it.
    # No claim of globally optimal latency: preserve the reference shape balance
    # approximately, and let the subsequent device comparison decide.
    while True:
        best = None
        best_mean = mean_emitted_batch(shares, proposed) * factor
        for i, (b, p) in enumerate(zip(proposed, shares)):
            if b <= 1 or p == 0:
                continue
            trial = proposed.copy()
            trial[i] -= 1
            value = mean_emitted_batch(shares, trial) * factor
            if (value >= target or math.isclose(value, target, rel_tol=1e-12)) and value < best_mean:
                best, best_mean = i, value
        if best is None:
            return proposed
        proposed[best] -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--old-accumulation", type=int, required=True)
    parser.add_argument("--new-accumulation", type=int, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--target-global-batch", type=float, required=True)
    parser.add_argument("--progress-start", type=float, required=True)
    parser.add_argument("--progress-end", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg.data
    stage = cfg.stage(args.stage)
    policy = CaptionPolicy(kind="beta", beta_start=data.caption_beta_start,
        beta_end=data.caption_beta_end, short_reserve=data.caption_short_reserve,
        short_threshold=data.caption_short_threshold, schedule="linear")
    raw = json.loads(Path(stage.bucket_plan).read_text())
    pooled = resolution_lengths(load_sidecar_lengths(parse_dataset_mix(flatten(cfg, args.stage)["dataset_mix"]),
        policy=policy, progress_start=args.progress_start,
        progress_end=args.progress_end, progress_grid=8))
    if set(map(int, raw)) != set(pooled):
        raise ValueError("plan and metadata aspect IDs differ")
    keys, sizes, shares = [], [], []
    for aspect in sorted(raw, key=int):
        buckets = raw[aspect]
        bounds = [b["max_length"] for b in buckets]
        if not bounds or bounds != sorted(set(bounds)) or bounds[0] < 1:
            raise ValueError("plan boundaries must be positive and strictly increasing")
        values = pooled[int(aspect)]
        assignments = np.searchsorted(bounds, values.lengths)
        if np.any(assignments >= len(bounds)):
            raise ValueError("plan does not cover retained caption lengths")
        masses = np.bincount(assignments, weights=values.weights, minlength=len(bounds))
        for i, bucket in enumerate(buckets):
            keys.append((aspect, i))
            sizes.append(bucket["batch_size"])
            shares.append(float(masses[i]))
    proposed = resize(sizes, shares, old_accumulation=args.old_accumulation,
        new_accumulation=args.new_accumulation, ranks=args.ranks,
        target=args.target_global_batch)
    result = copy.deepcopy(raw)
    for (aspect, i), size in zip(keys, proposed):
        result[aspect][i]["batch_size"] = size
    report = dict(accepted=False, memory_validated=False, throughput_measured=False,
        source_plan=stage.bucket_plan, configs=args.config,
        old_accumulation=args.old_accumulation, new_accumulation=args.new_accumulation,
        ranks=args.ranks, target_global_batch=args.target_global_batch,
        estimated_reference_global_batch=mean_emitted_batch(shares, sizes)*args.ranks*args.old_accumulation,
        estimated_candidate_global_batch=mean_emitted_batch(shares, proposed)*args.ranks*args.new_accumulation,
        progress_interval=[args.progress_start,args.progress_end], progress_grid=8,
        caveat="Equal-progress averaged probabilities; finite queue exposure and memory need validation.",
        buckets=[dict(aspect=a, index=i, draw_share=p, old_batch=b, candidate_batch=n)
                 for (a,i),p,b,n in zip(keys,shares,sizes,proposed)])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir/"plan.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.out_dir/"proposal.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="buckets"}, indent=2))


if __name__ == "__main__":
    main()
