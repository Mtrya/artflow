#!/usr/bin/env python3
"""Generate hero bucket plans for all three resolution stages.

For each stage, expands logical dataset names into the actual precomputed
dataset directories (some logical sets are sharded into parts), splits the
normalized mix weight across shards proportional to row counts (read from
each shard's length_metadata.npz sidecar), then runs scripts/bench/
plan_buckets.py with the fitted DiT-only calibration and the end-to-end
VRAM budget.

The calibration measures DiT-only peaks. Per-stage planner budgets leave
headroom for optimizer states, EMA, text encoding, DDP and compile allocations;
full-stream memory validation remains required. Time alignment must preserve
the global mean-batch target at the selected rank count and accumulation.
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.environ.get("ARTFLOW_ROOT", REPO)
PY = sys.executable
CALIB = f"{W}/bucket_plans/calib-533m/merged.json"
OUTDIR = os.environ.get("HERO_PLAN_OUTDIR", f"{W}/bucket_plans/hero/batch-targets-0914")
VRAM_GB = {
    # DiT-only budget passed to the planner. The end-to-end peak is
    # fixed_e2e + m1 * B * L, where fixed_e2e (weights + optimizer + EMA +
    # text encoder + DDP buffers + per-shape compile workspaces, which
    # accumulate to ~2.5 GB once all ~100 bucket shapes are compiled) is
    # measured from validation runs: ~11.7 GB at 256p, ~12.7 at 640p,
    # ~13.7 at 896p. These budgets include empirical margin for fit error;
    # changing alignment never raises the per-bucket memory ceiling.
    "256p": 33.0,
    "640p": 30.0,
    "896p": 30.0,
}
NUM_LENGTH_BUCKETS = 20
PROGRESS_INTERVALS = {"256p": (0.0, 0.75), "640p": (0.75, 0.95), "896p": (0.95, 1.0)}
ACCUMULATION = {"256p": 1, "640p": 5, "896p": 7}
GLOBAL_BATCH_TARGETS = {"256p": 640, "640p": 512, "896p": 400}
RANKS = 8

# Normalized mix weights (%) per stage — notes/hero_recipe.md 2026-09-14.
MIX = {
    "256p": {
        "d1": 9.01, "d2-wikiart": 12.66, "d2-museum": 0.60,
        "d3-human": 10.50, "d3-people": 10.14, "d3-pexels": 5.53,
        "d3-synth-v2": 0.64, "d4-inat": 0.14, "d4-megalith": 0.33,
        "d4-pd12m": 8.00, "d4-vintage": 9.59, "d4-zimage": 2.23,
        "d4-relaion": 30.64,
    },
    "640p": {
        "d1": 10.82, "d2-wikiart": 12.32, "d2-museum": 0.78,
        "d3-human": 12.19, "d3-people": 11.76, "d3-pexels": 6.66,
        "d3-synth-v2": 0.78, "d4-megalith": 0.38,
        "d4-pd12m": 8.43, "d4-vintage": 9.41, "d4-zimage": 3.22,
        "d4-relaion": 23.24,
    },
    "896p": {
        "d1": 17.36, "d2-wikiart": 11.45, "d2-museum": 0.52,
        "d3-human": 15.75, "d3-people": 15.14, "d3-pexels": 10.74,
        "d4-megalith": 0.13, "d4-pd12m": 11.17, "d4-vintage": 1.57,
        "d4-relaion": 16.16,
    },
}

IMAGE_TOKENS = {
    "256p": {"1": 256, "2": 252, "3": 252, "4": 252, "5": 252},
    "640p": {"1": 1600, "2": 1590, "3": 1590, "4": 1610, "5": 1610},
    "896p": {"1": 3136, "2": 3108, "3": 3108, "4": 3072, "5": 3072},
}


def dataset_dirs(name: str, stage: str):
    """Logical dataset -> actual precomputed dirs at this stage."""
    if name == "d3-people" and stage == "640p":
        return [f"d3-people-a@{stage}", f"d3-people-b@{stage}"]
    if name == "d4-relaion" and stage in ("640p", "896p"):
        return [f"d4-relaion-p{i}@{stage}" for i in range(5)]
    return [f"{name}@{stage}"]


def sidecar_rows(path: str) -> int:
    n = np.load(os.path.join(path, "length_metadata.npz"))
    return len(n["caption_offsets"]) - 1


def main():
    only = set(sys.argv[1:])
    os.makedirs(OUTDIR, exist_ok=True)
    for stage, weights in MIX.items():
        if only and stage not in only:
            continue
        dataset_args = []
        print(f"== {stage} ==")
        for name, w in weights.items():
            if w <= 0:
                continue
            dirs = dataset_dirs(name, stage)
            paths = [f"{W}/precomputed_dataset/{d}" for d in dirs]
            for p in paths:
                if not os.path.exists(os.path.join(p, "length_metadata.npz")):
                    sys.exit(f"MISSING sidecar: {p}")
            rows = [sidecar_rows(p) for p in paths]
            total = sum(rows)
            for p, r in zip(paths, rows):
                wd = w * r / total
                dataset_args.append(f"{p}:{wd:.6f}")
                print(f"  {os.path.basename(p):28s} rows={r:8d} w={wd:.4f}%")
        out = f"{OUTDIR}/hero-{stage}-k{NUM_LENGTH_BUCKETS}.json"
        report = f"{OUTDIR}/hero-{stage}-k{NUM_LENGTH_BUCKETS}.report.md"
        cmd = [
            PY, "-m", "scripts.bench.plan_buckets",
            *(a for d in dataset_args for a in ("--dataset", d)),
            "--calibration", CALIB,
            "--image-tokens", json.dumps(IMAGE_TOKENS[stage]),
            "--buckets", str(NUM_LENGTH_BUCKETS),
            "--vram-budget-gb", str(VRAM_GB[stage]),
            "--min-batch", "2",
            "--max-batch", "256",
            "--min-mean-batch", str(GLOBAL_BATCH_TARGETS[stage] / (RANKS * ACCUMULATION[stage])),
            "--length-cap", "2048",
            "--caption-policy", "beta",
            "--caption-beta-start", "-1", "--caption-beta-end", "1",
            "--caption-short-reserve", "0.20", "--caption-short-threshold", "256",
            "--caption-schedule", "linear",
            "--progress-start", str(PROGRESS_INTERVALS[stage][0]),
            "--progress-end", str(PROGRESS_INTERVALS[stage][1]),
            "--progress-grid", "8",
            "--out", out,
            "--report", report,
        ]
        env = dict(os.environ, PYTHONPATH=REPO)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout[-3000:])
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-3000:])
            sys.exit(f"plan_buckets failed for {stage}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
