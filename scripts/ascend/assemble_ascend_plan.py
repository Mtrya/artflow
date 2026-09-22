#!/usr/bin/env python
"""Assemble Ascend (910B, 64GiB) hero bucket plans.

Boundaries are reused verbatim from the 4090 batch-targets-0914 plans (the
length distribution and model shape are unchanged). Only per-bucket batch
sizes are re-derived for the eager-mode NPU memory model measured by
ascend_bench_sweep3.sh:

    peak_gib ~= static[img] + slope[img] * bs * (img_tokens + max_length)
    step_time ~= t0[img] + t1[img] * bs * (img_tokens + max_length)

bs is set to the largest value whose predicted peak stays under --budget,
then (optionally) nudged down one notch if predicted step time lands far
above the per-stage median (time alignment across DDP ranks).

Usage:
    .venv/bin/python scripts/ascend/assemble_ascend_plan.py \
        --sweep-log sweep3.log --plans-dir bucket_plans/hero/batch-targets-0914 \
        --out-dir bucket_plans/hero/ascend-0922
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

SWEEP_RE = re.compile(
    r"SWEEP img=(\d+) tl=(\d+) bs=(\d+): ([\d.]+)s [\d.]+spl/s [\d.]+ktok/s peak=([\d.]+)GiB OK"
)


def parse_sweep(text: str):
    """-> list of (img_tokens, tl, bs, step_s, peak_gib) for OK points."""
    rows = []
    for m in SWEEP_RE.finditer(text):
        img, tl, bs, dt, pk = m.groups()
        rows.append((int(img), int(tl), int(bs), float(dt), float(pk)))
    return rows


def fit(rows, budget):
    """Per image class: linear memory/time models over tokens = bs*S."""
    models = {}
    for img in sorted({r[0] for r in rows}):
        pts = [r for r in rows if r[0] == img]
        x = np.array([r[2] * (r[0] + r[1]) for r in pts], dtype=float)
        mem = np.array([r[4] for r in pts])
        tim = np.array([r[3] for r in pts])
        A = np.stack([np.ones_like(x), x], 1)
        (m0, m1), *_ = np.linalg.lstsq(A, mem, rcond=None)
        (t0, t1), *_ = np.linalg.lstsq(A, tim, rcond=None)
        models[img] = {"static": m0, "slope": m1, "t0": t0, "t1": t1,
                       "points": len(pts),
                       "mem_r2": _r2(A @ [m0, m1], mem),
                       "time_r2": _r2(A @ [t0, t1], tim)}
    return models


def _r2(pred, y):
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-log", required=True)
    ap.add_argument("--plans-dir", default="bucket_plans/hero/batch-targets-0914")
    ap.add_argument("--out-dir", default="bucket_plans/hero/ascend-0922")
    ap.add_argument("--budget", type=float, default=54.0,
                    help="peak GiB ceiling per bucket (64GiB card, headroom for "
                         "optimizer states + fragmentation)")
    ap.add_argument("--min-bs", type=int, default=2)
    args = ap.parse_args()

    rows = parse_sweep(Path(args.sweep_log).read_text())
    if not rows:
        raise SystemExit("no OK sweep points found in log")
    models = fit(rows, args.budget)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for plan_path in sorted(Path(args.plans_dir).glob("hero-*-k20.json")):
        plan = json.loads(plan_path.read_text())
        new_plan = {}
        for res_id, buckets in plan.items():
            # image tokens per res id come from the 0914 report; fall back to
            # the nearest measured image class.
            new_buckets = []
            for b in buckets:
                max_len = b["max_length"]
                # nearest image class by token count is resolved by the caller
                # via --image-tokens; here each res id's image tokens are read
                # from the sibling report file.
                new_buckets.append({"max_length": max_len,
                                    "batch_size": b["batch_size"]})
            new_plan[res_id] = new_buckets
        summary.append((plan_path.name, plan, new_plan))

    # image tokens per (plan, res id) from the 0914 reports
    for name, old_plan, new_plan in summary:
        report = Path(args.plans_dir) / name.replace(".json", ".report.md")
        m = re.search(r"image tokens \| `(\{[^}]+\})`", report.read_text())
        img_tokens = {int(k): int(v) for k, v in json.loads(m.group(1)).items()}
        stage_times = []
        bs_map = {}
        for res_id, buckets in new_plan.items():
            itok = img_tokens[int(res_id)]
            model = models[min(models, key=lambda k: abs(k - itok))]
            for b in buckets:
                S = itok + b["max_length"]
                bs_cap = int((args.budget - model["static"]) / (model["slope"] * S))
                bs_cap = max(args.min_bs, bs_cap)
                b["batch_size"] = bs_cap
                t = model["t0"] + model["t1"] * bs_cap * S
                stage_times.append(t)
                bs_map[(res_id, b["max_length"])] = (bs_cap, t)
        out_path = out_dir / name
        out_path.write_text(json.dumps(new_plan, indent=2))
        times = np.array(stage_times)
        print(f"{name}: buckets={len(stage_times)} "
              f"bs range {min(b['batch_size'] for bs in new_plan.values() for b in bs)}"
              f"..{max(b['batch_size'] for bs in new_plan.values() for b in bs)} "
              f"pred step time median={np.median(times):.3f}s "
              f"p5={np.percentile(times, 5):.3f} p95={np.percentile(times, 95):.3f}")
    print("models:")
    for img, mo in models.items():
        print(f"  img={img}: static={mo['static']:.2f}GiB slope={mo['slope']*1e6:.0f}KiB/tok "
              f"mem_r2={mo['mem_r2']:.3f} time_r2={mo['time_r2']:.3f} ({mo['points']} pts)")


if __name__ == "__main__":
    main()
