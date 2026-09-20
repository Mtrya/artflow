"""Cost the frozen three-stage hero schedule from supplied eight-rank measurements.

This is accounting, not a measurement or readiness verifier. Input JSON requires
world_size=8, an evidence description, recovery_gpu_hours, allocated_idle_gpu_hours,
final_kid_seconds, and stages keyed by 256p/640p/896p. Each stage requires
step_seconds, startup_seconds, checkpoint_seconds, grid_seconds, loss_probe_seconds.
Use conservative representative stage rates, including normal optimizer/EMA/health
and logging costs. Startup includes compilation excess above those rates and
model/data/evaluation initialization, but excludes separately counted grid/probe work.
Checkpoint, grid, probe and KID durations are whole-allocation wall seconds,
including waiting ranks. Recovery/allocated idle are separate GPU-hour bounds.
Never enter a lower-rank extrapolation or missing overhead as measured evidence.
The budget must be explicit; GPU-hour budgets from different devices are not
interchangeable. Use measurements from the hardware being budgeted.
"""

import argparse
import json
import math
from pathlib import Path

STAGES = ("256p", "640p", "896p")
FIELDS = ("step_seconds", "startup_seconds", "checkpoint_seconds",
          "grid_seconds", "loss_probe_seconds")


def validate(measurements):
    if measurements.get("world_size") != 8:
        raise ValueError("hero costing requires supplied eight-rank measurements")
    if not isinstance(measurements.get("evidence"), str) or not measurements["evidence"].strip():
        raise ValueError("describe the measurement artifacts, uncertainty and accounting scope")
    if set(measurements["stages"]) != set(STAGES):
        raise ValueError("supply exactly 256p, 640p and 896p costs")
    values = [measurements[key] for key in
              ("recovery_gpu_hours", "allocated_idle_gpu_hours", "final_kid_seconds")]
    values += [measurements["stages"][stage][key] for stage in STAGES for key in FIELDS]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("all costs must be explicit finite nonnegative numbers")
    if any(measurements["stages"][stage]["step_seconds"] <= 0 for stage in STAGES):
        raise ValueError("step_seconds must be positive")


def stage_counts(start, end, *, incoming):
    """Scalar or NumPy-array arithmetic; overlaps trigger once, as in the trainer.

    Launch policy is jobs/hero_stage.sh plus configs/hero.toml: checkpoint 2k,
    grid 10k/end/incoming/+2k, loss probe 500 plus one baseline per stage.
    Incoming image baseline is distinct
    from the previous stage's endpoint image because the resolution differs.
    """
    checkpoints = end // 2000 - start // 2000 + (end % 2000 != 0)
    grids = end // 10000 - start // 10000 + (end % 10000 != 0)
    if incoming:
        grids = grids + 1 + ((start + 2000 < end) & ((start + 2000) % 10000 != 0))
    return dict(updates=end - start, checkpoints=checkpoints, grids=grids,
                loss_probes=1 + end // 500 - start // 500)


def _cost(total, measurements, *, details=False):
    endpoints = (0, total * 75 // 100, total * 95 // 100, total)
    wall = measurements["final_kid_seconds"]
    stages = {}
    for i, stage in enumerate(STAGES):
        counts = stage_counts(endpoints[i], endpoints[i + 1], incoming=i > 0)
        costs = measurements["stages"][stage]
        components = dict(
            training_seconds=counts["updates"] * costs["step_seconds"],
            startup_seconds=costs["startup_seconds"],
            checkpoint_seconds=counts["checkpoints"] * costs["checkpoint_seconds"],
            grid_seconds=counts["grids"] * costs["grid_seconds"],
            loss_probe_seconds=counts["loss_probes"] * costs["loss_probe_seconds"],
        )
        wall = wall + sum(components.values())
        if details:
            stages[stage] = dict(start=endpoints[i], stop=endpoints[i + 1],
                                 counts=counts, **components)
    reserved = measurements["recovery_gpu_hours"] + measurements["allocated_idle_gpu_hours"]
    gpu_hours = wall * 8 / 3600 + reserved
    if not details:
        return gpu_hours
    return dict(total_steps=total, stages=stages, gpu_hours=gpu_hours,
                allocated_wall_hours=gpu_hours / 8, recovery_and_idle_gpu_hours=reserved,
                optimizer_steps_per_gpu_hour=total / gpu_hours,
                optimizer_steps_per_allocated_wall_hour=total * 8 / gpu_hours,
                gpu_hours_per_1000_steps=gpu_hours * 1000 / total,
                production_steps_selected=False,
                final_kid_seconds=measurements["final_kid_seconds"],
                evidence=measurements["evidence"], readiness_established=False,
                caveat="Conditional cost from supplied measurements, not independently verified "
                       "throughput, saturation or Stage 5 readiness. Queue wall time is not "
                       "allocated wall time; recovery/idle assume eight allocated GPUs. "
                       "Ratios include this illustrative horizon's overhead and need not "
                       "remain constant at a different horizon; production T is not selected.")


def cost(total, measurements):
    validate(measurements)
    if type(total) is not int or total < 5:
        raise ValueError("total must be an integer producing three nonempty stages (>=5)")
    return _cost(total, measurements, details=True)


def maximize(measurements, budget):
    """Search every feasible integer T, not a monotonic binary-search assumption.

    Rounding and overlapping eval triggers can make cost locally nonmonotonic.
    Chunked CPU vectorization finds the actual largest feasible integer under
    the supplied cost model without launching experiments or allocating GPUs.
    """
    import numpy as np

    validate(measurements)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be finite and positive")
    minimum = min(measurements["stages"][s]["step_seconds"] for s in STAGES)
    upper = math.floor(budget * 3600 / (8 * minimum))
    best = None
    for low in range(5, upper + 1, 100000):
        totals = np.arange(low, min(low + 100000, upper + 1), dtype=np.int64)
        feasible = totals[_cost(totals, measurements) <= budget]
        if feasible.size:
            best = int(feasible[-1])
    return None if best is None else cost(best, measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--budget-gpu-hours", type=float, required=True,
                        help="Explicit budget for the measured device (hero: 800 H100/H200 GPU-hours)")
    parser.add_argument("--steps", type=int, default=400000)
    args = parser.parse_args()
    measurements = json.loads(args.measurements.read_text())
    print(json.dumps(dict(requested=cost(args.steps, measurements),
                          maximum=maximize(measurements, args.budget_gpu_hours)), indent=2))


if __name__ == "__main__":
    main()
