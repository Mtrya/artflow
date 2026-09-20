"""Bounded full-pipeline probes; never launches a hero run.

Run inside the selected GPU job image from a pinned repository snapshot.
Stage policies sample a specified fraction of their global curriculum interval
(midpoint by default), not an independently restarted short-caption schedule.
Use separately labelled fractions to measure within-stage distribution changes;
no single slice proves an entire stage's throughput.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage-config-dir", type=Path, default=Path("configs/stages"))
    parser.add_argument("--ranks", type=int, choices=range(1, 9), default=8,
                        help="Lower ranks diagnose bottlenecks; only 8 validates hero rates")
    parser.add_argument("--stage", action="append", choices=["256p", "640p", "896p"])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--accumulation", type=int,
                        help="Explicit diagnostic override; does not change hero defaults")
    parser.add_argument("--bucket-plan", type=Path,
                        help="Single-stage candidate plan; never edits the original config")
    parser.add_argument("--stage-timeout", type=int, default=2200)
    parser.add_argument("--curriculum-fraction", type=float, default=0.5,
                        help="Position within each stage's global interval, from 0 to 1")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--cpu-wall-profile", action=argparse.BooleanOptionalAction, default=True,
                        help="Optional diagnostic instrumentation; record the final chosen policy")
    parser.add_argument("--gpu-health-snapshot", action="store_true")
    parser.add_argument("--foreach-updates", action="store_true")
    parser.add_argument("--local-cache-clear", action="store_true")
    parser.add_argument("--dynamic-blocks", action="store_true")
    parser.add_argument("--autotune-blocks", action="store_true")
    parser.add_argument("--native-flash-varlen", action="store_true")
    parser.add_argument("--real-rope", action="store_true")
    parser.add_argument("--hoist-double-rope", action="store_true")
    parser.add_argument("--muon-compile-square-ns", action="store_true")
    parser.add_argument("--disable-ddp-compile-split", action="store_true")
    parser.add_argument("--ddp-gradient-bucket-views", action="store_true")
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--cache-clear-interval", type=int, default=None,
                         help="Explicit diagnostic cadence; otherwise inherit the config")
    cleanup.add_argument("--no-cache-clear", action="store_const", const=0,
                         dest="cache_clear_interval", help="Disable periodic allocator cleanup")
    parser.add_argument("--record-metrics", action="store_true")
    parser.add_argument("--trace-start", type=int, default=-1)
    parser.add_argument("--trace-steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps <= 50 or args.stage_timeout <= 0:
        parser.error("need >50 steps and a positive timeout")
    if args.cache_clear_interval is not None and args.cache_clear_interval < 0:
        parser.error("cache clear interval must be nonnegative")
    if not math.isfinite(args.curriculum_fraction) or not 0 <= args.curriculum_fraction <= 1:
        parser.error("curriculum fraction must be finite and in [0, 1]")
    if args.ddp_gradient_bucket_views and args.ranks < 2:
        parser.error("--ddp-gradient-bucket-views requires at least two ranks")
    if args.accumulation is not None and args.accumulation < 1:
        parser.error("accumulation must be positive")
    if (args.accumulation is not None or args.bucket_plan is not None) and len(args.stage or []) != 1:
        parser.error("candidate overrides require exactly one explicit stage")
    if args.bucket_plan is not None and not args.bucket_plan.is_file():
        parser.error("candidate bucket plan must exist")
    target = args.root / "runs" / args.tag
    target.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", ARTFLOW_ROOT=str(args.root),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TORCHINDUCTOR_CACHE_DIR=os.environ.get(
                   "TORCHINDUCTOR_CACHE_DIR", str(args.root / "torchinductor-cache")))
    env.pop("ARTFLOW_LOG_SHAPES", None)
    if args.record_metrics:
        env["ARTFLOW_INFRA_METRICS"] = "1"
        env["ARTFLOW_TRACE_START"] = str(args.trace_start)
        env["ARTFLOW_TRACE_STEPS"] = str(args.trace_steps)
    hardware = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
    (target / "hardware.txt").write_text(hardware.stdout + hardware.stderr)
    topology = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True)
    (target / "topology.txt").write_text(topology.stdout + topology.stderr)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (target / "environment.txt").write_text(freeze.stdout)
    stages = {"256p": (1, 0., .75), "640p": (5, .75, .95), "896p": (7, .95, 1.)}
    results = []
    for stage in args.stage or list(stages):
        accum, start, end = stages[stage]
        if args.accumulation is not None:
            accum = args.accumulation
        progress = start + (end - start) * args.curriculum_fraction
        override = target / f"{stage}.toml"
        override_text = f'''# Diagnostic fixed slice, not a production curriculum.
[data]
{('bucket_plan = ' + json.dumps(str(args.bucket_plan.resolve()))) if args.bucket_plan else ''}
curriculum_start = {progress}
curriculum_end = {progress}
[train]
max_steps = 400000
stop_at_step = {args.steps}
gradient_accumulation_steps = {accum}
checkpoint_interval = 1000000000
eval_interval = 1000000000
[eval]
dataset_path = ""
grid_steps = []
loss_interval = 0
kid_at_end = false
[paths]
output_dir = "{target}"
'''
        if args.cache_clear_interval is not None:
            override_text += f"[telemetry]\ncache_clear_interval = {args.cache_clear_interval}\n"
        override.write_text(override_text)
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={args.ranks}",
               "-m", "src.train.train", "--config", "configs/base.toml",
               "--config", str(args.stage_config_dir / f"hero-{stage}.toml"),
               "--config", "configs/hero.toml", "--config", str(override),
               "--run_name", stage]
        if args.cpu_wall_profile:
            cmd.append("--cpu_wall_profile")
        if args.gpu_health_snapshot:
            cmd.append("--gpu_health_snapshot")
        if args.breakdown:
            cmd.append("--step_breakdown")
        if args.foreach_updates:
            cmd.append("--foreach_updates")
        if args.local_cache_clear:
            cmd.append("--local_cache_clear")
        if args.dynamic_blocks:
            cmd.append("--compile_dynamic")
        if args.autotune_blocks:
            cmd.append("--compile_autotune")
        if args.native_flash_varlen:
            cmd.append("--native_flash_varlen")
        if args.real_rope:
            cmd.append("--real_rope")
        if args.hoist_double_rope:
            cmd.append("--hoist_double_rope")
        if args.muon_compile_square_ns:
            cmd.append("--muon_compile_square_ns")
        if args.disable_ddp_compile_split:
            cmd.append("--disable_ddp_compile_split")
        if args.ddp_gradient_bucket_views:
            cmd.append("--ddp_gradient_bucket_views")
        print(f"INFRA_START stage={stage} timeout={args.stage_timeout}s", flush=True)
        started = time.time()
        with (target / f"{stage}.log").open("w") as log:
            # timeout controls the entire torchrun process group, including
            # ranks/compile workers; do not leave orphan GPU workers behind.
            proc = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s",
                                   f"{args.stage_timeout}s", *cmd],
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {"stage": stage, "started_unix": started, "ended_unix": time.time(),
                  "returncode": proc.returncode, "command": cmd,
                  "caption_progress": progress, "world_size": args.ranks,
                  "curriculum_fraction": args.curriculum_fraction,
                  "stage_progress_interval": [start, end],
                  "cpu_wall_profile": args.cpu_wall_profile,
                  "accumulation": accum, "steps": args.steps}
        results.append(result)
        (target / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"INFRA_END {json.dumps(result)}", flush=True)
    return int(any(result["returncode"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
