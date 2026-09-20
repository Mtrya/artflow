"""Bounded eight-rank validation of an explicitly selected infra candidate.

No optimization search, resource submission, retries, production-T selection or
automatic readiness verdict. Run only after lower-rank candidate selection.
Each phase is serialized; use a separate, explicitly budgeted GPU allocation.
The midpoint trials reach the real health cadence. Equal-width early/mid/late
curriculum slices must receive equal weight when costing, despite unequal trial
lengths. Profiler windows and compilation need separate accounting.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STAGES = (("256p", 1), ("640p", 5), ("896p", 7))
TRAIN_FLAGS = ("--compile_dynamic", "--disable_ddp_compile_split",
               "--hoist_double_rope", "--native_flash_varlen", "--real_rope",
               "--muon_compile_square_ns", "--gpu_health_snapshot", "--local_cache_clear")
BENCH_FLAGS = ("--dynamic-blocks", "--disable-ddp-compile-split",
               "--hoist-double-rope", "--native-flash-varlen", "--real-rope",
               "--muon-compile-square-ns", "--gpu-health-snapshot", "--local-cache-clear")


def plan(root, tag, autotune, *, completion_only=False, stage_config_dir=None,
         accumulations=None):
    if accumulations is None:
        accumulations = dict(STAGES)
    if (set(accumulations) != {s for s, _ in STAGES}
            or any(type(v) is not int or v < 1 for v in accumulations.values())):
        raise ValueError("supply positive integer accumulation for all three stages")
    out = root / "runs" / tag
    phases = []

    def add(name, command, limit):
        phases.append(dict(name=name, command=command, timeout_seconds=limit))

    config_dir = Path(stage_config_dir) if stage_config_dir is not None else out / "configs"
    if stage_config_dir is None:
        for stage, _ in STAGES:
            add(f"config-{stage}", [sys.executable, "-m", "scripts.bench.render_hero_stage",
                stage, "--root", str(root), "--out", str(config_dir / f"hero-{stage}.toml")], 30)

    # Measure all resolutions before spending time on additional slices.
    for label, fraction, steps in (("mid", .5, 256), ("early", 1/6, 128), ("late", 5/6, 128)):
        for stage, _ in STAGES:
            command = [sys.executable, "-m", "scripts.bench.infra_baseline",
                       "--root", str(root), "--tag", f"{tag}-{stage}-{label}",
                       "--stage-config-dir", str(config_dir), "--stage", stage,
                       "--accumulation", str(accumulations[stage]),
                       "--ranks", "8", "--steps", str(steps), "--stage-timeout", "3600",
                       "--curriculum-fraction", str(fraction), "--record-metrics",
                       "--no-cpu-wall-profile", "--no-cache-clear", *BENCH_FLAGS]
            if stage in autotune:
                command.append("--autotune-blocks")
            if label == "mid":
                command += ["--trace-start", "128", "--trace-steps", "3"]
            add(f"rates-{stage}-{label}", command, 3650)

    for stage, accumulation in STAGES:
        accumulation = accumulations[stage]
        memory = out / f"memory-{stage}"
        add(f"prepare-{stage}", [sys.executable, "-m", "scripts.bench.infra_memory_stress",
            "prepare", "--config", "configs/base.toml", "--config",
            str(config_dir / f"hero-{stage}.toml"), "--config", "configs/hero.toml",
            "--out", str(memory), "--ranks", "8", "--cycles", "2",
            "--accumulation", str(accumulation), "--health-interval", "1",
            "--cache-clear-interval", "0"], 300)
        flags = [*TRAIN_FLAGS, *(["--compile_autotune"] if stage in autotune else [])]
        add(f"memory-{stage}", [sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=8", "-m", "scripts.bench.infra_memory_stress", "train",
            "--manifest", str(memory / "manifest.json"),
            *[f"--train-arg={flag}" for flag in flags]], 4000)
        add(f"verify-memory-{stage}", [sys.executable, "-m", "scripts.bench.infra_memory_stress",
            "verify", "--manifest", str(memory / "manifest.json")], 60)

    add("cuda-primitives", [sys.executable, "-m", "torch.distributed.run",
                           "--nproc_per_node=8", "-m", "scripts.bench.infra_cuda_safety"], 180)
    add("stage-chain-and-evaluation", [sys.executable, "-m", "scripts.bench.infra_stage_chain",
        "--root", str(root), "--out", str(out / "stage-chain"), "--ranks", "8",
        "--stage-config-dir", str(config_dir),
        "--timeout", "4500", "--cache-clear-interval", "0", "--with-evaluation",
        *[f"--{stage}-accumulation={accumulations[stage]}" for stage, _ in STAGES],
        *[f"--train-arg={flag}" for flag in TRAIN_FLAGS],
        *[f"--{stage}-train-arg=--compile_autotune" for stage, _ in STAGES if stage in autotune]],
        9000)
    if completion_only:
        names = {"config-256p", "config-640p", "config-896p", "prepare-896p",
                 "memory-896p", "verify-memory-896p", "cuda-primitives",
                 "stage-chain-and-evaluation"}
        phases = [phase for phase in phases if phase["name"] in names]
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-seconds", type=int, required=True,
                        help="Whole workload deadline, within the approved allocation budget")
    parser.add_argument("--autotune-stage", action="append", default=[],
                        choices=[stage for stage, _ in STAGES])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--completion-only", action="store_true",
                        help="Fresh-output 896p stress, guards and state/eval chain only; "
                             "requires separate evidence for omitted phases")
    parser.add_argument("--stage-config-dir", type=Path,
                        help="Use already resolved, pinned hardware-specific stage configs")
    for stage, default in STAGES:
        parser.add_argument(f"--{stage}-accumulation", type=int, default=default)
    args = parser.parse_args()
    if (not args.tag or Path(args.tag).name != args.tag or args.tag in (".", "..")
            or args.max_seconds <= 0):
        parser.error("need a single-component tag and a positive deadline")
    phases = plan(args.root.resolve(), args.tag, set(args.autotune_stage),
                  completion_only=args.completion_only,
                  stage_config_dir=args.stage_config_dir,
                  accumulations={s: getattr(args, f"{s}_accumulation") for s, _ in STAGES})
    report = dict(world_size=8, autotune_stages=args.autotune_stage, phases=phases,
                  completion_only=args.completion_only,
                  complete=False, readiness_established=False, production_steps_selected=False,
                  caveat="Phase completion alone does not establish numerical, memory, "
                         "exposure, saturation or cost acceptance. Inspect artifacts and pins.")
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    import torch
    if torch.cuda.device_count() != 8:
        raise ValueError("this final validation requires exactly eight visible physical GPUs")
    out = args.root.resolve() / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ARTFLOW_ROOT=str(args.root.resolve()), OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.pop("ARTFLOW_LOG_SHAPES", None)
    deadline = time.monotonic() + args.max_seconds

    def save():
        (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for phase in phases:
        remaining = int(deadline - time.monotonic())
        if remaining <= 30:
            report["error"] = "whole-workload deadline reached; no further phase started"
            save()
            return 1
        limit = min(phase["timeout_seconds"], remaining - 30)
        phase.update(started_unix=time.time(), effective_timeout_seconds=limit)
        save()
        with (out / f"{phase['name']}.log").open("w") as log:
            result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s",
                                     f"{limit}s", *phase["command"]], env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        phase.update(ended_unix=time.time(), returncode=result.returncode)
        save()
        if result.returncode:
            return 1
    report["complete"] = True
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
