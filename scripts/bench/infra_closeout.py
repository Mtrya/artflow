"""Bounded matched runs for the three remaining infra opportunities.

Run one resolution per allocation. The 256p sequence isolates cache clearing,
health snapshots, and autotuning; high resolutions test autotuning with an A/B/A
order check. All runs use real input and the same fixed curriculum slice. This
is candidate selection, not final eight-GPU acceptance or a choice of hero T.
Do not run concurrently with synthetic preparation, checkpoint replay or hashing.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def variants(stage):
    if stage not in ("256p", "640p", "896p"):
        raise ValueError("unsupported resolution stage")
    rows = []
    if stage == "256p":
        rows += [("cache_cpu", ["--cache-clear-interval", "100"]),
                 ("nocache_cpu", ["--no-cache-clear"])]
    selected = ["--no-cache-clear", "--gpu-health-snapshot"]
    rows += [("nocache_gpu", selected),
             ("autotune_gpu", [*selected, "--autotune-blocks"]),
             ("nocache_gpu_repeat", selected)]
    return rows


def commands(root, tag, stage, ranks):
    steps = 256 if stage == "256p" else 128
    timeout = {"256p": 1200, "640p": 2200, "896p": 2600}[stage]
    common = [sys.executable, "-m", "scripts.bench.infra_baseline",
              "--root", str(root), "--stage-config-dir", str(root / "configs"),
              "--stage", stage, "--ranks", str(ranks), "--steps", str(steps),
              "--stage-timeout", str(timeout), "--record-metrics",
              "--no-cpu-wall-profile", "--dynamic-blocks",
              "--disable-ddp-compile-split", "--hoist-double-rope",
              "--native-flash-varlen", "--real-rope", "--muon-compile-square-ns",
              "--local-cache-clear", "--curriculum-fraction", "0.5"]
    return [dict(variant=name, tag=f"{tag}-{stage}-{name}", steps=steps,
                 command=[*common, "--tag", f"{tag}-{stage}-{name}", *flags])
            for name, flags in variants(stage)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", choices=["256p", "640p", "896p"], required=True)
    parser.add_argument("--ranks", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.tag or Path(args.tag).name != args.tag or args.tag in (".", ".."):
        parser.error("tag must be a nonempty single path component")
    plan = commands(args.root, args.tag, args.stage, args.ranks)
    report = dict(stage=args.stage, physical_ranks=args.ranks, cases=plan,
                  complete=False, eight_rank_acceptance_established=False,
                  hero_steps_selected=False,
                  caveat="Fixed midpoint distribution, short near-initial training; "
                         "not representative all-in hero costing or quality evidence.")
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    out = args.root / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=False)

    def save():
        (out / "closeout.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for case in plan:
        case["started_unix"] = time.time()
        save()
        # infra_baseline bounds the whole torchrun process group. The workload
        # launcher additionally bounds this orchestrator and all remaining cases.
        result = subprocess.run(case["command"])
        case.update(returncode=result.returncode, ended_unix=time.time())
        save()
        if result.returncode:
            print(json.dumps(dict(failed_variant=case["variant"],
                                  returncode=result.returncode)), flush=True)
            return 1
    report["complete"] = True
    save()
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
