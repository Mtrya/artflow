"""Bounded hardware migration pilot; not final hero acceptance or tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.bench.infra_acceptance import plan


def pilot_plan(root, tag):
    phases = plan(root, tag, {"256p", "640p", "896p"})
    configs = [p for p in phases if p["name"].startswith("config-")]
    guards = [p for p in phases if p["name"] == "cuda-primitives"]
    rates = [p for p in phases if p["name"].startswith("rates-")
             and p["name"].endswith("-mid")]
    for phase in rates:
        command = phase["command"]
        for key, value in (("--steps", "128"), ("--stage-timeout", "1800"),
                           ("--trace-start", "64")):
            command[command.index(key) + 1] = value
        phase["timeout_seconds"] = 1850
    return configs + guards + rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-seconds", type=int, default=6200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ("", ".", "..") or args.max_seconds <= 0:
        parser.error("single-component tag and positive deadline required")
    root = args.root.resolve()
    report = dict(phases=pilot_plan(root, args.tag), complete=False,
                  final_acceptance=False, bucket_plan_tuned=False,
                  caveat="Reference 4090 bucket/accumulation plan on Hopper; no final cost claim.")
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 8 or not all("H100" in n or "H200" in n for n in names):
        raise ValueError(f"requires eight H100/H200 GPUs; observed {names}")
    report["devices"] = names
    out = root / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    env = dict(os.environ, ARTFLOW_ROOT=str(root), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")

    def save():
        (out / "pilot.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for phase in report["phases"]:
        limit = min(phase["timeout_seconds"], int(deadline - time.monotonic()) - 30)
        if limit <= 0:
            report["error"] = "whole-pilot deadline reached"
            save()
            return 1
        phase.update(started_unix=time.time(), effective_timeout_seconds=limit)
        save()
        with (out / (phase["name"] + ".log")).open("w") as log:
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
