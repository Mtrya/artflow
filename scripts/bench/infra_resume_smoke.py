"""Bounded real-training resume smoke; never launches the hero run.

Supply the same resolved configs and execution flags as the source checkpoint.
For a resolution transition, supply the next resolution config and explicitly
request a sampler reset. Lower-rank smoke results do not qualify eight ranks.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from src.train.config import load_config
from src.train.stage_control import CHECKPOINT_RECORD, validate_checkpoint


def check_updates(directory, *, start, end, ranks):
    for rank in range(ranks):
        rows = [json.loads(line) for line in
                (directory / "infra" / f"rank-{rank}.jsonl").read_text().splitlines()]
        if [row["step"] for row in rows] != list(range(start + 1, end + 1)):
            raise ValueError(f"rank {rank}: incorrect resumed update interval")
        if any(row["rank"] != rank or not math.isfinite(row["loss"]) for row in rows):
            raise ValueError(f"rank {rank}: invalid/nonfinite update record")


def check_saved_state(directory, step):
    import torch

    torch.set_num_threads(4)
    for index in range(2):
        suffix = "" if index == 0 else f"_{index}"
        state = torch.load(directory / f"scheduler{suffix}.bin", map_location="cpu",
                           weights_only=False)
        if state["last_epoch"] != step:
            raise ValueError("saved scheduler did not continue to endpoint")
        state = torch.load(directory / f"optimizer{suffix}.bin", map_location="cpu",
                           weights_only=False)
        if not state["state"]:
            raise ValueError("saved optimizer state is empty")
        for parameter_state in state["state"].values():
            for name, value in parameter_state.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError("saved optimizer state is nonfinite")
                if name == "step" and int(value) != step:
                    raise ValueError("saved optimizer step did not continue")
        del state
    ema = torch.load(directory / "ema_weights.pt", map_location="cpu", weights_only=True)
    if not ema or any(not torch.isfinite(value).all() for value in ema.values()):
        raise ValueError("saved EMA is empty/nonfinite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--checkpoint-interval", type=int, default=1000000000,
                        help="Use a small interval to retain an intermediate replay reference")
    parser.add_argument("--reset-sampler", action="store_true")
    parser.add_argument("--train-arg", action="append", default=[],
                        help="Explicit execution flag, e.g. --train-arg=--compile_dynamic")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10 or args.timeout <= 0 or args.checkpoint_interval <= 0:
        parser.error("smoke requires 1–10 updates and positive timeout/checkpoint interval")
    config = load_config(args.config)
    start = validate_checkpoint(
        args.checkpoint, max_steps=config.train.max_steps, require_record=True,
        scheduler_count=2, use_ema=True, world_size=args.ranks,
    )
    end = start + args.steps
    if end > config.train.max_steps:
        parser.error("smoke endpoint exceeds original global T")
    source_record = (args.checkpoint / CHECKPOINT_RECORD).read_bytes()
    args.out.mkdir(parents=True, exist_ok=False)
    override = args.out / "smoke.toml"
    override.write_text(f'''# Diagnostic stop only: preserve the supplied global horizon and curriculum.
[train]
stop_at_step = {end}
checkpoint_interval = {args.checkpoint_interval}
eval_interval = 1000000000
[eval]
dataset_path = ""
grid_steps = []
loss_interval = 0
kid_at_end = false
[paths]
output_dir = {json.dumps(str(args.out.resolve()))}
''')
    command = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={args.ranks}", "-m", "src.train.train"]
    for path in [*args.config, str(override)]:
        command += ["--config", path]
    command += ["--run_name", "resume", "--resume", str(args.checkpoint),
                "--resume_full", *args.train_arg]
    if args.reset_sampler:
        command += ["--reset_sampler"]
    result = dict(command=command, world_size=args.ranks, start_step=start, end_step=end,
                  max_steps=config.train.max_steps, reset_sampler=args.reset_sampler,
                  caption_start=config.data.curriculum_start,
                  caption_end=config.data.curriculum_end, started_unix=time.time(),
                  caveat="Smoke, not an uninterrupted-reference equivalence or hero readiness verdict")
    (args.out / "request.json").write_text(json.dumps(result, indent=2) + "\n")
    env = dict(os.environ, ARTFLOW_INFRA_METRICS="1", ARTFLOW_TRACE_START="-1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (args.out / "train.log").open("w") as log:
        completed = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", f"{args.timeout}s", *command],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    result.update(returncode=completed.returncode, ended_unix=time.time(), verified=False)
    try:
        if completed.returncode:
            raise ValueError(f"training failed: exit {completed.returncode}")
        run = args.out / "resume"
        checkpoint = run / f"checkpoint_step_{end:06d}"
        validate_checkpoint(checkpoint, max_steps=config.train.max_steps,
                            expected_step=end, require_record=True,
                            scheduler_count=2, use_ema=True, world_size=args.ranks)
        check_updates(run, start=start, end=end, ranks=args.ranks)
        check_saved_state(checkpoint, end)
        if (args.checkpoint / CHECKPOINT_RECORD).read_bytes() != source_record:
            raise ValueError("source completion record changed")
        validate_checkpoint(args.checkpoint, max_steps=config.train.max_steps,
                            expected_step=start, require_record=True,
                            scheduler_count=2, use_ema=True, world_size=args.ranks)
        result["verified"] = True
    except Exception as exc:
        result["error"] = str(exc)
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
