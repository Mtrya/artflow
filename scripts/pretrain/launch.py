"""Launch one Ascend stage from a complete recipe, resuming complete checkpoints.

Run with the prepared environment and data already installed:
  python -m scripts.pretrain.launch --config configs/hero.toml --stage 256p --storage-root /external/artflow --nproc_per_node 16

The platform owns environment setup, resource selection and retry policy. This
launcher owns the single-writer lock, checkpoint preflight and worker lifetime.
"""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.pretrain.config import load_config, flatten
from src.pretrain.stage_control import validate_checkpoint, CHECKPOINT_RECORD
from src.pretrain.train import parse_args
from src.pretrain.tracking import resume_run_id


def resolve_resume(config, stage_name, world_size):
    """Ignore incomplete same-stage writes; require a complete predecessor."""
    stage = config.stage(stage_name)
    start = config.stage_start(stage_name)
    output = Path(config.paths.output_dir)
    # All stages share one run directory; checkpoints are ordered by global step.
    run = output / config.train.run_name
    for candidate in sorted(run.glob("checkpoint_step_[0-9]*"), reverse=True):
        marker = candidate / CHECKPOINT_RECORD
        if marker.is_file():
            record = json.loads(marker.read_text())
            if (
                record.get("max_steps") != config.max_steps
                or record.get("world_size") != world_size
            ):
                raise ValueError(
                    f"incompatible checkpoint horizon or world size: {candidate}"
                )
            if record.get("global_step", 0) > stage.end_step:
                raise ValueError(
                    f"checkpoint is beyond the selected stage: {candidate}"
                )
        try:
            step = validate_checkpoint(
                candidate,
                max_steps=config.max_steps,
                stop_at_step=stage.end_step,
                scheduler_count=2,
                use_ema=True,
                world_size=world_size,
                device_type="npu",
            )
        except (ValueError, OSError) as exc:
            print(f"Ignoring unusable checkpoint {candidate}: {exc}", flush=True)
            continue
        if step < start:
            raise ValueError(f"{candidate} is before the selected stage")
        return candidate
    if start:
        candidate = run / f"checkpoint_step_{start:06d}"
        step = validate_checkpoint(
            candidate,
            max_steps=config.max_steps,
            stop_at_step=stage.end_step,
            scheduler_count=2,
            use_ema=True,
            world_size=world_size,
            device_type="npu",
        )
        if step != start:
            raise ValueError(f"previous stage must finish at step {start}")
        return candidate
    return None


def watch(command, log_path):
    """Mirror only this attempt's log; kill all ranks on NPU OOM or termination."""
    with log_path.open("a", buffering=1) as output, log_path.open() as reader:
        reader.seek(0, 2)
        proc = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )

        def stop(signum, _frame):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            raise SystemExit(128 + signum)

        handlers = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            tail = ""
            complete_since = None
            while True:
                chunk = reader.read()
                if chunk:
                    print(chunk, end="", flush=True)
                    combined = tail + chunk
                    tail = combined[-128:]
                    if "[training-complete]" in combined and complete_since is None:
                        complete_since = time.monotonic()
                    if "NPU out of memory" in combined and proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                status = proc.poll()
                if status is not None:
                    print(reader.read(), end="", flush=True)
                    return status
                # Some NPU runtimes hang while tearing down worker services.
                # Success requires the all-rank marker after checkpoints and
                # evaluations finish, never just the final optimizer step.
                if (
                    complete_since is not None
                    and time.monotonic() - complete_since >= 180
                ):
                    os.killpg(proc.pid, signal.SIGKILL)
                    return 0
                time.sleep(1)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    parser = parse_args()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.nproc_per_node < 1:
        parser.error("--nproc_per_node must be positive")
    config = load_config(args.config, storage_root=args.storage_root)
    resolved = flatten(config, args.stage)
    run = Path(config.paths.output_dir) / resolved["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another launcher holds the writer lock for {run}")
        resume = (
            Path(args.resume)
            if args.resume
            else resolve_resume(config, args.stage, args.nproc_per_node)
        )
        if resume:
            validate_checkpoint(
                resume,
                max_steps=config.max_steps,
                stop_at_step=resolved["stop_at_step"],
                min_step=resolved["stage_start"],
                scheduler_count=2,
                use_ema=True,
                world_size=args.nproc_per_node,
                device_type="npu",
            )
            resume_run_id(resume)
            from dataclasses import asdict

            if json.loads((resume / "run_config.json").read_text()) != asdict(config):
                raise ValueError(
                    "checkpoint config differs from the complete run config"
                )
        if not Path(resolved["bucket_plan"]).is_file():
            raise ValueError(f"missing stage bucket plan: {resolved['bucket_plan']}")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            "--log-dir",
            str(run / "elastic"),
            "-m",
            "src.pretrain.train",
            "--config",
            str(Path(args.config).resolve()),
            "--stage",
            args.stage,
            "--storage-root",
            config.paths.storage_root,
        ]
        if resume:
            command += ["--resume", str(resume)]
        for name in (
            "verify_resume_state",
            "step_breakdown",
            "cpu_wall_profile",
            "log_shapes",
            "infra_metrics",
            "record_identity",
            "check_config",
        ):
            if getattr(args, name):
                command.append("--" + name)
        command += [
            "--trace_start",
            str(args.trace_start),
            "--trace_steps",
            str(args.trace_steps),
        ]
        print(
            json.dumps(
                {
                    "command": command,
                    "stage_start": resolved["stage_start"],
                    "stage_end": resolved["stop_at_step"],
                    "schedule_horizon": config.max_steps,
                }
            ),
            flush=True,
        )
        if args.dry_run:
            return
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Variable bucket shapes need expandable segments to avoid allocator
        # fragmentation. Set the qualified policy before child NPU runtimes
        # initialize; a caller's environment must not silently change it.
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["OMP_NUM_THREADS"] = "1"
        raise SystemExit(watch(command, run / "training.log"))


if __name__ == "__main__":
    main()
