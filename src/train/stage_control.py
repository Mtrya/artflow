"""Global-horizon stage boundaries and lightweight checkpoint preflight.

The completion record is written last, after every rank has saved its state.
File sizes detect missing/truncated artifacts; actual deserialization remains
the trainer's responsibility. No model loading or GPU is needed for preflight.
"""

import argparse
import json
import os
from pathlib import Path
import re


CHECKPOINT_RECORD = "training_state.json"


def stage_endpoint(max_steps: int, stop_at_step: int = 0, resumed_step: int = 0) -> int:
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if type(stop_at_step) is not int or not 0 <= stop_at_step <= max_steps:
        raise ValueError("stop_at_step must be an integer in [0, max_steps] (0 disables it)")
    endpoint = stop_at_step or max_steps
    if type(resumed_step) is not int or not 0 <= resumed_step <= endpoint:
        raise ValueError(f"resume step {resumed_step} is beyond stage endpoint {endpoint}")
    return endpoint


def write_checkpoint_record(path, *, step, max_steps, scheduler_count, use_ema, world_size):
    """Call on rank zero only, after all checkpoint writes have completed."""
    root = Path(path)
    files = {p.name: p.stat().st_size for p in root.iterdir()
             if p.is_file() and p.name != CHECKPOINT_RECORD and not p.name.endswith(".tmp")}
    record = {"version": 1, "complete": True, "global_step": step,
              "max_steps": max_steps, "scheduler_count": scheduler_count,
              "use_ema": use_ema, "world_size": world_size, "files": files}
    temporary = root / (CHECKPOINT_RECORD + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / CHECKPOINT_RECORD)


def validate_checkpoint(path, *, max_steps, stop_at_step=0, expected_step=None,
                        min_step=0, require_record=False, scheduler_count=None,
                        use_ema=False, world_size=None):
    root = Path(path)
    match = re.fullmatch(r"checkpoint_step_(\d+)", root.name)
    if match is None or not root.is_dir():
        raise ValueError(f"not a checkpoint_step_* directory: {root}")
    step = int(match[1])
    stage_endpoint(max_steps, stop_at_step, step)
    if step < min_step or (expected_step is not None and step != expected_step):
        raise ValueError(f"checkpoint step {step} does not match the required stage interval/endpoint")
    record_path = root / CHECKPOINT_RECORD
    if not record_path.is_file():
        if require_record:
            raise ValueError(f"missing {record_path}; staged resume requires a completed checkpoint with recorded global T")
        return step  # Legacy, non-staged resume: scheduler files are checked by the trainer.
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("version") != 1 or record.get("complete") is not True:
        raise ValueError("checkpoint completion record is invalid")
    if record.get("global_step") != step or record.get("max_steps") != max_steps:
        raise ValueError("checkpoint step/global T does not match this invocation")
    if scheduler_count is not None and record.get("scheduler_count") != scheduler_count:
        raise ValueError("checkpoint scheduler count does not match")
    if use_ema and not record.get("use_ema"):
        raise ValueError("checkpoint has no recorded EMA state")
    if world_size is not None and record.get("world_size") != world_size:
        raise ValueError("checkpoint world size does not match")
    files = record.get("files", {})
    if not files:
        raise ValueError("checkpoint file inventory is empty")
    for name, size in files.items():
        if Path(name).name != name or type(size) is not int or size <= 0:
            raise ValueError("invalid checkpoint file inventory")
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size != size:
            raise ValueError(f"checkpoint file is missing or truncated: {artifact}")
    count = record["scheduler_count"]
    required = ["scheduler.bin" if i == 0 else f"scheduler_{i}.bin" for i in range(count)]
    if record["use_ema"]:
        required.append("ema_weights.pt")
    if any(name not in files for name in required):
        raise ValueError("checkpoint is missing required scheduler/EMA artifacts")
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--stop-at-step", type=int, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    validate_checkpoint(args.checkpoint, max_steps=args.max_steps,
                        stop_at_step=args.stop_at_step, expected_step=args.expected_step,
                        min_step=args.min_step, world_size=args.world_size,
                        require_record=True, scheduler_count=2, use_ema=True)


if __name__ == "__main__":
    main()
