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


def _require_recovery_inventory(record):
    """This trainer saves one optimizer per scheduler and state for every rank."""
    count, world = record.get("scheduler_count"), record.get("world_size")
    if type(count) is not int or count < 1 or type(world) is not int or world < 1:
        raise ValueError(
            "checkpoint optimizer/scheduler count and world size must be positive integers"
        )
    if type(record.get("use_ema")) is not bool:
        raise ValueError("checkpoint EMA setting must be boolean")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("checkpoint file inventory is empty or invalid")
    if not any(name in files for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("checkpoint is missing required model weights")
    required = [
        f"{kind}.bin" if i == 0 else f"{kind}_{i}.bin"
        for kind in ("optimizer", "scheduler")
        for i in range(count)
    ]
    required += [f"random_states_{rank}.pkl" for rank in range(world)]
    required += [f"sampler_state_rank_{rank:05d}.pt" for rank in range(world)]
    if record["use_ema"]:
        required.append("ema_weights.pt")
    if record.get("device_type") == "npu":
        required += [f"npu_rng_state_rank_{rank:05d}.pt" for rank in range(world)]
        required += ["run_config.json", "transformer_config.json", "bucket_plan.json"]
    missing = [name for name in required if name not in files]
    if missing:
        raise ValueError(
            f"checkpoint is missing required recovery artifacts: {', '.join(missing)}"
        )


def stage_endpoint(max_steps: int, stop_at_step: int = 0, resumed_step: int = 0) -> int:
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if type(stop_at_step) is not int or not 0 <= stop_at_step <= max_steps:
        raise ValueError(
            "stop_at_step must be an integer in [0, max_steps] (0 disables it)"
        )
    endpoint = stop_at_step or max_steps
    if type(resumed_step) is not int or not 0 <= resumed_step <= endpoint:
        raise ValueError(
            f"resume step {resumed_step} is beyond stage endpoint {endpoint}"
        )
    return endpoint


def write_checkpoint_record(
    path, *, step, max_steps, scheduler_count, use_ema, world_size, device_type=None
):
    """Call on rank zero only, after all checkpoint writes have completed."""
    root = Path(path)
    files = {
        p.name: p.stat().st_size
        for p in root.iterdir()
        if p.is_file() and p.name != CHECKPOINT_RECORD and not p.name.endswith(".tmp")
    }
    record = {
        "version": 1,
        "complete": True,
        "global_step": step,
        "max_steps": max_steps,
        "scheduler_count": scheduler_count,
        "use_ema": use_ema,
        "world_size": world_size,
        "device_type": device_type,
        "files": files,
    }
    _require_recovery_inventory(record)
    if any(size <= 0 for size in files.values()):
        raise ValueError("cannot publish a checkpoint with empty artifacts")
    temporary = root / (CHECKPOINT_RECORD + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / CHECKPOINT_RECORD)


def validate_checkpoint(
    path,
    *,
    max_steps,
    stop_at_step=0,
    expected_step=None,
    min_step=0,
    scheduler_count=None,
    use_ema=False,
    world_size=None,
    device_type=None,
):
    root = Path(path)
    match = re.fullmatch(r"checkpoint_step_(\d+)", root.name)
    if match is None or not root.is_dir():
        raise ValueError(f"not a checkpoint_step_* directory: {root}")
    step = int(match[1])
    stage_endpoint(max_steps, stop_at_step, step)
    if step < min_step or (expected_step is not None and step != expected_step):
        raise ValueError(
            f"checkpoint step {step} does not match the required stage interval/endpoint"
        )
    record_path = root / CHECKPOINT_RECORD
    if not record_path.is_file():
        raise ValueError(f"missing {record_path}; resume requires a completed checkpoint")
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
    if device_type is not None and record.get("device_type") != device_type:
        raise ValueError("checkpoint device type does not match")
    _require_recovery_inventory(record)
    files = record["files"]
    for name, size in files.items():
        if Path(name).name != name or type(size) is not int or size <= 0:
            raise ValueError("invalid checkpoint file inventory")
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size != size:
            raise ValueError(f"checkpoint file is missing or truncated: {artifact}")
    return step


def verify_restored_rng(path, *, process_index, device):
    """Reject Accelerate's silent RNG-load failure on a strict staged resume.

    Call after loading Accelerate and the rank's NPU sidecar. Inspect only its
    generator, never initialize peer-device contexts to verify their states.
    Checkpoints are trusted project artifacts, as with optimizer deserialization.
    """
    import random
    import numpy as np
    import torch

    try:
        state = torch.load(
            Path(path) / f"random_states_{process_index}.pkl",
            map_location="cpu",
            weights_only=False,
        )
        if random.getstate() != state["random_state"]:
            raise ValueError("Python RNG was not restored")
        actual_numpy, expected_numpy = np.random.get_state(), state["numpy_random_seed"]
        if len(actual_numpy) != len(expected_numpy) or not all(
            np.array_equal(a, b) for a, b in zip(actual_numpy, expected_numpy)
        ):
            raise ValueError("NumPy RNG was not restored")
        if not torch.equal(torch.get_rng_state(), state["torch_manual_seed"]):
            raise ValueError("CPU Torch RNG was not restored")
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "npu":
            expected = torch.load(
                Path(path) / f"npu_rng_state_rank_{process_index:05d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if not torch.equal(torch.npu.get_rng_state(device).cpu(), expected):
                raise ValueError("rank-local NPU RNG was not restored")
        elif device.type == "cuda":
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            if not torch.equal(
                torch.cuda.get_rng_state(device), state["torch_cuda_manual_seed"][index]
            ):
                raise ValueError("rank-local CUDA RNG was not restored")
    except Exception as exc:
        raise ValueError(
            f"checkpoint RNG continuation failed for rank {process_index}: {exc}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--stop-at-step", type=int, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    validate_checkpoint(
        args.checkpoint,
        max_steps=args.max_steps,
        stop_at_step=args.stop_at_step,
        expected_step=args.expected_step,
        min_step=args.min_step,
        world_size=args.world_size,
        scheduler_count=2,
        use_ema=True,
    )


if __name__ == "__main__":
    main()
