#!/usr/bin/env bash
# Stage-1 continuation experiment; architecture changes need a fresh experiment.
# Usage: ARTFLOW_ROOT=... bash scripts/ascend/stability_probe.sh reference|calibrated
set -euo pipefail
: "${ARTFLOW_ROOT:?Set the Ascend work root}"
ARM=${1:?Choose reference or calibrated}
case "$ARM" in reference|calibrated) ;; *) exit 2 ;; esac
export ARM
export PYTHONPATH="$ARTFLOW_ROOT/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

if [ -n "${SWANLAB_NETRC_B64:-}" ]; then
  mkdir -p "$HOME/.swanlab"
  (umask 077; echo "$SWANLAB_NETRC_B64" | base64 -d > "$HOME/.swanlab/.netrc")
  unset SWANLAB_NETRC_B64
fi
if [ ! -f "$HOME/.swanlab/.netrc" ]; then
  export SWANLAB_MODE=offline
fi

python3 -u - <<'PY'
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.pretrain.stage_control import validate_checkpoint

root = Path(os.environ["ARTFLOW_ROOT"])
arm = os.environ["ARM"]
run = "ascend-stability-0924-" + arm
out = root / "runs" / run
cfg = out / "inputs"
cfg.mkdir(parents=True, exist_ok=True)
source = root / "logs/bnprobe-a2/config-bnprobe-branch-a2b"
checkpoint = root / "runs/bnprobe-branch-a2b/checkpoint_step_002000"
validate_checkpoint(checkpoint, max_steps=200000, stop_at_step=3000,
                    expected_step=2000, require_record=True,
                    scheduler_count=2, use_ema=True, world_size=16)
if (out / "training_metrics.jsonl").exists():
    raise RuntimeError(f"Refusing to overwrite an existing experiment: {out}")
for name in ("inputs.toml", "plan.json"):
    (cfg / name).write_bytes((source / name).read_bytes())
(cfg / "paths.toml").write_text('[data]\nbucket_plan = ' + json.dumps(str(cfg / "plan.json")) + '\n')
if arm == "calibrated":
    panel = root / "runs/ascend-stability-0924-reference/stability_inputs.pt"
    (out / "stability_inputs.pt").write_bytes(panel.read_bytes())
command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=16",
           "--master_port=29523", "-m", "src.pretrain.train",
           "--config", "configs/base.toml", "--config", str(cfg / "inputs.toml"),
           "--config", "configs/hero.toml", "--config", str(cfg / "paths.toml"),
           "--config", "configs/experiments/stability_0924/reference.toml"]
if arm == "calibrated":
    command += ["--config", "configs/experiments/stability_0924/calibrated.toml"]
command += ["--dataloader_numpy_batch", "--no-compile", "--foreach_updates",
            "--resume", str(checkpoint), "--resume_full", "--run_name", run]
(cfg / "command.json").write_text(json.dumps(command, indent=2) + "\n")
log = out / "train.log"
print(f"EXPERIMENT_START {run} source_step=2000 endpoint=3000", flush=True)
with log.open("w") as output:
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True)
    cursor, completed_at = 0, None
    deadline = time.monotonic() + 7200
    while process.poll() is None:
        time.sleep(5)
        with log.open(errors="replace") as stream:
            stream.seek(cursor)
            chunk = stream.read()
            cursor = stream.tell()
        for line in chunk.splitlines():
            if any(tag in line for tag in ("[stability@", "[eval-loss@", "[throughput-summary]",
                                            "[stop]", "Training finished", "swanlab: View")):
                print(line[-5000:], flush=True)
        if "Training finished" in chunk and completed_at is None:
            validate_checkpoint(out / "checkpoint_step_003000", max_steps=200000,
                                expected_step=3000, require_record=True,
                                scheduler_count=2, use_ema=True, world_size=16)
            completed_at = time.monotonic()
        # NPU TBE workers can hang after successful teardown. Signal only
        # this owned process group, after verifying its completed checkpoint.
        if (completed_at and time.monotonic() - completed_at > 30) or time.monotonic() > deadline:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
    if process.returncode and completed_at is None:
        print(log.read_text(errors="replace")[-16000:], flush=True)
        raise SystemExit(process.returncode)
validate_checkpoint(out / "checkpoint_step_003000", max_steps=200000,
                    expected_step=3000, require_record=True,
                    scheduler_count=2, use_ema=True, world_size=16)
records = [json.loads(line) for line in (out / "stability.jsonl").read_text().splitlines()]
if not records or records[-1]["step"] != 3000:
    raise RuntimeError("Missing endpoint stability telemetry")
print(f"EXPERIMENT_COMPLETE {run} observations={len(records)} panel={records[-1]['panel_id']}", flush=True)
PY
