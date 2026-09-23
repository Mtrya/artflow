"""Exercise the H200 launcher's recipe, checkpoint isolation and writer lock."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from src.pretrain.stage_control import write_checkpoint_record


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/pretrain/hero_stage_h200.sh"


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "artflow"
    source = root / "source"
    package = source / "src/pretrain"
    package.mkdir(parents=True)
    (source / "src/__init__.py").touch()
    (package / "__init__.py").touch()
    shutil.copy(REPO / "src/pretrain/stage_control.py", package / "stage_control.py")
    (source / "configs").mkdir()
    (source / "configs/base.toml").write_text("[train]\nmax_steps = 50000\n")
    (source / "configs/hero.toml").write_text("[optim]\nmuon_lr = 0.02\nlearning_rate = 0.0003\n")
    inputs = root / "inputs"
    inputs.mkdir()
    for stage, accumulation in [("256p", 1), ("640p", 10), ("896p", 14)]:
        (inputs / f"hero-{stage}.toml").write_text(
            f'[train]\ngradient_accumulation_steps = {accumulation}\n'
        )
    shutil.copy(REPO / "configs/hero/h200.toml", inputs / "h200.toml")
    (inputs / "inputs.sha256").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n"
        for p in sorted(inputs.glob("*.toml"))
    ))
    (root / "jobs").mkdir()
    (root / "jobs/swanlab_login.sh").write_text("# Test: no credentials.\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python3"
    python.write_text(f'''#!{sys.executable}
import json, os, sys, time, tomllib
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['-m', 'torch.distributed.run']:
    config = {{}}
    for i, value in enumerate(args):
        if value == '--config':
            data = tomllib.loads(Path(args[i+1]).read_text())
            for section, entries in data.items():
                config.setdefault(section, {{}}).update(entries)
    Path(os.environ['ARTFLOW_ROOT'], 'invocation.json').write_text(json.dumps({{'args': args, 'config': config}}))
    if os.environ.get('TEST_HOLD'):
        time.sleep(20)
else:
    os.execv(sys.executable, [sys.executable, *args])
''')
    python.chmod(0o755)
    env = dict(os.environ, ARTFLOW_ROOT=str(root), ARTFLOW_SOURCE=str(source),
               ARTFLOW_H200_INPUTS=str(inputs), ARTFLOW_PRETRAIN_PACKAGE="src.pretrain",
               ARTFLOW_H200_COMPILER_CACHE=str(root / "inductor"),
               ARTFLOW_H200_TRITON_CACHE=str(root / "triton"),
               PATH=f"{binaries}:{os.environ['PATH']}")
    return root, env


def checkpoint(root, stage, step, *, horizon=600000):
    path = root / "runs" / stage / f"checkpoint_step_{step:06d}"
    path.mkdir(parents=True)
    names = ["model.safetensors", "ema_weights.pt", "optimizer.bin", "optimizer_1.bin",
             "scheduler.bin", "scheduler_1.bin"]
    names += [f"random_states_{rank}.pkl" for rank in range(4)]
    names += [f"sampler_state_rank_{rank:05d}.pt" for rank in range(4)]
    for name in names:
        (path / name).write_bytes(b"inventory-only fixture")
    write_checkpoint_record(path, step=step, max_steps=horizon,
                            scheduler_count=2, use_ema=True, world_size=4)
    return path


def run(workspace, stage="256p"):
    root, env = workspace
    result = subprocess.run(["bash", str(LAUNCHER), stage], env=env,
                            capture_output=True, text=True, timeout=10)
    return result, root / "invocation.json"


def test_fresh_h200_does_not_resume_another_hero(workspace):
    root, _ = workspace
    checkpoint(root, "hero-256p", 52000, horizon=480000)
    checkpoint(root, "ascend-hero-256p-v2", 26000)
    result, invocation = run(workspace)
    assert result.returncode == 0, result.stderr
    record = json.loads(invocation.read_text())
    assert "--resume" not in record["args"]
    assert record["args"][record["args"].index("--run_name") + 1] == "hero-h200-256p"
    assert record["config"]["train"] == {
        "max_steps": 600000, "gradient_accumulation_steps": 1, "stop_at_step": 450000,
        "ema_decay_warmup": True,
    }
    assert record["config"]["optim"]["lr_warmup_steps"] == 20000


def test_same_stage_uses_last_complete_h200_checkpoint(workspace):
    root, _ = workspace
    saved = checkpoint(root, "hero-h200-256p", 2000)
    (saved.parent / "checkpoint_step_004000").mkdir()
    result, invocation = run(workspace)
    assert result.returncode == 0, result.stderr
    args = json.loads(invocation.read_text())["args"]
    assert args[args.index("--resume") + 1] == str(saved)
    assert "--resume_full" in args and "--reset_sampler" not in args
    assert "Skipping unusable checkpoint checkpoint_step_004000" in result.stderr


def test_transition_requires_h200_endpoint_and_resets_sampler(workspace):
    root, _ = workspace
    result, invocation = run(workspace, "640p")
    assert result.returncode != 0 and not invocation.exists()
    saved = checkpoint(root, "hero-h200-256p", 450000)
    result, invocation = run(workspace, "640p")
    assert result.returncode == 0, result.stderr
    record = json.loads(invocation.read_text())
    assert record["args"][record["args"].index("--resume") + 1] == str(saved)
    assert "--reset_sampler" in record["args"]
    assert record["config"]["train"]["stop_at_step"] == 570000


def test_changed_qualified_inputs_fail_before_training(workspace):
    root, _ = workspace
    (root / "inputs/h200.toml").write_text("[train]\nmax_steps = 1\n")
    result, invocation = run(workspace)
    assert result.returncode != 0 and not invocation.exists()


def test_h200_writer_lock_prevents_concurrent_training(workspace):
    root, env = workspace
    first = subprocess.Popen(["bash", str(LAUNCHER), "256p"],
                             env=dict(env, TEST_HOLD="1"), start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (root / "invocation.json").exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert (root / "invocation.json").exists()
        result, _ = run(workspace)
        assert result.returncode != 0
        assert "another H200 launcher holds the writer lock" in result.stderr
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGTERM)
        first.communicate(timeout=5)
    assert subprocess.run(["flock", "-n", str(root / "runs/hero-h200-256p/.writer.lock"), "true"]).returncode == 0
