"""The production launcher reads one recipe and prevents concurrent writers."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from scripts.pretrain.launch import resolve_resume
from src.pretrain.config import load_config
from src.pretrain.stage_control import write_checkpoint_record

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipe(tmp_path):
    text = (
        (REPO / "configs/hero.toml")
        .read_text()
        .replace('storage_root = ".."', f'storage_root = "{tmp_path}"')
    )
    path = tmp_path / "run.toml"
    path.write_text(text)
    config = load_config(path)
    for stage in config.stages:
        plan = Path(stage.bucket_plan)
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text("{}")
    return path, config


def checkpoint(config, stage, step):
    root = (
        Path(config.paths.output_dir)
        / f"{config.train.run_name}-{stage}"
        / f"checkpoint_step_{step:06d}"
    )
    root.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "optimizer.bin",
        "optimizer_1.bin",
        "scheduler.bin",
        "scheduler_1.bin",
        "random_states_0.pkl",
        "sampler_state_rank_00000.pt",
        "ema_weights.pt",
        "npu_rng_state_rank_00000.pt",
        "transformer_config.json",
        "bucket_plan.json",
    ):
        (root / name).write_bytes(b"test")
    (root / "run_config.json").write_text(json.dumps(asdict(config)))
    write_checkpoint_record(
        root,
        step=step,
        max_steps=config.max_steps,
        scheduler_count=2,
        use_ema=True,
        world_size=1,
        device_type="npu",
    )
    return root


def test_resume_prefers_complete_own_checkpoint_then_predecessor(recipe):
    _, config = recipe
    assert resolve_resume(config, "256p", 1) is None
    with pytest.raises(ValueError):
        resolve_resume(config, "640p", 1)
    endpoint = checkpoint(config, "256p", 450000)
    assert resolve_resume(config, "640p", 1) == endpoint
    own = checkpoint(config, "640p", 452000)
    broken = checkpoint(config, "640p", 454000)
    (broken / "optimizer.bin").unlink()
    assert resolve_resume(config, "640p", 1) == own


def test_dry_run_has_one_config_and_global_stage_schedule(recipe):
    path, config = recipe
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pretrain.launch",
            "--config",
            str(path),
            "--stage",
            "256p",
            "--nproc_per_node",
            "1",
            "--dry_run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual["command"].count("--config") == 1
    assert actual["schedule_horizon"] == 600000 and actual["stage_end"] == 450000
    assert "--muon_lr" not in actual["command"]


def test_launcher_rejects_concurrent_writer_and_releases_on_exit(recipe):
    path, config = recipe
    marker = path.parent / "started"
    code = (
        "from scripts.pretrain import launch; import pathlib,time; "
        f"launch.watch=lambda *args: (pathlib.Path({str(marker)!r}).touch(),time.sleep(30),0)[-1]; launch.main()"
    )
    args = ["--config", str(path), "--stage", "256p", "--nproc_per_node", "1"]
    first = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        cwd=REPO,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while (
            not marker.exists() and first.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert marker.exists()
        second = subprocess.run(
            [sys.executable, "-m", "scripts.pretrain.launch", *args, "--dry_run"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert second.returncode != 0 and "writer lock" in second.stderr
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGTERM)
        first.communicate(timeout=5)
    lock = Path(config.paths.output_dir) / "ascend-hero-256p/.writer.lock"
    assert subprocess.run(["flock", "-n", str(lock), "true"], timeout=5).returncode == 0
