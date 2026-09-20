import json
import subprocess
import sys

import pytest

from scripts.bench import infra_stage_chain as chain
from src.train.config import load_config


def test_diagnostic_override_keeps_continuous_horizon_and_accumulation(tmp_path):
    for _, _, end, accumulation in chain.STAGES:
        path = tmp_path / "override.toml"
        path.write_text(chain.override_text(tmp_path, end=end, accumulation=accumulation))
        config = load_config(["configs/base.toml", "configs/hero.toml", str(path)])
        assert config.train.max_steps == 40
        assert config.train.stop_at_step == end
        assert config.optim.lr_warmup_steps == 5
        assert config.train.gradient_accumulation_steps == accumulation
        assert (config.data.curriculum_start, config.data.curriculum_end) == (0, 1)
        assert config.eval.loss_interval == 0


def test_progress_rejects_resolution_local_restart(tmp_path):
    (tmp_path / "infra").mkdir()
    path = tmp_path / "infra/rank-0.jsonl"
    rows = [dict(step=s, rank=0, loss=.2, progress=s / 40) for s in (39, 40)]
    path.write_text("\n".join(map(json.dumps, rows)))
    chain.check_progress(tmp_path, start=38, end=40, ranks=1)
    rows[0]["progress"] = .5
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="horizon"):
        chain.check_progress(tmp_path, start=38, end=40, ranks=1)


def test_override_can_pin_no_periodic_cleanup(tmp_path):
    path = tmp_path / "override.toml"
    path.write_text(chain.override_text(tmp_path, end=30, accumulation=1,
                                        cache_clear_interval=0))
    config = load_config(["configs/base.toml", str(path)])
    assert config.telemetry.cache_clear_interval == 0
    with pytest.raises(ValueError, match="nonnegative"):
        chain.override_text(tmp_path, end=30, accumulation=1, cache_clear_interval=-1)


def test_eval_override_keeps_real_dataset_and_full_final_kid(tmp_path):
    stage_config = tmp_path / "stage.toml"
    stage_config.write_text('[eval]\ndataset_path = "/diagnostic/real-eval"\n')
    for _, _, end, accumulation in chain.STAGES:
        path = tmp_path / "eval.toml"
        path.write_text(chain.override_text(tmp_path, end=end, accumulation=accumulation,
                                            with_evaluation=True))
        config = load_config(["configs/base.toml", str(stage_config), "configs/hero.toml", str(path)])
        assert config.eval.dataset_path == "/diagnostic/real-eval"
        assert config.eval.grid_steps == [end]
        assert config.eval.loss_interval == end
        assert config.eval.loss_samples == 512
        assert config.eval.kid_num_fake == 2000
        assert config.eval.kid_at_end == (end == chain.HORIZON)


def test_evaluation_requires_post_update_loss_and_full_fake_count(tmp_path):
    (tmp_path / "samples").mkdir()
    panel = dict(step=40, weights="ema", ode_steps=50, precision="bf16", solver="euler",
                 cfg_scale=1., prompts=[{}] * 48)
    (tmp_path / "samples/panel_step_000040.json").write_text(json.dumps(panel))
    for i in range(12):
        (tmp_path / f"samples/grid_step_000040_{i}.png").touch()
    events = [dict(phase=phase, complete=True, rank=0, seconds=1.)
              for phase in ("loss_setup", "loss", "loss", "grid", "kid")]
    timing = tmp_path / "eval-rank-0.jsonl"
    timing.write_text("\n".join(map(json.dumps, events)))
    kid = dict(zip(("kid/num_fake", "kid/num_real", "kid/mean", "kid/std"), (2000, 2432, .5, .01)))
    path = tmp_path / "kid_step_000040.json"
    path.write_text(json.dumps(kid))
    chain.check_evaluation(tmp_path, end=40, ranks=1, expected_real=2432)
    kid["kid/num_real"] = 1038
    path.write_text(json.dumps(kid))
    chain.check_evaluation(tmp_path, end=40, ranks=1, expected_real=1038)
    with pytest.raises(ValueError, match="KID"):
        chain.check_evaluation(tmp_path, end=40, ranks=1, expected_real=1039)
    kid["kid/num_fake"] = 128
    path.write_text(json.dumps(kid))
    with pytest.raises(ValueError, match="KID"):
        chain.check_evaluation(tmp_path, end=40, ranks=1, expected_real=1038)
    events.pop(2)
    timing.write_text("\n".join(map(json.dumps, events)))
    with pytest.raises(ValueError, match="sequence"):
        chain.check_evaluation(tmp_path, end=40, ranks=1)


@pytest.mark.parametrize("invariant_exact", [True, False])
def test_chain_commands_and_failfast(tmp_path, monkeypatch, invariant_exact):
    out = tmp_path / "chain"
    monkeypatch.setattr(sys, "argv", ["chain", "--root", str(tmp_path), "--out", str(out),
                                     "--ranks", "8", "--train-arg=--compile_dynamic",
                                     "--cache-clear-interval", "0",
                                     "--256p-accumulation", "2",
                                     "--640p-accumulation", "2",
                                     "--896p-accumulation", "3",
                                     "--256p-train-arg=--compile_autotune"])
    commands = []
    monkeypatch.setattr(chain, "validate_checkpoint", lambda *a, **kw: None)
    monkeypatch.setattr(chain, "check_saved_state", lambda *a, **kw: None)
    monkeypatch.setattr(chain, "check_progress", lambda *a, **kw: None)
    monkeypatch.setattr(chain, "check_restoration", lambda *a, **kw: None)
    monkeypatch.setattr(chain, "check_replay_inputs", lambda *a, **kw: None)
    deterministic = ["scheduler.bin", "scheduler_1.bin"] + [f"random_states_{r}.pkl" for r in range(8)]
    monkeypatch.setattr(chain, "compare_checkpoints", lambda *a: dict(
        exact=False, files={name: dict(exact=invariant_exact) for name in deterministic}))

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    assert chain.main() == (0 if invariant_exact else 1)
    assert len(commands) == (4 if invariant_exact else 2)
    for command in commands:
        assert "--nproc_per_node=8" in command
        assert "--compile_dynamic" in command
    assert "--resume" not in commands[0]
    assert all("--compile_autotune" in command for command in commands[:2])
    assert all("--compile_autotune" not in command for command in commands[2:])
    for name in ("256p", "replay-256p"):
        assert load_config([str(out / f"{name}.toml")]).telemetry.cache_clear_interval == 0
        assert load_config([str(out / f"{name}.toml")]).train.gradient_accumulation_steps == 2
    assert "--resume_full" in commands[1]
    assert "--verify_resume_state" in commands[1]
    assert "--reset_sampler" not in commands[1]
    assert "checkpoint_step_000028" in commands[1][commands[1].index("--resume") + 1]
    for command, expected in zip(commands[2:], (30, 38)):
        assert "--reset_sampler" in command
        assert f"checkpoint_step_{expected:06d}" in command[command.index("--resume") + 1]
    report = json.loads((out / "results.json").read_text())
    assert report["verified"] is invariant_exact
    assert report["replay"]["exact"] is False  # Numerical differences remain visible.
    assert report["world_size"] == 8
    assert [case['accumulation'] for case in report['cases']] == (
        [2, 2, 2, 3] if invariant_exact else [2, 2])


def test_restoration_report_requires_all_state_and_rng(tmp_path):
    files = {name: dict(exact=True) for name in (
        "model.safetensors", "optimizer.bin", "optimizer_1.bin", "scheduler.bin",
        "scheduler_1.bin", "ema_weights.pt")}
    report = dict(exact=True, rng_exact=True, files=files, step=28, rank=0, max_steps=40)
    path = tmp_path / "restore_step_000028_rank_00000.json"
    path.write_text(json.dumps(report))
    chain.check_restoration(tmp_path, step=28, ranks=1)
    del files["ema_weights.pt"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="restoration"):
        chain.check_restoration(tmp_path, step=28, ranks=1)


def test_replay_inputs_require_actual_identity_not_only_shapes(tmp_path):
    a, b = tmp_path / "reference", tmp_path / "candidate"
    rows = [dict(step=step, rank=0, global_samples=4, progress=step/40,
                 shapes=[dict(shape=[2, 16, 4, 4, 32], count=1)],
                 sample_identity_sha256="a"*64) for step in (29, 30)]
    for path in (a, b):
        (path / "infra").mkdir(parents=True)
        (path / "infra/rank-0.jsonl").write_text("\n".join(map(json.dumps, rows)))
    chain.check_replay_inputs(a, b, start=28, end=30, ranks=1)
    rows[0]["sample_identity_sha256"] = "b"*64
    (b / "infra/rank-0.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="identity"):
        chain.check_replay_inputs(a, b, start=28, end=30, ranks=1)
