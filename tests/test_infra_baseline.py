import json
import ast
import inspect
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from scripts.bench import infra_baseline
from src.train import train
from src.train.config import flatten, load_config


@pytest.mark.parametrize("ranks", [1, 2, 4, 6, 8])
def test_throughput_probe_disables_eval_initialization(tmp_path, monkeypatch, ranks):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
                                     "--tag", "probe", "--ranks", str(ranks)])
    assert infra_baseline.main() == 0
    results = json.loads((tmp_path / "runs/probe/results.json").read_text())
    assert len(results) == 3
    for row in results:
        config = tomllib.loads((tmp_path / "runs/probe" / f"{row['stage']}.toml").read_text())
        # A large interval does not disable the initial EvalLossProbe load.
        assert config["eval"]["loss_interval"] == 0
        assert config["eval"]["grid_steps"] == []
        assert config["eval"]["kid_at_end"] is False
        assert config["train"]["max_steps"] == 400000
        assert row["world_size"] == ranks
        assert f"--nproc_per_node={ranks}" in row["command"]
    assert len([cmd for cmd, _ in calls if cmd[0] == "timeout"]) == 3


def test_dynamic_compile_is_opt_in():
    parser = train.parse_args()
    base = ["--config", "configs/base.toml"]
    assert parser.parse_args(base).compile_dynamic is False
    assert parser.parse_args([*base, "--compile_dynamic"]).compile_dynamic is True
    assert parser.parse_args(base).compile_autotune is False
    assert parser.parse_args([*base, "--compile_autotune"]).compile_autotune is True
    assert parser.parse_args(base).disable_ddp_compile_split is False
    assert parser.parse_args([*base, "--disable_ddp_compile_split"]).disable_ddp_compile_split is True
    assert parser.parse_args(base).ddp_gradient_bucket_views is False
    assert parser.parse_args([*base, "--ddp_gradient_bucket_views"]).ddp_gradient_bucket_views is True
    assert parser.parse_args(base).native_flash_varlen is False
    assert parser.parse_args([*base, "--native_flash_varlen"]).native_flash_varlen is True
    assert parser.parse_args(base).real_rope is False
    assert parser.parse_args([*base, "--real_rope"]).real_rope is True
    assert parser.parse_args(base).muon_compile_square_ns is False
    assert parser.parse_args([*base, "--muon_compile_square_ns"]).muon_compile_square_ns is True


def test_candidate_plan_and_accumulation_are_isolated(tmp_path, monkeypatch):
    plan = tmp_path / 'candidate plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
        "--tag", "candidate", "--stage", "640p", "--accumulation", "2",
        "--bucket-plan", str(plan)])
    assert infra_baseline.main() == 0
    cfg = tomllib.loads((tmp_path / 'runs/candidate/640p.toml').read_text())
    assert cfg['train']['gradient_accumulation_steps'] == 2
    assert cfg['data']['bucket_plan'] == str(plan)
    assert plan.read_text() == '{}'


def test_candidate_override_requires_single_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
        "--tag", "invalid", "--accumulation", "2"])
    with pytest.raises(SystemExit):
        infra_baseline.main()
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize("flags,expected", [(["--no-cache-clear"], 0),
                                         (["--cache-clear-interval", "100"], 100)])
def test_explicit_cleanup_cadence_survives_hero_default(tmp_path, monkeypatch, flags, expected):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
                        "--tag", "cleanup", "--stage", "256p", *flags])
    assert infra_baseline.main() == 0
    config = tomllib.loads((tmp_path / "runs/cleanup/256p.toml").read_text())
    assert config["telemetry"]["cache_clear_interval"] == expected


@pytest.mark.parametrize("fraction", [0., 1 / 6, .5, 5 / 6, 1.])
def test_probe_slices_global_stage_intervals(tmp_path, monkeypatch, fraction):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
                                     "--tag", "slices", "--curriculum-fraction", str(fraction),
                                     "--no-cpu-wall-profile", "--gpu-health-snapshot"])
    assert infra_baseline.main() == 0
    results = json.loads((tmp_path / "runs/slices/results.json").read_text())
    for row, interval in zip(results, [(0., .75), (.75, .95), (.95, 1.)]):
        progress = interval[0] + (interval[1] - interval[0]) * fraction
        assert row["caption_progress"] == pytest.approx(progress)
        config = tomllib.loads((tmp_path / "runs/slices" / f"{row['stage']}.toml").read_text())
        assert config["data"]["curriculum_start"] == pytest.approx(progress)
        assert config["data"]["curriculum_end"] == pytest.approx(progress)
        assert row["stage_progress_interval"] == list(interval)
        assert "--cpu_wall_profile" not in row["command"]
        assert "--gpu_health_snapshot" in row["command"]


@pytest.mark.parametrize("fraction", ["-0.1", "1.1", "nan", "inf"])
def test_probe_rejects_invalid_curriculum_slice(tmp_path, monkeypatch, fraction):
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
                                     "--tag", "invalid", "--curriculum-fraction", fraction])
    with pytest.raises(SystemExit):
        infra_baseline.main()
    assert not (tmp_path / "runs").exists()


def test_ddp_split_workaround_is_explicit_in_diagnostic_command(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
                                     "--tag", "split", "--ranks", "4", "--stage", "256p",
                                     "--dynamic-blocks", "--disable-ddp-compile-split", "--real-rope",
                                     "--ddp-gradient-bucket-views"])
    assert infra_baseline.main() == 0
    result = json.loads((tmp_path / "runs/split/results.json").read_text())[0]
    assert "--disable_ddp_compile_split" in result["command"]
    assert "--compile_dynamic" in result["command"]
    assert "--real_rope" in result["command"]
    assert "--ddp_gradient_bucket_views" in result["command"]


def test_autotune_probe_preserves_explicit_cache_and_flags(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    cache = str(tmp_path / "isolated-compiler-cache")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", cache)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["infra_baseline", "--root", str(tmp_path),
                                     "--tag", "autotune", "--ranks", "2", "--stage", "640p",
                                     "--dynamic-blocks", "--autotune-blocks"])
    assert infra_baseline.main() == 0
    command, kwargs = next((cmd, kw) for cmd, kw in calls if cmd[0] == "timeout")
    assert "--compile_autotune" in command
    assert "--compile_dynamic" in command
    assert kwargs["env"]["TORCHINDUCTOR_CACHE_DIR"] == cache


def test_disabled_cache_housekeeping_never_divides_by_zero(tmp_path):
    config = tmp_path / "cache.toml"
    config.write_text("[telemetry]\ncache_clear_interval = 0\n")
    assert flatten(load_config([str(config)]))["cache_clear_interval"] == 0
    tree = ast.parse(inspect.getsource(train.main))
    condition = next(n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                     and "global_step % args.cache_clear_interval" in ast.unparse(n.test))
    code = compile(ast.Expression(condition), "<cache guard>", "eval")
    for interval, expected in [(0, False), (100, True), (101, False)]:
        assert eval(code, dict(global_step=100, args=SimpleNamespace(cache_clear_interval=interval))) == expected


@pytest.mark.parametrize("value", ["-1", "true", "1.5"])
def test_invalid_cache_interval_rejected(tmp_path, value):
    path = tmp_path / "cache.toml"
    path.write_text(f"[telemetry]\ncache_clear_interval = {value}\n")
    with pytest.raises(ValueError, match="cache_clear_interval"):
        load_config([str(path)])
