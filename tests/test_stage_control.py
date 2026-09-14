"""Stage stopping must not shorten the global scheduler/curriculum horizon."""

import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest
import torch

from src.train import train
from src.train.config import flatten, load_config
from src.train.stage_control import (
    CHECKPOINT_RECORD, stage_endpoint, validate_checkpoint, write_checkpoint_record,
)


@pytest.mark.parametrize("total", [400000, 420000])
@pytest.mark.parametrize("fraction", [.75, .95, 1.0])
def test_actual_loop_guard_stops_at_endpoint_without_resume_update(total, fraction):
    endpoint = int(total * fraction)
    tree = ast.parse(inspect.getsource(train.main))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                and ast.unparse(n.test) == "global_step < end_step")
    # Execute the production loop guard around a one-update stub.
    module = ast.Module(body=[ast.While(test=copy.deepcopy(loop.test),
                       body=ast.parse("global_step += 1").body, orelse=[])], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), "<loop guard>", "exec")
    for resumed in (endpoint - 2, endpoint):
        ns = {"global_step": resumed, "end_step": stage_endpoint(total, endpoint, resumed)}
        exec(code, ns)
        assert ns["global_step"] == endpoint
    with pytest.raises(ValueError, match="beyond"):
        stage_endpoint(total, endpoint, endpoint + 1)


@pytest.mark.parametrize("stop", [-1, True, 1.5, "100", 401])
def test_invalid_stop_rejected(stop):
    with pytest.raises(ValueError, match="stop_at_step"):
        stage_endpoint(400, stop)


def test_zero_disabled_and_config_flattening(tmp_path):
    assert stage_endpoint(400, 0) == 400
    assert stage_endpoint(0, 0) == 0  # Existing eval-only mode.
    path = tmp_path / "stage.toml"
    path.write_text("[train]\nmax_steps = 420000\nstop_at_step = 315000\n")
    args = flatten(load_config([str(path)]))
    assert args["max_steps"] == 420000
    assert args["stop_at_step"] == 315000
    path.write_text("[train]\nmax_steps = 400\nstop_at_step = true\n")
    with pytest.raises(ValueError, match="stop_at_step"):
        load_config([str(path)])


def checkpoint(tmp_path, step=300000, total=400000):
    root = tmp_path / f"checkpoint_step_{step:06d}"
    root.mkdir()
    for name in ("model.safetensors", "optimizer.bin", "optimizer_1.bin", "scheduler.bin",
                 "scheduler_1.bin", "ema_weights.pt", "sampler_state_rank_00000.pt"):
        (root / name).write_bytes(b"test-artifact")
    write_checkpoint_record(root, step=step, max_steps=total, scheduler_count=2,
                            use_ema=True, world_size=8)
    return root


def validate(root, **kwargs):
    return validate_checkpoint(root, max_steps=400000, stop_at_step=380000,
                               require_record=True, scheduler_count=2, use_ema=True,
                               world_size=8, **kwargs)


def test_predecessor_endpoint_and_same_stage_range(tmp_path):
    root = checkpoint(tmp_path)
    assert validate(root, expected_step=300000) == 300000
    with pytest.raises(ValueError, match="interval/endpoint"):
        validate(root, expected_step=299999)
    with pytest.raises(ValueError, match="interval/endpoint"):
        validate(root, min_step=300001)


@pytest.mark.parametrize("fault", ["record", "horizon", "step", "scheduler", "truncated", "ema", "world"])
def test_incomplete_or_incompatible_checkpoints_rejected(tmp_path, fault):
    root = checkpoint(tmp_path)
    record = root / CHECKPOINT_RECORD
    if fault == "record":
        record.unlink()
    elif fault in ("scheduler", "ema"):
        (root / ("scheduler_1.bin" if fault == "scheduler" else "ema_weights.pt")).unlink()
    elif fault == "truncated":
        (root / "optimizer.bin").write_bytes(b"x")
    else:
        data = json.loads(record.read_text())
        data[{"horizon": "max_steps", "step": "global_step", "world": "world_size"}[fault]] += 1
        record.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate(root)


def test_legacy_resume_does_not_claim_verified_horizon(tmp_path):
    root = tmp_path / "checkpoint_step_000010"
    root.mkdir()
    assert validate_checkpoint(root, max_steps=100) == 10
    with pytest.raises(ValueError, match="recorded global T"):
        validate_checkpoint(root, max_steps=100, stop_at_step=75, require_record=True)


def test_save_before_eval_once_even_when_endpoint_is_off_cadence():
    tree = ast.parse(inspect.getsource(train.main))
    condition = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "global_step % args.checkpoint_interval == 0 or global_step == end_step")
    code = compile(ast.Expression(condition.test), "<checkpoint guard>", "eval")
    for step, end, expected in [(315000,315000,True), (300000,300000,True),
                                (314000,315000,True), (314999,315000,False)]:
        assert eval(code, {"global_step":step,"end_step":end,
                           "args":SimpleNamespace(checkpoint_interval=2000)}) is expected
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                and ast.unparse(n.test) == "global_step < end_step")
    evaluation = next(n for n in ast.walk(loop) if isinstance(n,ast.Call)
                      and isinstance(n.func,ast.Name) and n.func.id == "evaluate_grid")
    assert condition.lineno < evaluation.lineno
    saves = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Name) and n.func.id == "save_checkpoint"]
    assert len(saves) == 1


def test_record_published_after_state_writes_and_barrier():
    tree = ast.parse(inspect.getsource(train.main))
    save = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == "save_checkpoint")
    source = ast.unparse(save)
    assert source.index("accelerator.save_state") < source.index("write_checkpoint_record")
    assert source.index("ema_weights.pt") < source.index("write_checkpoint_record")
    assert source.count("accelerator.wait_for_everyone()") >= 4


def test_actual_checkpoint_writer_produces_resumable_scheduler_files(tmp_path):
    # Execute the actual nested writer with a CPU save-state stand-in. This
    # exercises the file ordering, record, scheduler/EMA writes and preflight.
    tree = ast.parse(inspect.getsource(train.main))
    function = copy.deepcopy(next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint"))
    events = []

    def save_state(path):
        events.append("state")
        for name in ("model.safetensors", "optimizer.bin", "optimizer_1.bin"):
            (Path(path) / name).write_bytes(b"cpu-stand-in")

    class Scheduler:
        def state_dict(self):
            return {"last_epoch": 315000}

    run_dir = tmp_path / "hero-256p"
    namespace = {**vars(train),
                 "args": SimpleNamespace(output_dir=str(tmp_path),run_name="hero-256p",max_steps=420000),
                 "accelerator":SimpleNamespace(is_main_process=True,num_processes=1,
                     save_state=save_state,wait_for_everyone=lambda:events.append("barrier")),
                 "schedulers":[Scheduler(),Scheduler()],
                 "sampler":SimpleNamespace(state_dict=lambda:{"stage":.75}),
                 "sampler_state_name":"sampler_state_rank_00000.pt",
                 "ema_model":torch.nn.Linear(1,1),"run_dir":str(run_dir),
                 "runtime_path":str(run_dir / "runtime.json")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 "<production writer>","exec"),namespace)
    namespace["save_checkpoint"](315000)
    root = run_dir / "checkpoint_step_315000"
    assert validate_checkpoint(root,max_steps=420000,stop_at_step=315000,
        require_record=True,scheduler_count=2,use_ema=True,world_size=1) == 315000
    for name in ("scheduler.bin","scheduler_1.bin"):
        assert torch.load(root / name,weights_only=False)["last_epoch"] == 315000
    assert events[-1] == "barrier"


def test_scheduler_roundtrip_preserves_next_lr_and_global_curriculum():
    def setup():
        param = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([param], lr=.02)
        sch = train.build_linear_cosine_scheduler(opt, num_warmup_steps=5,
                num_training_steps=100, min_learning_rate=.001,
                base_learning_rate=.02, start_learning_rate=.0001)
        return opt, sch
    opt, sch = setup()
    for _ in range(75):
        opt.step()
        sch.step()
    state, optim_state = copy.deepcopy(sch.state_dict()), copy.deepcopy(opt.state_dict())
    resumed_opt, resumed_sch = setup()
    resumed_opt.load_state_dict(optim_state)
    resumed_sch.load_state_dict(state)
    for _ in range(20):
        opt.step(); sch.step()
        resumed_opt.step(); resumed_sch.step()
        assert resumed_sch.get_last_lr() == sch.get_last_lr()
    assert resumed_sch.last_epoch == 95
    assert sch.get_last_lr()[0] > .001  # The schedule has not ended at 95%.


def launcher_env(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "repo").symlink_to(repo, target_is_directory=True)
    (workspace / "jobs").mkdir()
    (workspace / "jobs/swanlab_login.sh").write_text("# No external login in tests.\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # Real CPU preflight, fake torchrun: capture the resolved launcher arguments
    # and its temporary override without loading models or starting a GPU job.
    python = bindir / "python3"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, runpy, sys, tomllib
if sys.argv[1:3] == ["-m", "src.train.stage_control"]:
    sys.argv = [sys.argv[2], *sys.argv[3:]]
    runpy.run_module("src.train.stage_control", run_name="__main__")
else:
    paths = [sys.argv[i+1] for i,a in enumerate(sys.argv) if a == "--config"]
    config = tomllib.loads(pathlib.Path(paths[-1]).read_text())
    pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps({"args":sys.argv,"config":config}))
''')
    python.chmod(0o755)
    env = {**os.environ, "ARTFLOW_ROOT": str(workspace), "PYTHONPATH": str(repo),
           "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
           "CAPTURE": str(tmp_path / "captured.json")}
    return repo, workspace, env


@pytest.mark.parametrize("total", [400000, 420000])
@pytest.mark.parametrize("stage,start_frac,end_frac,accum,prev", [
    ("256p",0,75,1,None), ("640p",75,95,5,"256p"), ("896p",95,100,7,"640p"),
])
def test_actual_launcher_preflight_and_overrides(tmp_path, total, stage, start_frac, end_frac, accum, prev):
    repo, workspace, env = launcher_env(tmp_path)
    before = None
    if prev:
        run = workspace / "runs" / f"hero-{prev}"
        run.mkdir(parents=True)
        predecessor = checkpoint(run, step=total * start_frac // 100, total=total)
        before = {p.name:p.read_bytes() for p in predecessor.iterdir()}
    result = subprocess.run(["bash", str(repo / "jobs/hero_stage.sh"), stage, str(total)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads(Path(env["CAPTURE"]).read_text())
    config = captured["config"]
    assert config["train"]["max_steps"] == total
    assert config["train"]["stop_at_step"] == total * end_frac // 100
    assert config["train"]["gradient_accumulation_steps"] == accum
    assert ("--reset_sampler" in captured["args"]) is bool(prev)
    if prev:
        assert {p.name:p.read_bytes() for p in predecessor.iterdir()} == before


@pytest.mark.parametrize("fault", ["step", "horizon", "incomplete", "own_past_end"])
def test_launcher_rejects_bad_checkpoint_before_torchrun(tmp_path, fault):
    repo, workspace, env = launcher_env(tmp_path)
    run = workspace / "runs" / ("hero-640p" if fault == "own_past_end" else "hero-256p")
    run.mkdir(parents=True)
    step = {"step":298000,"own_past_end":380001}.get(fault,300000)
    root = checkpoint(run, step=step, total=420000 if fault == "horizon" else 400000)
    if fault == "incomplete":
        (root / CHECKPOINT_RECORD).unlink()
    result = subprocess.run(["bash",str(repo / "jobs/hero_stage.sh"),"640p","400000"],
                            env=env,capture_output=True,text=True)
    assert result.returncode != 0
    assert not Path(env["CAPTURE"]).exists()
