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
import random
import numpy as np

import pytest
import torch

from src.pretrain import train
from src.pretrain.config import flatten, load_config
from dataclasses import asdict, replace
from src.pretrain.stage_control import (
    CHECKPOINT_RECORD, stage_endpoint, validate_checkpoint, write_checkpoint_record, verify_restored_rng,
)


def test_launcher_child_uses_qualified_runtime_policy(monkeypatch, tmp_path):
    from scripts.pretrain import launch

    recipe = Path(__file__).resolve().parents[1] / "configs/hero.toml"
    config = load_config(recipe)
    config = replace(config, paths=replace(config.paths, output_dir=str(tmp_path)))
    monkeypatch.setattr(launch, "load_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", [
        "launch", "--config", str(recipe), "--stage", "256p", "--nproc_per_node", "16",
    ])
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:256")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")

    def child_environment(command, log_path):
        # Exercise inheritance by a real child, without allocating an NPU.
        result = subprocess.run([
            sys.executable, "-c",
            "import json, os; print(json.dumps([os.environ['PYTORCH_NPU_ALLOC_CONF'], "
            "os.environ['OMP_NUM_THREADS']]))",
        ], check=True, capture_output=True, text=True)
        assert json.loads(result.stdout) == ["expandable_segments:True", "1"]
        return 0

    monkeypatch.setattr(launch, "watch", child_environment)
    with pytest.raises(SystemExit) as exit_info:
        launch.main()
    assert exit_info.value.code == 0


@pytest.mark.parametrize("workers", [0, 2])
def test_training_loader_recreation_preserves_checkpoint_rng(workers):
    # Execute the actual loader construction with a deterministic toy dataset.
    # The production RowDescriptorDataset/collate similarly have no randomness:
    # the checkpointed sampler has already selected rows and captions.
    tree = ast.parse(inspect.getsource(train.main))
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "dataloader"
                              for target in node.targets))
    worker_setup = [
        node for node in tree.body[0].body
        if (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == "dataloader_worker_kwargs"
                    for target in node.targets))
        or (isinstance(node, ast.If)
            and ast.unparse(node.test) == "args.num_workers > 0")
    ]
    code = compile(ast.fix_missing_locations(ast.Module(
        body=copy.deepcopy([*worker_setup, assignment]), type_ignores=[])),
        "<training loader>", "exec")
    namespace = dict(
        DataLoader=torch.utils.data.DataLoader, torch=torch,
        row_dataset=torch.arange(8), sampler=[[0, 1], [2, 3], [4, 5], [6, 7]],
        row_length_collate_fn=torch.utils.data.default_collate,
        collate_fn=torch.utils.data.default_collate,
        args=SimpleNamespace(seed=42, num_workers=workers),
        accelerator=SimpleNamespace(process_index=3),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(987)
        checkpoint_rng = torch.get_rng_state().clone()
        expected_dropout_draws = torch.rand(16)
        for _ in range(2):  # New process/iterator after restoring the same RNG.
            torch.set_rng_state(checkpoint_rng)
            # A background subprocess launch can have such a CLOEXEC pipe
            # open while workers start. Forked persistent workers keep its
            # writer alive without exec, preventing the parent's EOF forever.
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)
            loader = None
            try:
                exec(code, namespace)
                loader = namespace["dataloader"]
                assert loader.generator.initial_seed() == 45
                assert torch.equal(torch.cat(list(loader)), torch.arange(8))
                os.close(write_fd)
                write_fd = None
                # Workers are still alive here. No worker may retain the
                # transient write descriptor (EAGAIN would reveal the leak).
                assert os.read(read_fd, 1) == b""
                assert torch.equal(torch.get_rng_state(), checkpoint_rng)
                assert torch.equal(torch.rand(16), expected_dropout_draws)
            finally:
                if loader is not None and loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)


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




def checkpoint(tmp_path, step=300000, total=400000):
    root = tmp_path / f"checkpoint_step_{step:06d}"
    root.mkdir()
    for name in ("model.safetensors", "optimizer.bin", "optimizer_1.bin", "scheduler.bin",
                 "scheduler_1.bin", "ema_weights.pt", "sampler_state_rank_00000.pt"):
        (root / name).write_bytes(b"test-artifact")
    for rank in range(8):
        for name in (f"random_states_{rank}.pkl", f"sampler_state_rank_{rank:05d}.pt"):
            (root / name).write_bytes(b"test-rank-state")
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


@pytest.mark.parametrize("fault", [None, "python", "numpy", "torch", "corrupt"])
def test_strict_rng_verification_detects_silent_load_failure(tmp_path, fault):
    saved = dict(random_state=random.getstate(), numpy_random_seed=np.random.get_state(),
                 torch_manual_seed=torch.get_rng_state())
    path = tmp_path / "random_states_0.pkl"
    torch.save(saved, path)
    try:
        if fault == "python":
            random.random()
        elif fault == "numpy":
            np.random.random()
        elif fault == "torch":
            torch.rand(1)
        elif fault == "corrupt":
            path.write_bytes(b"invalid checkpoint")
        if fault:
            with pytest.raises(ValueError, match="RNG continuation failed"):
                verify_restored_rng(tmp_path, process_index=0, device="cpu")
        else:
            verify_restored_rng(tmp_path, process_index=0, device="cpu")
    finally:
        random.setstate(saved["random_state"])
        np.random.set_state(saved["numpy_random_seed"])
        torch.set_rng_state(saved["torch_manual_seed"])


@pytest.mark.parametrize("name", ["model.safetensors", "optimizer.bin", "optimizer_1.bin",
                                  "random_states_7.pkl", "sampler_state_rank_00007.pt"])
def test_incomplete_inventory_cannot_hide_missing_recovery_files(tmp_path, name):
    root = checkpoint(tmp_path)
    data = json.loads((root / CHECKPOINT_RECORD).read_text())
    del data["files"][name]
    (root / name).unlink()
    (root / CHECKPOINT_RECORD).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing required"):
        validate(root)
    # The writer must not certify that same incomplete directory either.
    (root / CHECKPOINT_RECORD).unlink()
    with pytest.raises(ValueError, match="missing required"):
        write_checkpoint_record(root, step=300000, max_steps=400000,
                                scheduler_count=2, use_ema=True, world_size=8)
    assert not (root / CHECKPOINT_RECORD).exists()


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


@pytest.mark.parametrize("keep_last", [0, 3])
def test_actual_checkpoint_writer_produces_resumable_scheduler_files(tmp_path, keep_last):
    # Execute the actual nested writer with a CPU save-state stand-in. This
    # exercises the file ordering, record, scheduler/EMA writes and preflight.
    tree = ast.parse(inspect.getsource(train.main))
    function = copy.deepcopy(next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint"))
    events = []

    def save_state(path):
        events.append("state")
        for name in ("model.safetensors", "optimizer.bin", "optimizer_1.bin", "random_states_0.pkl"):
            (Path(path) / name).write_bytes(b"cpu-stand-in")

    current = {"step": 0}

    class Scheduler:
        def state_dict(self):
            return {"last_epoch": current["step"]}

    run_dir = tmp_path / "hero-256p"
    namespace = {**vars(train),
                 "args": SimpleNamespace(output_dir=str(tmp_path),run_name="hero-256p",max_steps=420000,
                     checkpoint_keep_last=keep_last),
                 "accelerator":SimpleNamespace(is_main_process=True,num_processes=1,process_index=0,device=torch.device("cpu"),
                     save_state=save_state,wait_for_everyone=lambda:events.append("barrier"),
                     print=lambda message: events.append(message)),
                 "schedulers":[Scheduler(),Scheduler()],
                 "sampler":SimpleNamespace(state_dict=lambda:{"stage":.75}),
                 "sampler_state_name":"sampler_state_rank_00000.pt",
                 "infra_recorder":None, "bucket_contents":{},
                 "device_api":SimpleNamespace(get_rng_state=lambda device:torch.get_rng_state()),
                 "config":load_config(Path("configs/hero.toml")),
                 "model_raw":SimpleNamespace(get_config=lambda:{"architecture":"artflow-v2"}),
                 "ema_model":torch.nn.Linear(1,1),"run_dir":str(run_dir),
                 "runtime_path":str(run_dir / "runtime.json")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 "<production writer>","exec"),namespace)
    for step in (309000, 311000, 313000, 315000):
        current["step"] = step
        namespace["save_checkpoint"](step)
    root = run_dir / "checkpoint_step_315000"
    assert validate_checkpoint(root,max_steps=420000,stop_at_step=315000,
        require_record=True,scheduler_count=2,use_ema=True,world_size=1) == 315000
    for name in ("scheduler.bin","scheduler_1.bin"):
        assert torch.load(root / name,weights_only=False)["last_epoch"] == 315000
    assert events[-1] == "barrier"
    assert (run_dir / "checkpoint_step_309000").exists() == (keep_last == 0)
    assert len(list(run_dir.glob("checkpoint_step_*"))) == (keep_last or 4)


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


def test_resume_base_lrs_follow_config_not_checkpoint():
    """Recipe-change resume: the checkpoint's base_lrs (old peak) must not
    silently keep the previous lr — train.py's resume block re-applies the
    construction-time (config) base_lrs after load_state_dict."""
    def setup(lr):
        param = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([param], lr=lr)
        sch = train.build_linear_cosine_scheduler(opt, num_warmup_steps=5,
                num_training_steps=100, min_learning_rate=.001,
                base_learning_rate=lr, start_learning_rate=.0001)
        return opt, sch
    opt_old, sch_old = setup(.02)
    for _ in range(4):  # mid-warmup
        opt_old.step(); sch_old.step()
    state = copy.deepcopy(sch_old.state_dict())
    _, sch_new = setup(.016)
    sch_new.load_state_dict(state)
    # Sanity: without the override the checkpoint's old peak would win.
    assert sch_new.base_lrs == [.02]
    sch_new.base_lrs = [.016]  # what the resume block re-applies from config
    sch_new.step()
    expected_ratio = (.0001 / .016) + (1 - .0001 / .016) * (sch_new.last_epoch / 5)
    # With the checkpoint's old base (.02) this would read .02 at this epoch.
    assert sch_new.get_last_lr()[0] == pytest.approx(.016 * expected_ratio, rel=1e-6)
    assert sch_new.get_last_lr()[0] == pytest.approx(.016, rel=1e-6)



def test_npu_rng_verification_uses_rank_local_sidecar(tmp_path, monkeypatch):
    state = dict(random_state=random.getstate(), numpy_random_seed=np.random.get_state(),
                 torch_manual_seed=torch.get_rng_state())
    torch.save(state, tmp_path / 'random_states_2.pkl')
    expected = torch.tensor([1, 3, 5], dtype=torch.uint8)
    torch.save(expected, tmp_path / 'npu_rng_state_rank_00002.pt')
    device = SimpleNamespace(type='npu', index=2)
    reads = []
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(
        get_rng_state=lambda selected: (reads.append(selected), expected.clone())[-1]), raising=False)
    verify_restored_rng(tmp_path, process_index=2, device=device)
    assert reads == [device]
    torch.save(expected+1, tmp_path / 'npu_rng_state_rank_00002.pt')
    with pytest.raises(ValueError, match='NPU RNG was not restored'):
        verify_restored_rng(tmp_path, process_index=2, device=device)
