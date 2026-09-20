"""Bounded real-training stage-chain acceptance; never launches the hero run.

Use the final pinned execution flags and eight ranks for acceptance. A diagnostic
T=40 and warmup=5 exercise warmup, cosine and global caption progression within
30/8/2 updates. This is not the production horizon or a performance estimate.
Run separately from storage-sensitive timing measurements.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from scripts.bench.compare_resume_state import compare_checkpoints
from scripts.bench.infra_resume_smoke import check_saved_state, check_updates
from src.train.stage_control import validate_checkpoint


HORIZON = 40
STAGES = (("256p", 0, 30, 1), ("640p", 30, 38, 5), ("896p", 38, 40, 7))


def check_restoration(run, *, step, ranks):
    required = {"optimizer.bin", "optimizer_1.bin", "scheduler.bin", "scheduler_1.bin",
                "ema_weights.pt"}
    for rank in range(ranks):
        path = run / f"restore_step_{step:06d}_rank_{rank:05d}.json"
        report = json.loads(path.read_text())
        files = report.get("files", {})
        if (report.get("exact") is not True or report.get("rng_exact") is not True
                or report.get("rank") != rank or report.get("step") != step
                or report.get("max_steps") != HORIZON or not required <= files.keys()
                or not {"model.safetensors", "pytorch_model.bin"} & files.keys()
                or any(value.get("exact") is not True for value in files.values())):
            raise ValueError(f"rank {rank}: incomplete or failed live-state restoration proof")


def check_replay_inputs(reference, candidate, *, start, end, ranks):
    for rank in range(ranks):
        def read(run):
            rows = [json.loads(line) for line in
                    (run / "infra" / f"rank-{rank}.jsonl").read_text().splitlines()]
            return [row for row in rows if start < row["step"] <= end]
        a, b = read(reference), read(candidate)
        if ([row["step"] for row in a] != list(range(start + 1, end + 1))
                or [row["step"] for row in b] != list(range(start + 1, end + 1))):
            raise ValueError("incomplete replay input window")
        for x, y in zip(a, b):
            for row in (x, y):
                digest = row.get("sample_identity_sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("missing replay row/caption identity digest")
            for field in ("rank", "global_samples", "progress", "shapes", "sample_identity_sha256"):
                if x[field] != y[field]:
                    raise ValueError(f"rank {rank}, step {x['step']}: replay {field} differs")


def check_progress(run, *, start, end, ranks):
    check_updates(run, start=start, end=end, ranks=ranks)
    for rank in range(ranks):
        rows = [json.loads(line) for line in
                (run / "infra" / f"rank-{rank}.jsonl").read_text().splitlines()]
        # InfraRecorder records the updated sampler stage after each update.
        for row in rows:
            if not math.isclose(row["progress"], row["step"] / HORIZON,
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"rank {rank}: caption horizon restarted or drifted")


def override_text(out, *, end, accumulation, cache_clear_interval=None,
                  with_evaluation=False):
    evaluation = (f'''grid_steps = [{end}]
loss_interval = {end}
loss_samples = 512
kid_at_end = {str(end == HORIZON).lower()}
kid_num_fake = 2000
ode_steps = 50
''' if with_evaluation else '''dataset_path = ""
grid_steps = []
loss_interval = 0
kid_at_end = false
''')
    text = f'''# Diagnostic horizon, not a replacement hero recipe.
[data]
curriculum_start = 0.0
curriculum_end = 1.0
stage_sync_interval = 1
[train]
max_steps = {HORIZON}
stop_at_step = {end}
gradient_accumulation_steps = {accumulation}
checkpoint_interval = 28
eval_interval = 1000000000
[optim]
lr_warmup_steps = 5
[eval]
{evaluation}
[paths]
output_dir = {json.dumps(str(out.resolve()))}
'''
    if cache_clear_interval is not None:
        if type(cache_clear_interval) is not int or cache_clear_interval < 0:
            raise ValueError("cache clear interval must be a nonnegative integer")
        text += f"[telemetry]\ncache_clear_interval = {cache_clear_interval}\n"
    return text


def check_evaluation(run, *, end, ranks, expected_real=None):
    panel = json.loads((run / "samples" / f"panel_step_{end:06d}.json").read_text())
    if (panel.get("step") != end or panel.get("weights") != "ema"
            or panel.get("ode_steps") != 50 or panel.get("precision") != "bf16"
            or panel.get("solver") != "euler" or panel.get("cfg_scale") != 1.0
            or len(panel.get("prompts", [])) != 48):
        raise ValueError("evaluation panel does not match the frozen policy")
    if len(list((run / "samples").glob(f"grid_step_{end:06d}_*.png"))) != 12:
        raise ValueError("incomplete evaluation panel images")
    expected = {"loss_setup": 1, "loss": 2, "grid": 1, "kid": int(end == HORIZON)}
    for rank in range(ranks):
        events = [json.loads(line) for line in
                  (run / f"eval-rank-{rank}.jsonl").read_text().splitlines()]
        if {name: sum(e["phase"] == name for e in events) for name in expected} != expected:
            raise ValueError("incomplete evaluation phase sequence")
        for event in events:
            if (event.get("complete") is not True or event.get("rank") != rank
                    or not math.isfinite(event["seconds"]) or event["seconds"] <= 0
                    or any(not math.isfinite(v) for v in event.get("metrics", {}).values())):
                raise ValueError("failed or nonfinite evaluation phase")
    if end == HORIZON:
        kid = json.loads((run / f"kid_step_{end:06d}.json").read_text())
        if (expected_real is None or expected_real < 100
                or kid.get("kid/num_fake") != 2000
                or kid.get("kid/num_real") != expected_real
                or any(not math.isfinite(kid.get(key, float("nan")))
                       for key in ("kid/mean", "kid/std"))):
            raise ValueError("final KID is incomplete or nonfinite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage-config-dir", type=Path,
                        help="Resolved versioned stage inputs; defaults to ROOT/configs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Seconds per bounded training subprocess")
    parser.add_argument("--train-arg", action="append", default=[],
                        help="Explicit final execution flag, e.g. --train-arg=--compile_dynamic")
    parser.add_argument("--cache-clear-interval", type=int,
                        help="Override the periodic allocator policy; 0 disables it")
    parser.add_argument("--with-evaluation", action="store_true",
                        help="Measure full panel/loss/final 2000-fake KID with training state resident")
    for stage, *_ in STAGES:
        parser.add_argument(f"--{stage}-train-arg", action="append", default=[],
                            help="Additional final flag for this resolution, including replay")
        parser.add_argument(f"--{stage}-accumulation", type=int,
                            help="Explicit selected-hardware accumulation; applies to replay too")
    args = parser.parse_args()
    stage_config_dir = args.stage_config_dir or args.root / "configs"
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.cache_clear_interval is not None and args.cache_clear_interval < 0:
        parser.error("cache clear interval must be nonnegative")
    for stage, *_ in STAGES:
        value = getattr(args, f"{stage}_accumulation")
        if value is not None and value < 1:
            parser.error("stage accumulation must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ARTFLOW_ROOT=str(args.root), ARTFLOW_INFRA_METRICS="1",
               ARTFLOW_INFRA_IDENTITIES="1",
               ARTFLOW_TRACE_START="-1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    report = dict(verified=False, world_size=args.ranks, max_steps=HORIZON, cases=[],
                  caveat="Diagnostic T=40/warmup=5; no performance or production-horizon claim. "
                         "Lower ranks do not qualify eight ranks. No image-quality claim.",
                  evaluation_requested=args.with_evaluation)

    def save_report():
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def run_stage(stage, start, end, accumulation, *, checkpoint=None, branch=False):
        name = "replay-256p" if branch else stage
        evaluate = args.with_evaluation and not branch
        override = args.out / f"{name}.toml"
        override.write_text(override_text(args.out, end=end, accumulation=accumulation,
                                         cache_clear_interval=args.cache_clear_interval,
                                         with_evaluation=evaluate))
        command = [sys.executable, "-m", "torch.distributed.run",
                   f"--nproc_per_node={args.ranks}", "-m",
                   "scripts.bench.infra_eval_timing" if evaluate else "src.train.train"]
        for config in (Path("configs/base.toml"), stage_config_dir / f"hero-{stage}.toml",
                       Path("configs/hero.toml"), override):
            command += ["--config", str(config)]
        command += ["--run_name", name, *args.train_arg,
                    *getattr(args, f"{stage}_train_arg")]
        if checkpoint is not None:
            validate_checkpoint(checkpoint, max_steps=HORIZON, expected_step=start,
                                require_record=True, scheduler_count=2, use_ema=True,
                                world_size=args.ranks)
            command += ["--resume", str(checkpoint), "--resume_full", "--verify_resume_state"]
            if not branch:
                command += ["--reset_sampler"]
        case = dict(stage=stage, branch=branch, start=start, end=end,
                    accumulation=accumulation, command=command)
        report["cases"].append(case)
        save_report()
        case_env = dict(env, ARTFLOW_EVAL_TIMING_DIR=str(args.out / name))
        with (args.out / f"{name}.log").open("w") as log:
            result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s",
                                     f"{args.timeout}s", *command], env=case_env,
                                    stdout=log, stderr=subprocess.STDOUT)
        case["returncode"] = result.returncode
        if result.returncode:
            raise ValueError(f"{name}: training exited {result.returncode}")
        run = args.out / name
        saved = run / f"checkpoint_step_{end:06d}"
        validate_checkpoint(saved, max_steps=HORIZON, expected_step=end, require_record=True,
                            scheduler_count=2, use_ema=True, world_size=args.ranks)
        check_saved_state(saved, end)
        check_progress(run, start=start, end=end, ranks=args.ranks)
        if evaluate:
            expected_real = None
            if end == HORIZON:
                from datasets import load_from_disk
                from src.train.config import load_config
                evaluation_config = load_config(["configs/base.toml",
                    str(stage_config_dir / f"hero-{stage}.toml"), "configs/hero.toml",
                    str(override)])
                expected_real = len(load_from_disk(evaluation_config.eval.dataset_path))
            check_evaluation(run, end=end, ranks=args.ranks, expected_real=expected_real)
            case["evaluation_verified"] = True
        if checkpoint is not None:
            check_restoration(run, step=start, ranks=args.ranks)
        case["verified"] = True
        save_report()
        return saved

    try:
        previous = None
        for stage, start, end, accumulation in STAGES:
            accumulation = getattr(args, f"{stage}_accumulation") or accumulation
            previous = run_stage(stage, start, end, accumulation, checkpoint=previous)
            if stage == "256p":
                replay = run_stage(stage, 28, end, accumulation,
                                   checkpoint=args.out / stage / "checkpoint_step_000028",
                                   branch=True)
                report["replay"] = compare_checkpoints(previous, replay)
                check_replay_inputs(previous.parent, replay.parent, start=28, end=30,
                                    ranks=args.ranks)
                deterministic_files = ["scheduler.bin", "scheduler_1.bin"] + [
                    f"random_states_{rank}.pkl" for rank in range(args.ranks)]
                if any(not report["replay"]["files"][name]["exact"] for name in deterministic_files):
                    raise ValueError("resumed scheduler or RNG state differs")
                # V51 establishes that native attention backward is not bitwise
                # deterministic. Keep numerical differences visible, but require
                # exact live restoration, consumed inputs, schedulers and RNG.
                report["replay_contract_verified"] = True
        report["verified"] = True
    except Exception as exc:
        report["error"] = str(exc)
    save_report()
    print(json.dumps(report), flush=True)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
