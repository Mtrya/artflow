"""Run a sequential Stage-3 A/B using the same data/config and corrected timer.

Run from the repository root on an Inspire GPU job. Outputs include the exact
TOML override, each arm's stdout, and a machine-readable gate result.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MIX = {
    "d1": .150, "d2-wikiart": .191, "d2-museum": .009,
    "d3-human": .076, "d3-people": .074, "d4-vintage": .094,
    "d4-zimage": .022, "d4-megalith": .003, "d4-inat": .001,
    "d4-pd12m": .079, "d4-relaion": .301,
}
BUCKETS = [[64, 16], [128, 16], [192, 16], [256, 16], [384, 8],
           [512, 8], [768, 4], [1536, 2], [2048, 1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workroot", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    root = Path(args.workroot)
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    out = root / "runs" / "stage3" / args.name
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        PYTHONPATH=str(repo), HF_HOME=str(root / "cache/hf"),
        TORCH_HOME=str(root / "models/torch_home"),
        SWANLAB_LOG_DIR=str(out / "swanlog"), TOKENIZERS_PARALLELISM="false",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    auth = Path.home() / ".swanlab" / ".netrc"
    auth.parent.mkdir(exist_ok=True)
    shutil.copyfile(root / "cache/swanlab.netrc", auth)
    auth.chmod(0o600)

    # Prepare metadata once before either timed arm, using their exact tokenizer.
    from src.dataset.length_metadata import ensure_sidecar

    tokenizer = str(root / "models/Qwen3-0.6B")
    paths = [root / "precomputed_dataset" / f"{name}@256p" for name in MIX]
    resolution_ids = set()
    for path in paths:
        metadata = ensure_sidecar(str(path), tokenizer)
        resolution_ids.update(map(int, metadata.resolution_ids))
        print(f"metadata ready: {path.name}", flush=True)
    plan = json.dumps({str(r): BUCKETS for r in sorted(resolution_ids)})
    mix = " ".join(f"{path}:{weight}" for path, weight in zip(paths, MIX.values()))
    config = out / "gate.toml"
    config.write_text(
        f'[data]\nmix = {json.dumps(mix)}\nbucket_plan = {json.dumps(plan)}\n'
        f'[text_encoder]\npath = {json.dumps(tokenizer)}\n'
        f'[paths]\nvae = {json.dumps(str(root / "models/e2e-qwenimage-vae"))}\n'
        f'output_dir = {json.dumps(str(out))}\n'
        f'[train]\nmax_steps = {args.steps}\ncheckpoint_interval = {args.steps}\n'
        f'[eval]\ndataset_path = {json.dumps(str(root / "precomputed_dataset/light-eval@256p"))}\n'
        '[telemetry]\nswanlab_project = "artflow-stage3"\n'
    )
    launcher = [sys.executable, "-m", "src.train.train"]
    if args.gpus > 1:
        launcher = ["accelerate", "launch", "--multi_gpu", "--num_processes",
                    str(args.gpus), "--num_machines", "1", "--mixed_precision",
                    "bf16", "--dynamo_backend", "no", "-m", "src.train.train"]
    common = ["--config", "configs/base.toml", "--config", str(config)]
    baseline = [f"--no-{flag}" for flag in (
        "fast_caption_dropout", "fast_telemetry", "fast_text_slice",
        "attn_bias_hoist", "compile", "compile_blocks", "muon_batched_ns",
        "ddp_boundary_sync",
    )] + ["--text_encoder_exit_mode", "full_forward_slice"]
    summaries = {}
    samples = {}
    for arm, flags in (("baseline", baseline), ("fullstack", [])):
        command = launcher + common + ["--run_name", f"{args.name}-{arm}"] + flags
        print(json.dumps({"arm": arm, "command": command}), flush=True)
        with (out / f"{arm}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if "[throughput-summary]" in line:
                    fields = dict(re.findall(r"(\w+)=([\d.]+)", line))
                    summaries[arm] = fields
                if "[eval-loss@" in line:
                    samples.setdefault(arm, []).append(line.strip())
            if process.wait() != 0:
                raise SystemExit(f"{arm} failed with exit code {process.returncode}")
    ratio = (float(summaries["fullstack"]["samples_per_sec_steady"])
             / float(summaries["baseline"]["samples_per_sec_steady"]))
    matched = summaries["baseline"]["samples"] == summaries["fullstack"]["samples"]
    result = {"summaries": summaries, "steady_speedup": ratio,
              "matched_total_samples": matched, "throughput_pass": ratio >= 1.25 and matched,
              "eval_probes": samples,
              "note": "Throughput decision only; inspect eval probes for accuracy."}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("[gate-result] " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
