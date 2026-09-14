#!/usr/bin/env python3
"""Write the three hero stage configs (mix + bucket plan) to $W/configs/.

The mix weights are the normalized per-resolution weights the user fixed in
notes/hero_recipe.md (2026-09-14); sharded datasets (relaion p0-p4 at
640p/896p, people a/b at 640p) split their logical weight proportional to
row counts, matching gen_hero_bucket_plans.py exactly.
"""
import os

import numpy as np

from scripts.bench.gen_hero_bucket_plans import MIX, OUTDIR, W, dataset_dirs

TEMPLATE = """# Hero {stage} stage: dataset mix (user-fixed weights, normalized per shard)
# and the 20-bucket compute-optimal plan solved from the 533M calibration.
# Layer after configs/base.toml; model shape and schedule come from the rung
# config layered on top.

[data]
mix = "{mix}"
bucket_plan = "{plan_dir}/hero-{stage}-k20.json"

[text_encoder]
path = "{W}/models/Qwen3-0.6B"

[eval]
dataset_path = "{W}/precomputed_dataset/light-eval@{stage}"
prompts_file = "assets/eval/hero_monitor_v1.jsonl"
ode_steps = 50

[paths]
vae = "{W}/models/e2e-qwenimage-vae"
output_dir = "{W}/runs"
"""


def main():
    outdir = f"{W}/configs"
    os.makedirs(outdir, exist_ok=True)
    for stage, weights in MIX.items():
        parts = []
        for name, w in weights.items():
            if w <= 0:
                continue
            dirs = dataset_dirs(name, stage)
            rows = []
            for d in dirs:
                n = np.load(f"{W}/precomputed_dataset/{d}/length_metadata.npz")
                rows.append(len(n["caption_offsets"]) - 1)
            total = sum(rows)
            for d, r in zip(dirs, rows):
                parts.append(f"{W}/precomputed_dataset/{d}:{w * r / total:.6f}")
        text = TEMPLATE.format(stage=stage, mix=" ".join(parts), W=W, plan_dir=OUTDIR)
        path = f"{outdir}/hero-{stage}.toml"
        with open(path, "w") as f:
            f.write(text)
        print("wrote", path, f"({len(parts)} datasets)")


if __name__ == "__main__":
    main()
