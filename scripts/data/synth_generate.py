"""Generate the synthetic dress set from its prompt grid.

Each grid row is used twice: it is the prompt the generator is given, and it is
the caption the finished picture is trained on.  Nothing here rewrites the
text, so a picture and its label cannot drift apart - changing the label means
changing the request.

Generating the full grid takes hours, so the work is split by ``--shard``: every
shard of a run walks the same grid in the same order and takes every n-th row,
which keeps a re-run of one shard reproducible and lets several shards run on
different GPUs or machines without coordinating.

CLI:
    python -m scripts.data.synth_generate --prompts grid.jsonl --out-dir images \\
        --model models/Z-Image-Turbo --recipe z-image --limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

# Inference recipe per generator.  Z-Image-Turbo is guidance-distilled, so the
# guidance scale is zero and a handful of steps is the whole sampler; Qwen-Image
# is a full 50-step model with classifier-free guidance.
# Qwen-Image reads its guidance from `true_cfg_scale` and only applies it when
# a negative prompt is present, so both travel together; `guidance` stays None
# and the pipeline's own default is used.
#
# When `resolutions` is set the requested frame is replaced by the closest of
# the sizes the model was trained on, matched by aspect ratio; a model asked for
# a frame it never saw drifts in composition.
RECIPES: Dict[str, Dict] = {
    "z-image": {"pipeline": "ZImagePipeline", "steps": 9, "guidance": 0.0, "extra": {}},
    "qwen-image": {"pipeline": "QwenImagePipeline", "steps": 50, "guidance": None,
                   "extra": {"true_cfg_scale": 4.0, "negative_prompt": " "}},
    # Same weights with the distilled sampler applied (see --lora): eight steps
    # and no classifier-free guidance, which is what the distillation expects.
    "qwen-image-lightning": {"pipeline": "QwenImagePipeline", "steps": 8,
                             "guidance": 1.0, "extra": {}},
    "ernie-image-turbo": {"pipeline": "ErnieImagePipeline", "steps": 8,
                          "guidance": 1.0, "extra": {"use_pe": False},
                          "resolutions": [(1024, 1024), (1264, 848), (848, 1264),
                                          (1376, 768), (768, 1376)]},
}


def read_prompts(path: str) -> List[Dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select(rows: List[Dict], shard: int, shards: int, offset: int, limit: int) -> List[Dict]:
    """The slice of the grid one worker is responsible for.

    ``offset`` drops that many rows from the head of the grid, so a run can be
    pointed at a different stretch of it; ``shard`` then takes every
    ``shards``-th row of what is left.  Both are needed - taking the stride from
    the offset alone hands every worker the same rows, which is silent: the
    workers all report progress and between them they produce one worker's
    worth of pictures.
    """
    rows = rows[offset:]
    if shards > 1:
        rows = rows[shard::shards]
    return rows[:limit] if limit else rows


def snap_resolution(width: int, height: int, supported: List) -> tuple:
    """Closest trained frame to the requested one, compared by aspect ratio."""
    target = width / height
    best = min(supported, key=lambda wh: abs(wh[0] / wh[1] - target))
    return best


def build_pipeline(args, recipe, torch):
    """The pipeline for a recipe, weights where the recipe says they are."""
    import diffusers

    pipeline_class = getattr(diffusers, recipe["pipeline"])
    pipe = pipeline_class.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    if args.lora:
        pipe.load_lora_weights(os.path.dirname(args.lora),
                               weight_name=os.path.basename(args.lora))
    if args.offload:
        pipe.enable_model_cpu_offload()
    elif args.text_encoder_cpu:
        pipe.text_encoder.to("cpu")
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True, help="prompt grid JSONL")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True, help="local directory of the pipeline")
    parser.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    parser.add_argument("--only-model", default=None,
                        help="render only the grid rows assigned to this recipe; the "
                             "grid carries a `model` column when several generators "
                             "share the work")
    parser.add_argument("--steps", type=int, default=None, help="override the recipe")
    parser.add_argument("--guidance", type=float, default=None, help="override the recipe")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--lora", default=None,
                        help="LoRA weights to apply to the transformer, e.g. a distilled "
                             "sampler that replaces the long schedule")
    parser.add_argument("--offload", action="store_true",
                        help="keep the pipeline in host memory and move each module to "
                             "the GPU as it is used; needed when the weights do not fit")
    parser.add_argument("--text-encoder-cpu", action="store_true",
                        help="run the text encoder in host memory and keep the rest on "
                             "the GPU; the encoder runs once per prompt, so this is much "
                             "cheaper than moving every module in and out per step")
    args = parser.parse_args()

    import torch

    recipe = dict(RECIPES[args.recipe])
    if args.steps is not None:
        recipe["steps"] = args.steps
    if args.guidance is not None:
        recipe["guidance"] = args.guidance

    pipe = build_pipeline(args, recipe, torch)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(args.prompts)
    if args.only_model:
        prompts = [row for row in prompts if row.get("model") == args.only_model]
    rows = select(prompts, args.shard, args.shards, args.offset, args.limit)
    print(f"{len(rows)} prompts for shard {args.shard}/{args.shards} -> {out_dir}", flush=True)

    started = time.time()
    done = 0
    with (out_dir / f"generated_shard{args.shard}.jsonl").open("w", encoding="utf-8") as sink:
        for row in rows:
            image_path = out_dir / f"{row['prompt_id']}.jpg"
            width, height = row["width"], row["height"]
            if recipe.get("resolutions"):
                # Each model was trained on a short list of frames; asking for
                # one it never saw shifts the composition, so the nearest
                # trained frame is used instead.
                width, height = snap_resolution(width, height, recipe["resolutions"])
            record = {
                "prompt_id": row["prompt_id"],
                "image_id": row["prompt_id"],
                "text": row["text"],
                "language": row["language"],
                "family": row["family"],
                "subject": row["subject"],
                "subject_group": row["subject_group"],
                "aspect": row["aspect"],
                "width": width,
                "height": height,
                "recipe": args.recipe,
                "steps": recipe["steps"],
                "guidance": recipe["guidance"],
                "path": str(image_path),
            }
            if image_path.exists():
                record["seconds"] = None
                record["skipped"] = "already generated"
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            generator = torch.Generator(device="cuda").manual_seed(args.seed + done)
            call_started = time.time()
            kwargs = {
                "height": height,
                "width": width,
                "num_inference_steps": recipe["steps"],
                "generator": generator,
                **recipe["extra"],
            }
            if recipe["guidance"] is not None:
                kwargs["guidance_scale"] = recipe["guidance"]
            if args.text_encoder_cpu:
                # Encoding once per prompt in host memory is far cheaper than
                # moving the encoder in and out for every denoising step.
                embeds, mask = pipe.encode_prompt(
                    prompt=row["text"], device=torch.device("cpu"),
                    num_images_per_prompt=1, max_sequence_length=1024)
                kwargs["prompt_embeds"] = embeds
                kwargs["prompt_embeds_mask"] = mask
            else:
                kwargs["prompt"] = row["text"]
            image = pipe(**kwargs).images[0]
            record["seconds"] = round(time.time() - call_started, 2)
            image.save(image_path, quality=args.quality)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            done += 1
            if done % 5 == 0:
                rate = done / (time.time() - started)
                print(f"  {done}/{len(rows)}  {rate:.2f} img/s  "
                      f"{record['seconds']}s for the last one", flush=True)
    print(f"done: {done} images in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
