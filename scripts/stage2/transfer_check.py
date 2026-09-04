#!/usr/bin/env python
"""
256p -> 640p (or arbitrary scale) transfer check for the 2.1 RoPE gate.

Loads a trained checkpoint (EMA weights) and samples the fixed prompt suite at
scaled-up latent shapes with the same per-prompt seeds used by training-time
grids, so the output is directly comparable to the 256p grids. Qualitative:
the grids get a human look for structure breakdown / artifacts.

Usage:
  python scripts/stage2/transfer_check.py \
      --ckpt <run_dir>/checkpoint_step_006000/ema_weights.pt \
      --text_encoder_path <...> --vae_path <...> \
      --eval_dataset_path <light-eval@256p> --scale 2.5 \
      --out_dir transfer_out/rope-new-640p [--rope_centered]
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.prompt_grid import (  # noqa: E402
    assign_bucket,
    bucket_shapes_from_dataset,
    load_prompt_suite,
    prompt_seed,
)
from src.evaluation.visualize import make_image_grid  # noqa: E402
from src.flow.solvers import sample_ode  # noqa: E402
from src.models.artflow import ArtFlow  # noqa: E402
from src.utils.encode_text import encode_text  # noqa: E402
from src.utils.vae_codec import get_vae_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to ema_weights.pt")
    p.add_argument("--text_encoder_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--eval_dataset_path", required=True,
                   help="256p eval dataset; bucket shapes are read here then scaled")
    p.add_argument("--prompts_file", default="assets/eval/prompts_v1.jsonl")
    p.add_argument("--scale", type=float, default=2.5, help="latent scale (256p->640p = 2.5)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ode_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rope_centered", action="store_true",
                   help="override config inference for centered-grid RoPE checkpoints")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {"rope_centered_grid": True} if args.rope_centered else {}
    model = ArtFlow.from_single_file(args.ckpt, **overrides).to(device).eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.text_encoder_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    vae_mean, vae_std = get_vae_stats(args.vae_path, device=device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    # Base shapes from the 256p eval set, scaled to the target resolution.
    # Latent dims must stay divisible by patch_size=2 (strided patch conv),
    # so snap to even after scaling.
    base_shapes = bucket_shapes_from_dataset(args.eval_dataset_path)

    def _snap(x: float) -> int:
        return max(8, int(round(x / 2)) * 2)

    shapes: Dict[int, Tuple[int, int]] = {
        bid: (_snap(h * args.scale), _snap(w * args.scale))
        for bid, (h, w) in base_shapes.items()
    }
    print("Scaled bucket shapes:",
          {b: f"{h * 8}x{w * 8}px" for b, (h, w) in shapes.items()})

    prompts = load_prompt_suite(args.prompts_file)
    for prompt in prompts:
        prompt["_bucket"] = assign_bucket(prompt, shapes)

    by_bucket: Dict[int, List[dict]] = {}
    for prompt in prompts:
        by_bucket.setdefault(prompt["_bucket"], []).append(prompt)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        for bid, plist in sorted(by_bucket.items()):
            h_lat, w_lat = shapes[bid]
            images = []
            for start in range(0, len(plist), args.batch_size):
                chunk = plist[start : start + args.batch_size]
                txt, txt_mask, txt_pooled = encode_text(
                    [c["text"] for c in chunk], text_encoder, tokenizer,
                    pooling=True,
                )
                noise = torch.stack(
                    [
                        torch.randn(
                            (16, h_lat, w_lat),
                            generator=torch.Generator(device="cpu").manual_seed(
                                prompt_seed(c["id"])
                            ),
                        )
                        for c in chunk
                    ]
                ).to(device, torch.bfloat16)

                def model_fn(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask):
                    if isinstance(t, float):
                        t = torch.tensor(t, device=x.device).expand(x.shape[0])
                    return model(x, t, txt, txt_pooled, txt_mask)

                samples = sample_ode(model_fn, noise, steps=args.ode_steps,
                                     t_start=0.0, t_end=1.0)
                samples = samples.to(torch.bfloat16) * vae_std + vae_mean
                imgs = vae.decode(samples.unsqueeze(2)).sample.squeeze(2)
                imgs = torch.clamp((imgs + 1) / 2, 0, 1).cpu().float()
                images.extend(list(imgs))
                print(f"  bucket {bid}: {len(images)}/{len(plist)} done")

            grid_path = os.path.join(
                args.out_dir, f"transfer_bucket{bid}_{h_lat * 8}x{w_lat * 8}.png"
            )
            make_image_grid(torch.stack(images), save_path=grid_path,
                            normalize=True, value_range=(0, 1))
            print(f"saved {grid_path}")

    print("Transfer check done.")


if __name__ == "__main__":
    main()
