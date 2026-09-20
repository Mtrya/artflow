"""One-off diagnostic: probe and grid a hero checkpoint with live vs EMA weights.

The training loop evaluates the EMA copy (probe at train.py's periodic
``eval_probe.evaluate(ema_model)`` and the fixed-prompt grids). With
ema_decay=0.9999 the EMA retains 37% of the random init at step 10k, so those
readings may describe EMA lag rather than the model actually being trained.
This script loads one checkpoint, rebuilds the model from the exact hero
config stack, and runs the standard eval-loss probe plus the standard
fixed-prompt grid twice: once with the live weights (``model.safetensors``)
and once with the EMA weights (``ema_weights.pt``).

Run on a single GPU from the repo snapshot root:

    python3 -m scripts.diag.live_probe \
        --config configs/base.toml --config /tmp/inputs.toml \
        --config configs/hero.toml --config /tmp/override.toml \
        --checkpoint $ARTFLOW_ROOT/runs/hero-256p/checkpoint_step_010000 \
        --out $ARTFLOW_ROOT/runs/hero-diag-liveprobe
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.eval_loss import EvalLossProbe
from src.evaluation.prompt_grid import load_prompt_plan, resolved_prompt_seed, sample_prompt_images
from src.evaluation.visualize import make_image_grid
from src.models.artflow import ArtFlow
from src.models.dit_blocks import set_real_rope
from src.pretrain.config import flatten, load_config
from src.utils.vae_codec import get_vae_stats


def build_model(args, device):
    model = ArtFlow(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        double_stream_depth=args.double_stream_depth,
        single_stream_depth=args.single_stream_depth,
        mlp_ratio=args.mlp_ratio,
        conditioning_scheme=args.conditioning_scheme,
        qkv_bias=args.qkv_bias,
        double_stream_modulation=args.double_stream_modulation,
        single_stream_modulation=args.single_stream_modulation,
        ffn_type=args.ffn_type,
        rope_centered_grid=args.rope_centered_grid,
        patch_size=2,
        in_channels=16,
        txt_in_features=1024,
    )
    set_real_rope(model, args.real_rope)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def save_grids(results, shapes, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    grids = {}
    for prompt, image in results:
        grids.setdefault((prompt.get("scene_id", ""), prompt["_bucket"]), []).append(
            (prompt, image)
        )
    for (scene, bucket_id), items in sorted(grids.items()):
        h_lat, w_lat = shapes[bucket_id]
        if scene:
            order = {("zh", "short"): 0, ("en", "short"): 1,
                     ("zh", "long"): 2, ("en", "long"): 3}
            items.sort(key=lambda item: order[(item[0]["lang"], item[0]["variant"])])
        images = torch.stack([image for _, image in items])
        name = f"grid_diag_{tag}_{scene or 'suite'}_bucket{bucket_id}_{h_lat * 8}x{w_lat * 8}.png"
        make_image_grid(images, save_path=os.path.join(out_dir, name),
                        normalize=True, value_range=(0, 1))
        print(f"saved {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ode_steps", type=int, default=50)
    cli = parser.parse_args()

    args = SimpleNamespace(**flatten(load_config(cli.config)))
    # The training launch passes --real_rope as a CLI flag outside the TOML.
    args.real_rope = True

    device = torch.device("cuda:0")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.text_encoder_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, local_files_only=True,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
    vae_mean, vae_std = get_vae_stats(args.vae_path, device=device)
    vae_mean = vae_mean.to(torch.bfloat16)
    vae_std = vae_std.to(torch.bfloat16)
    pooling = args.conditioning_scheme == "fused"

    probe = EvalLossProbe(
        args.eval_dataset_path, text_encoder, tokenizer,
        pooling=pooling, exit_layer=args.text_encoder_exit_layer,
        vae_mean=vae_mean, vae_std=vae_std,
        num_samples=args.eval_loss_samples, device=device,
    )
    print(f"probe ready: {len(probe.latents)} samples")

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    prompts, shapes = load_prompt_plan(args.prompts_file, args.eval_dataset_path)
    print(f"prompt suite: {len(prompts)} prompts, {len(shapes)} buckets")

    weights = {
        "live": load_file(os.path.join(cli.checkpoint, "model.safetensors")),
        "ema": torch.load(os.path.join(cli.checkpoint, "ema_weights.pt"),
                          map_location="cpu", weights_only=False),
    }
    summary = {}
    for tag, state in weights.items():
        model = build_model(args, device)
        model.load_state_dict(state)
        metrics = probe.evaluate(model)
        summary[tag] = metrics
        print(f"[probe:{tag}] " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        results = sample_prompt_images(
            model, vae, vae_mean, vae_std, text_encoder, tokenizer,
            prompts, shapes, batch_size=args.eval_batch_size,
            ode_steps=cli.ode_steps, pooling=pooling, device=device,
            exit_layer=args.text_encoder_exit_layer,
            amp_context=lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        )
        save_grids(results, shapes, cli.out, tag)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(cli.out, "probe_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("done")


if __name__ == "__main__":
    main()
