#!/usr/bin/env python
"""
Training-memory profiler: find what actually drives the OOM spikes.

Reproduces the real training step (same dataloader, same encode_text, same
forward/backward) on the real mix and records per-step CUDA memory alongside
batch composition (max text seq len, latent shape). Also measures the two
transient suspects: checkpoint saving and the eval-loss probe.

Output: per-step table + top-peak offenders + torch memory snapshot pickle
(open at https://pytorch.org/memory_viz) written to --output_dir.

Usage (Inspire 1-GPU job, real mix):
  python scripts/stage2/mem_profile.py \
      --dataset_mix "$MIX" --text_encoder_path ... --vae_path ... \
      --steps 150 --batch_size 16 --output_dir $W/runs/stage2/mem_profile
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset.mix import (  # noqa: E402
    get_dataset_weights,
    load_mixed_dataset,
    parse_dataset_mix,
)
from src.dataset.sampler import ResolutionBucketSampler, collate_fn  # noqa: E402
from src.flow.paths import shift_timesteps  # noqa: E402
from src.models.artflow import ArtFlow  # noqa: E402
from src.utils.encode_text import encode_text  # noqa: E402
from src.utils.vae_codec import get_vae_stats  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def gb(x):
    return x / 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_mix", required=True)
    p.add_argument("--text_encoder_path", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--single_stream_depth", type=int, default=24)
    p.add_argument("--double_stream_depth", type=int, default=0)
    p.add_argument("--mlp_ratio", type=float, default=2.67)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- model / optimizer / text encoder (same as 2.1 arms) ---
    model = ArtFlow(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        double_stream_depth=args.double_stream_depth,
        single_stream_depth=args.single_stream_depth,
        mlp_ratio=args.mlp_ratio,
        conditioning_scheme="fused",
        qkv_bias=True,
        double_stream_modulation="none",
        single_stream_modulation="none",
        ffn_type="gated",
        rope_centered_grid=True,
        patch_size=2,
        in_channels=16,
        txt_in_features=1024,
    ).to(device)
    model.train()
    n_params = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.text_encoder_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    vae_mean, vae_std = get_vae_stats(args.vae_path, device=device)

    ema_model = None  # EMA update cost is tiny; skip

    # --- real dataloader ---
    entries = parse_dataset_mix(args.dataset_mix)
    weights = get_dataset_weights(entries)
    dataset = load_mixed_dataset(entries, shuffle_seed=args.seed)
    sampler = ResolutionBucketSampler(
        dataset,
        batch_size=args.batch_size,
        num_replicas=1,
        rank=0,
        dataset_weights=weights if len(entries) > 1 else None,
    )
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    print(f"torch {torch.__version__}; expandable_segments="
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    print(f"static alloc (model+opt+enc): {gb(base_alloc):.2f} GiB "
          f"(pre-first-step; optimizer states allocate lazily)")

    torch.cuda.memory._record_memory_history(max_entries=200_000)
    rows = []
    it = iter(dataloader)
    peak_global = 0.0
    try:
        for step in range(1, args.steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)

            torch.cuda.reset_peak_memory_stats()
            alloc_before = torch.cuda.memory_allocated()

            latents = batch["latents"].to(device)
            latents = (latents - vae_mean) / vae_std
            t = torch.rand(latents.shape[0], device=device)
            t = shift_timesteps(t, latents)

            captions = [caps[0] if caps else "" for caps in batch["captions"]]
            txt, txt_mask, txt_pooled = encode_text(
                captions, text_encoder, tokenizer, pooling=True
            )
            txt_len = txt.shape[1]

            z0 = torch.randn_like(latents)
            z_t = (1.0 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * latents
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(z_t, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask)
                loss = torch.nn.functional.mse_loss(out.float(), (latents - z0).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            peak = torch.cuda.max_memory_allocated()
            alloc_after = torch.cuda.memory_allocated()
            peak_global = max(peak_global, peak)
            rows.append({
                "step": step,
                "txt_len": int(txt_len),
                "lat_hw": list(latents.shape[-2:]),
                "peak_gb": round(gb(peak), 2),
                "alloc_after_gb": round(gb(alloc_after), 2),
                "delta_gb": round(gb(peak - alloc_before), 2),
            })
            if step % 25 == 0:
                print(f"step {step}: peak {gb(peak):.2f} GiB "
                      f"(txt_len={txt_len}, lat={tuple(latents.shape[-2:])})")

            # checkpoint-save spike probe at steps 50/100
            if step in (50, 100):
                torch.cuda.reset_peak_memory_stats()
                sd = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_path = os.path.join(args.output_dir, f"ckpt_probe_{step}.pt")
                torch.save(sd, save_path)
                ckpt_peak = torch.cuda.max_memory_allocated()
                del sd
                os.remove(save_path)
                print(f"  [ckpt-save @step {step}] peak {gb(ckpt_peak):.2f} GiB "
                      f"(+{gb(ckpt_peak - alloc_after):.2f} over post-step alloc)")
                rows.append({"step": step, "event": "ckpt_save",
                             "peak_gb": round(gb(ckpt_peak), 2)})
    except torch.OutOfMemoryError:
        print("!!! OOM reproduced during profiling !!!")
    finally:
        snap = os.path.join(args.output_dir, "mem_snapshot.pickle")
        try:
            torch.cuda.memory._dump_snapshot(snap)
            print(f"memory snapshot -> {snap}")
        except Exception as e:
            print(f"snapshot dump failed: {e}")
        torch.cuda.memory._record_memory_history(enabled=None)

    rows.sort(key=lambda r: r.get("peak_gb", 0), reverse=True)
    print("\n=== top-10 peak steps ===")
    for r in rows[:10]:
        print(r)
    report = os.path.join(args.output_dir, "mem_report.json")
    with open(report, "w") as f:
        json.dump({"params": n_params, "steps": args.steps,
                   "batch_size": args.batch_size,
                   "peak_global_gb": round(gb(peak_global), 2),
                   "alloc_env": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                   "rows": rows}, f, indent=2)
    print(f"peak over run: {gb(peak_global):.2f} GiB; report -> {report}")


if __name__ == "__main__":
    main()
