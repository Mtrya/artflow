"""DiT-only throughput ceiling: forward+backward sweeps of the training DiT.

Measures the transformer alone — no text encoder, no data path: dummy text
feeds the model directly. Sweeps over (txt_seq x latent side x micro_batch)
of forward+backward-only steps and reports per-GPU samples/s, ms/step, and
peak VRAM. txt_seq sweeps show what long captions cost inside the DiT
(single-stream concat attention); latent-side sweeps give the higher-
resolution ceilings (latent 32/80/112 = 256p/640p/896p).

The architecture is the training DiT (see ARCH below: hidden 1152, 24
single-stream layers with per-layer modulation, centered-grid RoPE, fused
conditioning, ~485M params). If fvcore is installed, forward FLOPs/sample at
latent 32 are measured once and anchor an MFU estimate: other latent sizes
extrapolate by token count (image tokens + text tokens), backward is counted
as ~2x forward, and the 4090 bf16 dense peak is ~330 TFLOPS — approximate,
and marked as such in the output.

Usage:
    python scripts/bench/transformer_ceiling.py --out ceiling256.json \
        --latent 32 --micro 8 16 32 --txt-seq 64 128 256 512
    python scripts/bench/transformer_ceiling.py --out ceiling-highres.json \
        --latent 80 112 --micro 8 --txt-seq 128
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.artflow import ArtFlow

PEAK_BF16_TFLOPS = 330.0  # RTX 4090 dense tensor-core, bf16

ARCH = dict(
    hidden_size=1152,
    num_heads=16,
    double_stream_depth=0,
    single_stream_depth=24,
    mlp_ratio=2.67,
    conditioning_scheme="fused",
    qkv_bias=True,
    double_stream_modulation="none",
    single_stream_modulation="layer",
    ffn_type="gated",
    rope_centered_grid=True,
    patch_size=2,
    in_channels=16,
    txt_in_features=1024,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--micro", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--txt-seq", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--latent", nargs="+", type=int, default=[32, 80, 112])
    ap.add_argument("--steps", type=int, default=25, help="measured steps")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model before measuring (single config only: "
        "graph compilation takes ~minutes per shape; pass one micro/txt-seq)",
    )
    ap.add_argument(
        "--compile-blocks",
        action="store_true",
        help="With --compile: compile each DiT block instead of the whole "
        "model (all 24 blocks share one graph per shape).",
    )
    ap.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    ap.add_argument(
        "--attn-backend",
        default="efficient",
        choices=["efficient", "cudnn", "efficient+cudnn"],
    )
    args = ap.parse_args()

    from src.models.dit_blocks import set_sdpa_backends

    if args.attn_backend == "cudnn":
        set_sdpa_backends(["CUDNN_ATTENTION", "MATH"])
    elif args.attn_backend == "efficient+cudnn":
        set_sdpa_backends(["CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"])
    else:
        set_sdpa_backends(["EFFICIENT_ATTENTION", "MATH"])

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True  # ceiling measurement: allow autotune

    model = ArtFlow(**ARCH).to(device="cuda", dtype=torch.bfloat16)
    if args.compile:
        print("compiling...", flush=True)
        t0 = time_monotonic()
        if args.compile_blocks:
            import importlib

            dynamo = importlib.import_module("torch._dynamo")
            dynamo.config.recompile_limit = 64
            for block in model.blocks:
                block.forward = torch.compile(
                    block.forward, mode=args.compile_mode, dynamic=False
                )
        else:
            model = torch.compile(model, mode=args.compile_mode)
        torch.cuda.synchronize()
        print(f"compile wrapper ready in {time_monotonic() - t0:.1f}s", flush=True)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.1f}M", flush=True)

    # Exact fwd FLOPs @ latent32 (one sample) for the MFU anchor.
    mfu_anchor = None
    try:
        from fvcore.nn import FlopCountAnalysis

        x = torch.randn(1, 16, 32, 32, dtype=torch.bfloat16, device="cuda")
        t = torch.rand(1, device="cuda")
        txt = torch.randn(1, 128, 1024, dtype=torch.bfloat16, device="cuda")
        pooled = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
        mask = torch.ones(1, 128, dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fl = FlopCountAnalysis(model, (x, t, txt, pooled, mask)).total()
        mfu_anchor = {"fwd_flops_sample_lat32_seq128": int(fl), "n_params": n_params}
        print(f"fvcore fwd FLOPs @lat32/seq128 = {fl/1e9:.2f} G", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"fvcore unavailable ({e}); MFU anchor skipped", flush=True)

    results = {}
    for lat in args.latent:
        img_tokens = (lat // 2) ** 2
        lat_res = results.setdefault(str(lat), {})
        for micro in args.micro:
            B = micro
            x = torch.randn(B, 16, lat, lat, dtype=torch.bfloat16, device="cuda")
            t = torch.rand(B, device="cuda")
            target_norm = torch.randn(B, 16, lat, lat, dtype=torch.bfloat16, device="cuda")
            micro_res = lat_res.setdefault(str(micro), {})
            for seq in args.txt_seq:
                txt = torch.randn(B, seq, 1024, dtype=torch.bfloat16, device="cuda")
                pooled = torch.randn(B, 1024, dtype=torch.bfloat16, device="cuda")
                mask = torch.ones(B, seq, dtype=torch.long, device="cuda")

                def step():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(x, t, txt=txt, txt_pooled=pooled, txt_mask=mask)
                        loss = ((out - target_norm) ** 2).mean()
                    loss.backward()
                    model.zero_grad(set_to_none=True)

                # A large (micro x txt_seq) corner can exceed 48GB once the
                # compiled graph has materialized its buffers; record the OOM
                # instead of losing the rest of the sweep.
                try:
                    for _ in range(args.warmup):
                        step()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time_monotonic()
                    for _ in range(args.steps):
                        step()
                    torch.cuda.synchronize()
                    dt = time_monotonic() - t0
                except torch.cuda.OutOfMemoryError:
                    micro_res[str(seq)] = {"error": "cuda_oom"}
                    print(f"lat={lat} micro={micro} seq={seq}: OOM (skipped)", flush=True)
                    del txt, pooled, mask
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                sps = args.steps * B / dt
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                entry = {
                    "samples_per_sec": round(sps, 2),
                    "ms_per_step": round(1000 * dt / args.steps, 2),
                    "peak_vram_gb": round(peak_gb, 2),
                    "img_tokens": img_tokens,
                }
                if mfu_anchor:
                    # extrapolate fwd flops by (img_tokens + txt_seq) ratio
                    fwd_fl = mfu_anchor["fwd_flops_sample_lat32_seq128"] * (
                        img_tokens + seq
                    ) / (16**2 + 128)
                    eff_tflops = sps * fwd_fl * 3.0 / 1e12  # fwd + ~2x bwd
                    entry["est_mfu_pct"] = round(100 * eff_tflops / PEAK_BF16_TFLOPS, 1)
                    entry["est_tflops"] = round(eff_tflops, 1)
                micro_res[str(seq)] = entry
                print(
                    f"lat={lat} micro={micro} seq={seq}: {entry}", flush=True
                )
                del txt, pooled, mask
                torch.cuda.empty_cache()

    out = {
        "arch": ARCH,
        "n_params": n_params,
        "mfu_anchor": mfu_anchor,
        "note": "est_mfu_pct extrapolates fwd FLOPs by token ratio from the "
        "lat32/seq128 fvcore anchor; bwd assumed 2x fwd; 4090 bf16 peak "
        "330 TFLOPS. samples/s = fwd+bwd only, no optimizer, no Qwen3.",
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {args.out}", flush=True)


def time_monotonic():
    import time

    return time.monotonic()


if __name__ == "__main__":
    main()
