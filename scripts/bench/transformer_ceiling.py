"""DiT-only throughput and memory ceiling for the training transformer.

Measures the transformer alone — no text encoder, no data path, no optimizer:
dummy text feeds the model directly.  Every (latent shape x text length x
micro-batch) point runs a forward+backward loop and records the peak allocated
memory and the steady-state milliseconds per step.  Those points are the
calibration input for the bucket-plan sizer
(``scripts/bench/plan_buckets.py``), which fits its memory and time models on
them, so the sweep should bracket the shapes the plan will actually serve.

The architecture is selected on the command line and defaults to the training
DiT (1152 wide, 1 double-stream + 24 single-stream blocks, ~533M parameters).
``--fast-attn`` measures the hoisted-RoPE / hoisted-attention-bias forward that
the training loop uses by default.

Latents are given as ``HxW`` (the trainer's shape log prints ``latent=HxW``) or
as a single number for a square latent, so non-square resolution buckets can be
measured too.  Text lengths sweep the caption side of the padded sequence: the
image tokens and the text tokens share the single-stream attention, so a long
caption costs what its sequence length costs.

If fvcore is installed, forward FLOPs/sample at the anchor shape are measured
once and anchor an MFU estimate: other points extrapolate by token count (image
tokens + text tokens), backward is counted as ~2x forward, and the 4090 bf16
dense peak is ~330 TFLOPS — approximate, and marked as such in the output.

Usage:
    python scripts/bench/transformer_ceiling.py --out ceiling256.json \
        --latent 32 --micro 8 16 32 --txt-seq 64 128 256 512
    python scripts/bench/transformer_ceiling.py --out ceiling640.json \
        --latent 80x80 106x60 92x70 --micro 4 8 16 --txt-seq 128 256 \
        --fast-attn
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.artflow import ArtFlow

PEAK_BF16_TFLOPS = 330.0  # RTX 4090 dense tensor-core, bf16

# Defaults are the training DiT: 1152 wide, one double-stream block followed by
# 24 single-stream blocks, ~533M parameters.
ARCH_DEFAULTS = dict(
    hidden_size=1152,
    num_heads=16,
    double_stream_depth=1,
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


def parse_latent(spec: str) -> tuple:
    """``"80"`` or ``"80x80"`` -> ``(height, width)``; ``"70x92"`` -> (70, 92).

    The order matches the trainer's shape log (``latent=HxW``).  Image-token
    count is the product of the two patch grids either way, so only the tensor
    the sweep allocates depends on the order.
    """
    text = str(spec).strip().lower()
    if "x" in text:
        height, _, width = text.partition("x")
    else:
        height = width = text
    try:
        hw = (int(height), int(width))
    except ValueError:
        raise ValueError(f"--latent {spec!r} is not 'H' or 'HxW'") from None
    if hw[0] < 1 or hw[1] < 1:
        raise ValueError(f"--latent {spec!r} must be positive")
    return hw


def latent_spec(hw: tuple) -> str:
    """The sweep's key for a latent: ``"80x80"``."""
    return f"{hw[0]}x{hw[1]}"


def image_tokens(hw: tuple, patch_size: int) -> int:
    """Patch-grid tokens of a latent, the image half of the attention sequence."""
    if hw[0] % patch_size or hw[1] % patch_size:
        raise ValueError(
            f"latent {latent_spec(hw)} is not divisible by patch size {patch_size}"
        )
    return (hw[0] // patch_size) * (hw[1] // patch_size)


def build_arch(args: argparse.Namespace) -> dict:
    arch = {
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "double_stream_depth": args.double_stream_depth,
        "single_stream_depth": args.single_stream_depth,
        "mlp_ratio": args.mlp_ratio,
        "conditioning_scheme": args.conditioning_scheme,
        "qkv_bias": args.qkv_bias,
        "double_stream_modulation": args.double_stream_modulation,
        "single_stream_modulation": args.single_stream_modulation,
        "ffn_type": args.ffn_type,
        "rope_centered_grid": args.rope_centered_grid,
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
        "txt_in_features": args.txt_in_features,
    }
    if arch["hidden_size"] % arch["num_heads"]:
        raise ValueError("--hidden-size must be divisible by --num-heads")
    if arch["double_stream_depth"] + arch["single_stream_depth"] < 1:
        raise ValueError("the model needs at least one block")
    return arch


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument(
        "--latent", nargs="+", default=["32", "80", "112"], metavar="H[xW]",
        help="latent shapes to sweep: 'H' for a square latent or 'HxW' for a "
             "non-square one (default 32 80 112, i.e. 256p/640p/896p squares)")
    ap.add_argument("--micro", nargs="+", type=int, default=[8, 16],
                    help="micro-batch sizes to sweep")
    ap.add_argument("--txt-seq", nargs="+", type=int, default=[64, 128, 256, 512],
                    help="dummy text lengths to sweep")
    ap.add_argument("--steps", type=int, default=25, help="measured steps")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--fast-attn",
        action="store_true",
        help="call the model with fast_attn=True: the hoisted-RoPE and "
             "hoisted-attention-bias path the training loop uses by default, "
             "which changes kernel shapes and memory, so measure it when the "
             "numbers feed a training plan",
    )
    arch = ap.add_argument_group("architecture (defaults: the training DiT)")
    arch.add_argument("--hidden-size", type=int, default=ARCH_DEFAULTS["hidden_size"])
    arch.add_argument("--num-heads", type=int, default=ARCH_DEFAULTS["num_heads"])
    arch.add_argument("--double-stream-depth", type=int,
                      default=ARCH_DEFAULTS["double_stream_depth"])
    arch.add_argument("--single-stream-depth", type=int,
                      default=ARCH_DEFAULTS["single_stream_depth"])
    arch.add_argument("--mlp-ratio", type=float, default=ARCH_DEFAULTS["mlp_ratio"])
    arch.add_argument("--conditioning-scheme", default=ARCH_DEFAULTS["conditioning_scheme"],
                      choices=["pure", "fused"])
    arch.add_argument("--double-stream-modulation",
                      default=ARCH_DEFAULTS["double_stream_modulation"],
                      choices=["none", "stream", "layer", "all"])
    arch.add_argument("--single-stream-modulation",
                      default=ARCH_DEFAULTS["single_stream_modulation"],
                      choices=["none", "stream", "layer", "all"])
    arch.add_argument("--ffn-type", default=ARCH_DEFAULTS["ffn_type"],
                      choices=["gated", "standard"])
    arch.add_argument("--no-qkv-bias", dest="qkv_bias", action="store_false",
                      help="drop the QKV projection bias")
    arch.add_argument("--no-rope-centered-grid", dest="rope_centered_grid",
                      action="store_false",
                      help="use the uncentered RoPE grid")
    arch.add_argument("--patch-size", type=int, default=ARCH_DEFAULTS["patch_size"])
    arch.add_argument("--in-channels", type=int, default=ARCH_DEFAULTS["in_channels"])
    arch.add_argument("--txt-in-features", type=int,
                      default=ARCH_DEFAULTS["txt_in_features"])
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
        "model (all blocks share one graph per shape).",
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
    return ap.parse_args(argv)


def main():
    args = parse_args()
    arch = build_arch(args)
    latents = [parse_latent(spec) for spec in args.latent]

    from src.models.dit_blocks import set_sdpa_backends

    if args.attn_backend == "cudnn":
        set_sdpa_backends(["CUDNN_ATTENTION", "MATH"])
    elif args.attn_backend == "efficient+cudnn":
        set_sdpa_backends(["CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"])
    else:
        set_sdpa_backends(["EFFICIENT_ATTENTION", "MATH"])

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True  # ceiling measurement: allow autotune

    model = ArtFlow(**arch).to(device="cuda", dtype=torch.bfloat16)
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
    print(f"params={n_params/1e6:.1f}M fast_attn={args.fast_attn}", flush=True)

    # Forward FLOPs at the anchor shape (one sample) for the MFU estimate.
    anchor_latent, anchor_seq = 32, 128
    mfu_anchor = None
    try:
        from fvcore.nn import FlopCountAnalysis

        x = torch.randn(1, arch["in_channels"], anchor_latent, anchor_latent,
                        dtype=torch.bfloat16, device="cuda")
        t = torch.rand(1, device="cuda")
        txt = torch.randn(1, anchor_seq, arch["txt_in_features"],
                          dtype=torch.bfloat16, device="cuda")
        pooled = torch.randn(1, arch["txt_in_features"], dtype=torch.bfloat16,
                             device="cuda")
        mask = torch.ones(1, anchor_seq, dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fl = FlopCountAnalysis(model, (x, t, txt, pooled, mask)).total()
        anchor_tokens = image_tokens((anchor_latent, anchor_latent),
                                     arch["patch_size"]) + anchor_seq
        mfu_anchor = {
            "fwd_flops_sample": int(fl),
            "latent": latent_spec((anchor_latent, anchor_latent)),
            "txt_seq": anchor_seq,
            "tokens": anchor_tokens,
            "n_params": n_params,
        }
        print(f"fvcore fwd FLOPs @lat{anchor_latent}/seq{anchor_seq} = "
              f"{fl/1e9:.2f} G", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"fvcore unavailable ({e}); MFU anchor skipped", flush=True)

    results = {}
    for hw in latents:
        spec = latent_spec(hw)
        img_tokens = image_tokens(hw, arch["patch_size"])
        lat_res = results.setdefault(spec, {})
        for micro in args.micro:
            B = micro
            x = torch.randn(B, arch["in_channels"], hw[0], hw[1],
                            dtype=torch.bfloat16, device="cuda")
            t = torch.rand(B, device="cuda")
            target_norm = torch.randn(B, arch["in_channels"], hw[0], hw[1],
                                      dtype=torch.bfloat16, device="cuda")
            for seq in args.txt_seq:
                txt = torch.randn(B, seq, arch["txt_in_features"],
                                  dtype=torch.bfloat16, device="cuda")
                pooled = torch.randn(B, arch["txt_in_features"],
                                     dtype=torch.bfloat16, device="cuda")
                mask = torch.ones(B, seq, dtype=torch.long, device="cuda")

                def step():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(x, t, txt=txt, txt_pooled=pooled,
                                    txt_mask=mask, fast_attn=args.fast_attn)
                        loss = ((out - target_norm) ** 2).mean()
                    loss.backward()
                    model.zero_grad(set_to_none=True)

                # Every config starts from a clean peak counter, so the recorded
                # number is this shape's own high-water mark (warm-up included:
                # the first iteration is where allocator and kernel autotuning
                # peaks land).
                torch.cuda.reset_peak_memory_stats()
                # A large (micro x txt_seq) corner can exceed 48GB once the
                # compiled graph has materialized its buffers; record the OOM
                # instead of losing the rest of the sweep.
                try:
                    for _ in range(args.warmup):
                        step()
                    torch.cuda.synchronize()
                    t0 = time_monotonic()
                    for _ in range(args.steps):
                        step()
                    torch.cuda.synchronize()
                    dt = time_monotonic() - t0
                except torch.cuda.OutOfMemoryError:
                    lat_res.setdefault(str(seq), {})[str(micro)] = {
                        "error": "cuda_oom",
                        "latent_hw": [hw[0], hw[1]],
                        "img_tokens": img_tokens,
                        "txt_seq": seq,
                        "micro_batch": B,
                    }
                    print(f"lat={spec} micro={micro} seq={seq}: OOM (skipped)", flush=True)
                    del txt, pooled, mask
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                sps = args.steps * B / dt
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                entry = {
                    "peak_mem_gb": round(peak_gb, 2),
                    "ms_per_step": round(1000 * dt / args.steps, 2),
                    "samples_per_sec": round(sps, 2),
                    "latent_hw": [hw[0], hw[1]],
                    "img_tokens": img_tokens,
                    "txt_seq": seq,
                    "micro_batch": B,
                }
                if mfu_anchor:
                    # extrapolate fwd flops by (img_tokens + txt_seq) ratio
                    tokens = img_tokens + seq
                    fwd_fl = mfu_anchor["fwd_flops_sample"] * tokens / mfu_anchor["tokens"]
                    eff_tflops = sps * fwd_fl * 3.0 / 1e12  # fwd + ~2x bwd
                    entry["est_mfu_pct"] = round(100 * eff_tflops / PEAK_BF16_TFLOPS, 1)
                    entry["est_tflops"] = round(eff_tflops, 1)
                lat_res.setdefault(str(seq), {})[str(micro)] = entry
                print(
                    f"lat={spec} micro={micro} seq={seq}: {entry}", flush=True
                )
                del txt, pooled, mask
                torch.cuda.empty_cache()

    out = {
        "arch": arch,
        "fast_attn": bool(args.fast_attn),
        "attn_backend": args.attn_backend,
        "compile": bool(args.compile),
        "n_params": n_params,
        "steps": args.steps,
        "warmup": args.warmup,
        "mfu_anchor": mfu_anchor,
        "note": "results is nested latent -> txt_seq -> micro_batch and every "
        "leaf carries its own latency/txt_seq/micro_batch/img_tokens, so a "
        "consumer can read the sweep without knowing the nesting. peak_mem_gb "
        "is torch.cuda.max_memory_allocated for that config, reset before its "
        "warm-up; ms_per_step is the steady-state loop. fwd+bwd only: no "
        "optimizer, no EMA, no text encoder, so this is the DiT floor of a "
        "training step's memory and time. est_mfu_pct extrapolates fwd FLOPs "
        "by token ratio from the fvcore anchor; bwd assumed 2x fwd; 4090 bf16 "
        "peak 330 TFLOPS.",
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
