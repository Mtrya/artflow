"""One-off MFU anchors for the hero budget report.

Builds the training DiT (1152/16, 1 double + 24 single stream, ~533M) and
times forward+backward at three bucket shapes representative of the hero
plans, counting FLOPs with torch.profiler (with_flops=True). Prints TFLOPS
achieved per shape so the run's step times can be compared against the
4090's bf16 dense peak (~165 TFLOPS).

Usage: python scripts/bench/flop_anchor.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.artflow import ArtFlow

ARCH = dict(
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

# (name, batch, latent HxW, padded text length) — mid-bucket anchors per stage
# Eager (uncompiled) autograd keeps every intermediate, so these probe
# batches are far below the training plan's; FLOPs scale linearly in batch.
SHAPES = [
    ("256p", 8, (32, 32), 106),
    ("640p", 4, (80, 80), 106),
    ("896p", 3, (112, 112), 106),
]


def run_shape(name, batch, hw, txt_len, model, device):
    # bf16 autocast over fp32 weights, matching accelerate mixed_precision.
    latents = torch.randn(batch, 16, hw[0], hw[1], device=device)
    txt = torch.randn(batch, txt_len, 1024, device=device)
    txt_mask = torch.ones(batch, txt_len, device=device, dtype=torch.bool)
    txt_pooled = torch.randn(batch, 1024, device=device)
    t = torch.rand(batch, device=device)  # float32, as in the training loop

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(latents, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask, fast_attn=True)
        loss = out.float().square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    # timed loop
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1e3

    # flop count on one step
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        step()
        torch.cuda.synchronize()
    flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
    tflops = flops / (ms / 1e3) / 1e12
    print(f"{name}: bs={batch} latent={hw[0]}x{hw[1]} txt={txt_len}  "
          f"{ms:.1f} ms/step  {flops/1e12:.2f} TFLOP/step  -> {tflops:.1f} TFLOPS  "
          f"({flops/batch/1e9:.1f} GFLOP/sample)", flush=True)
    return ms, flops


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = ArtFlow(**ARCH).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")
    for name, batch, hw, txt_len in SHAPES:
        if only and name != only:
            continue
        run_shape(name, batch, hw, txt_len, model, device)


if __name__ == "__main__":
    main()
