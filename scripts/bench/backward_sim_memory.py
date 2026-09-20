#!/usr/bin/env python
"""Measure backward-simulation peak memory for the Stage-6 DMD2 loop.

The joint DMD+RL loop's binding memory question: how many gradient-bearing
generator steps fit when the 8-step student's backward simulation backprops
through the last `grad_steps` of its inference grid? This probe builds the
hero-architecture model (533M: h1152, 16 heads, 1 double + 24 single) and
reports peak allocated/reserved memory per (latent size, micro-batch,
grad-depth).

Runs on any single CUDA GPU. Absolute numbers on a 16-GB card set a lower
bound; the grad-depth scaling is what transfers to the 48-GB hero nodes.

Example:
    python -m scripts.bench.backward_sim_memory --latent 32 80 --micro 1 2 \
        --grad-steps 1 2 4 8 --txt-len 512
"""

import argparse
import gc
import json

import torch

from src.models.artflow import ArtFlow
from src.posttrain.dmd import backward_simulate

HERO = dict(patch_size=2, in_channels=16, txt_in_features=1024,
            hidden_size=1152, num_heads=16,
            double_stream_depth=1, single_stream_depth=24,
            mlp_ratio=2.67, conditioning_scheme="fused",
            qkv_bias=True, double_stream_modulation="none",
            single_stream_modulation="layer", ffn_type="gated")


def measure(model, latent_hw, micro, grad_steps, txt_len, grid_steps=8,
            dtype=torch.bfloat16):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    grid = torch.linspace(0.0, 1.0, grid_steps + 1)
    start_idx = grid_steps - grad_steps
    z = torch.randn(micro, 16, latent_hw, latent_hw, device="cuda")
    txt = torch.randn(micro, txt_len, 1024, device="cuda")
    txt_pooled = torch.randn(micro, 1024, device="cuda")

    def model_fn(x, t_scalar):
        t = torch.full((x.shape[0],), float(t_scalar), device=x.device)
        with torch.autocast("cuda", dtype=dtype):
            return model(x, t, txt, txt_pooled=txt_pooled)

    out = backward_simulate(model_fn, z, grid, start_idx)
    loss = out.float().square().mean()
    loss.backward()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model.zero_grad(set_to_none=True)
    del out, loss, z, txt, txt_pooled
    return peak_alloc, peak_reserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=int, nargs="+", default=[32, 80, 112],
                        help="latent H=W; 32/80/112 = 256p/640p/896p")
    parser.add_argument("--micro", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--grad-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--txt-len", type=int, default=512)
    parser.add_argument("--grid-steps", type=int, default=8)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    model = ArtFlow(**HERO).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
    results = []
    for latent in args.latent:
        for micro in args.micro:
            for grad_steps in args.grad_steps:
                tag = dict(latent=latent, micro=micro, grad_steps=grad_steps)
                try:
                    alloc, reserved = measure(model, latent, micro, grad_steps,
                                              args.txt_len, args.grid_steps)
                    tag.update(peak_alloc_gib=round(alloc / 2**30, 3),
                               peak_reserved_gib=round(reserved / 2**30, 3))
                except torch.cuda.OutOfMemoryError:
                    tag.update(oom=True)
                    torch.cuda.empty_cache()
                print(json.dumps(tag), flush=True)
                results.append(tag)
    print(json.dumps({"params": n_params, "results": results}))


if __name__ == "__main__":
    main()
