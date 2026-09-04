#!/usr/bin/env python
"""
Instantiate ArtFlow in a given shape and print the trainable-param count.

Used to re-derive iso-param / iso-FLOP arm shapes after the 2.2a modulation
verdict (mod=layer vs mod=none changes params per layer, so the §5 shapes must
be re-matched to the winner). Pure CPU instantiation; no data, no CUDA.

Usage:
  python scripts/stage2/param_count.py \
      --hidden_size 1024 --num_heads 16 \
      --single_stream_depth 24 --double_stream_depth 0 \
      --single_stream_mod layer --double_stream_mod none
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.artflow import ArtFlow  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--single_stream_depth", type=int, default=24)
    p.add_argument("--double_stream_depth", type=int, default=0)
    p.add_argument("--single_stream_mod", default="none")
    p.add_argument("--double_stream_mod", default="none")
    p.add_argument("--mlp_ratio", type=float, default=2.67)
    p.add_argument("--txt_in_features", type=int, default=1024)
    args = p.parse_args()

    model = ArtFlow(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        double_stream_depth=args.double_stream_depth,
        single_stream_depth=args.single_stream_depth,
        mlp_ratio=args.mlp_ratio,
        conditioning_scheme="fused",
        qkv_bias=True,
        double_stream_modulation=args.double_stream_mod,
        single_stream_modulation=args.single_stream_mod,
        ffn_type="gated",
        rope_centered_grid=True,
        patch_size=2,
        in_channels=16,
        txt_in_features=args.txt_in_features,
    )
    n = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(
        f"h{args.hidden_size} d2x{args.double_stream_depth}+s{args.single_stream_depth} "
        f"heads{args.num_heads} mod(dbl={args.double_stream_mod},sgl={args.single_stream_mod}) "
        f"mlp{args.mlp_ratio} -> {n:,} params"
    )


if __name__ == "__main__":
    main()
