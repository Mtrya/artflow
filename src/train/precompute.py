"""
Script for offline dataset precomputation.
Downloads images, resizes them, and encodes them with VAE.
"""

import argparse
import ast
from typing import Dict, Tuple

from datasets import load_dataset, load_from_disk

from ..dataset.precompute import precompute


def parse_resolution_buckets(bucket_str: str) -> Dict[int, Tuple[int, int]]:
    """
    Parse resolution buckets string.
    Format examples:
    - "256,256" -> {1: (256, 256)}
    - "256,256 288,208" -> {1: (256, 256), 2: (288, 208)}
    - "[(256,256), (288,208)]" -> {1: (256, 256), 2: (288, 208)}
    """
    buckets = {}

    # Try parsing as python literal (list of tuples)
    try:
        parsed = ast.literal_eval(bucket_str)
        if isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    buckets[i + 1] = (int(item[0]), int(item[1]))
            if buckets:
                return buckets
    except (ValueError, SyntaxError):
        pass

    # Try parsing as space-separated string "w,h w,h"
    try:
        parts = bucket_str.strip().split()
        for i, part in enumerate(parts):
            w, h = map(int, part.split(","))
            buckets[i + 1] = (w, h)
        if buckets:
            return buckets
    except ValueError:
        pass

    raise ValueError(f"Could not parse resolution buckets: {bucket_str}")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute dataset for ArtFlow")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Hugging Face dataset name"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--image_field",
        type=str,
        default="image",
        help="Column name for images or URLs",
    )
    parser.add_argument(
        "--caption_fields",
        type=str,
        nargs="*",
        default=[],
        help="List of caption columns",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="REPA-E/e2e-qwenimage-vae",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--resolution_buckets",
        type=str,
        default="[(256,256)]",
        help="Resolution buckets (e.g. '[(256,256)]' or '256,256')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed dataset",
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--range", type=int, default=-1, help="Range of images to process (for testing)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--non_zh_drop_prob", type=float, default=0.0, help="Probability of dropping non-zh samples")
    parser.add_argument("--resolution_tolerance", type=float, default=1.0, help="Tolerance factor for resolution dropping")
    parser.add_argument("--min_caption_tokens", type=int, default=1, help="Minimum caption tokens to allow")
    parser.add_argument("--max_caption_tokens", type=int, default=1024, help="Maximum caption tokens to allow")
    parser.add_argument("--min_aesthetic_score", type=float, default=0.0, help="Minimum aesthetic score to allow")
    parser.add_argument("--min_watermark_prob", type=float, default=0.6, help="Minimum watermark probability to allow")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset {args.dataset_name}...")
    if args.range > 0:
        split = f"{args.split}[:{args.range}]"
    else:
        split = args.split

    try:
        print("Trying to load dataset from disk...")
        dataset = load_from_disk(args.dataset_name)
    except FileNotFoundError:
        print(f"Trying to load dataset from Hugging Face...")
        dataset = load_dataset(args.dataset_name, split=split)

    # Process caption fields
    raw_caption_fields = args.caption_fields
    if isinstance(raw_caption_fields, list):
        # Flatten and split
        caption_fields = []
        for item in raw_caption_fields:
            caption_fields.extend([x.strip() for x in item.split(",") if x.strip()])
    else:
        # Should be list due to nargs="*", but just in case
        caption_fields = [x.strip() for x in raw_caption_fields.split(",") if x.strip()]

    # Parse buckets
    buckets = parse_resolution_buckets(args.resolution_buckets)
    print(f"Using resolution buckets: {buckets}")

    print("Starting precomputation...")
    processed_dataset = precompute(
        dataset=dataset,
        image_field=args.image_field,
        caption_fields=caption_fields,
        vae_path=args.vae_path,
        resolution_buckets=buckets,
        batch_size=args.batch_size,
        device=args.device,
        non_zh_drop_prob=args.non_zh_drop_prob,
        resolution_tolerance=args.resolution_tolerance,
        min_caption_tokens=args.min_caption_tokens,
        max_caption_tokens=args.max_caption_tokens,
        min_aesthetic_score=args.min_aesthetic_score,
        min_watermark_prob=args.min_watermark_prob
    )
    print(f"Saving processed dataset to {args.output_dir}...")
    processed_dataset.save_to_disk(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()