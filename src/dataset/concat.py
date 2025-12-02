"""
Script for concatenating multiple precomputed datasets into one.
"""

import argparse
from datasets import load_from_disk, concatenate_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple precomputed datasets"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input precomputed datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the concatenated dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {len(args.inputs)} datasets...")
    datasets = []
    for path in args.inputs:
        print(f"  Loading {path}...")
        ds = load_from_disk(path)
        print(f"    -> {len(ds)} samples")
        datasets.append(ds)

    print("Concatenating datasets...")
    combined = concatenate_datasets(datasets)
    print(f"Combined dataset: {len(combined)} samples")

    print(f"Saving to {args.output}...")
    combined.save_to_disk(args.output)
    print("Done!")


if __name__ == "__main__":
    main()

