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
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate by prompt",
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
        print(f"    -> Format: {ds.format}")
        
        # Reset format to None before concatenation to avoid conflicts
        if ds.format is not None:
            ds = ds.with_format(None)
        
        datasets.append(ds)

    print("Concatenating datasets...")
    combined = concatenate_datasets(datasets)
    print(f"Combined dataset: {len(combined)} samples")
    print(f"Format after concatenation: {combined.format}")

    if args.dedup:
        print("Deduplicating by 'prompt' column...")
        original_len = len(combined)
        import pandas as pd
        prompts_df = pd.DataFrame({"prompt": combined.column("prompt").to_pylist()})

        keep_mask = ~prompts_df.duplicated(subset="prompt", keep="first")
        indices_to_keep = keep_mask[keep_mask].index.tolist()

        combined = combined.select(indices_to_keep)
        print(f"Deduplication: {original_len} -> {(len(combined))} samples")
    
    # Set torch format to match precomputed datasets
    print("Setting format to 'torch'...")
    combined = combined.with_format("torch")
    
    # Verify the latents column type
    if len(combined) > 0:
        sample = combined[0]
        print(f"Verification - Sample latents type: {type(sample['latents'])}")
        if hasattr(sample['latents'], 'shape'):
            print(f"Verification - Sample latents shape: {sample['latents'].shape}")
        else:
            print(f"WARNING: latents is not a tensor! Type: {type(sample['latents'])}")

    print(f"Saving to {args.output}...")
    combined.save_to_disk(args.output)
    print("Done!")


if __name__ == "__main__":
    main()

