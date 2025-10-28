"""
Script to inspect the huggan/wikiart dataset and get statistics by artist field.
"""

from datasets import load_dataset
from collections import Counter
import numpy as np


def extract_artist_names(batch: dict) -> dict:
    """Extract artist names from a batch of examples."""
    artist_names = []

    for artist_val in batch['artist']:
        if artist_val is None:
            artist_names.append('unknown')
        elif isinstance(artist_val, dict) and 'name' in artist_val:
            artist_names.append(artist_val['name'])
        elif isinstance(artist_val, str):
            artist_names.append(artist_val)
        else:
            artist_names.append(str(artist_val))

    return {'artist_name': artist_names}


def inspect_wikiart_dataset():
    """
    Load the huggan/wikiart dataset and compute statistics by artist field.
    Uses streaming to avoid loading entire dataset into memory.
    """
    print("Loading huggan/wikiart dataset...")

    dataset = load_dataset("huggan/wikiart")

    print(f"Dataset splits: {list(dataset.keys())}")
    for split_name, split_dataset in dataset.items():
        print(f"{split_name} split size: {len(split_dataset)}")
        print(f"{split_name} features: {split_dataset.features}")

    total_samples = sum(len(split_dataset) for split_dataset in dataset.values())
    print(f"\nTotal number of samples: {total_samples}")

    print("\nExtracting artist names (batched processing)...")
    artist_counts = Counter()

    for split_name, split_dataset in dataset.items():
        print(f"Processing {split_name} split...")
        processed = split_dataset.map(
            extract_artist_names,
            batched=True,
            batch_size=1000,
            remove_columns=split_dataset.column_names,
            desc="Extracting artists"
        )

        for artist_name in processed['artist_name']:
            artist_counts[artist_name] += 1

    print(f"\nNumber of unique artists: {len(artist_counts)}")
    print(f"Artists with most paintings:")

    top_artists = artist_counts.most_common(20)
    for i, (artist, count) in enumerate(top_artists, 1):
        print(f"{i:2d}. {artist:<30} : {count:>4d} paintings")

    counts = list(artist_counts.values())
    print(f"\nArtist statistics:")
    print(f"  Total paintings: {sum(counts)}")
    print(f"  Mean paintings per artist: {np.mean(counts):.2f}")
    print(f"  Median paintings per artist: {np.median(counts):.2f}")
    print(f"  Std dev paintings per artist: {np.std(counts):.2f}")
    print(f"  Min paintings by an artist: {min(counts)}")
    print(f"  Max paintings by an artist: {max(counts)}")

    total_paintings = sum(counts)
    top_10_count = sum(count for _, count in artist_counts.most_common(10))
    top_20_count = sum(count for _, count in artist_counts.most_common(20))
    top_50_count = sum(count for _, count in artist_counts.most_common(50))

    print(f"\nDataset coverage by top artists:")
    print(f"  Top 10 artists: {top_10_count/total_paintings*100:.2f}% of dataset")
    print(f"  Top 20 artists: {top_20_count/total_paintings*100:.2f}% of dataset")
    print(f"  Top 50 artists: {top_50_count/total_paintings*100:.2f}% of dataset")

    return artist_counts

def main():
    """
    Main function to run the inspection.
    """
    print("Starting huggan/wikiart dataset inspection...")
    artist_counts = inspect_wikiart_dataset()

    print("\nInspection completed!")


if __name__ == "__main__":
    main()