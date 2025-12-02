"""
Dataset mixing utilities for multi-dataset training.

Provides functionality to:
- Parse dataset mix specifications from CLI
- Load and concatenate multiple datasets with tracking
- Support weighted sampling across datasets
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from datasets import Dataset, concatenate_datasets, load_from_disk


@dataclass
class DatasetEntry:
    """A single dataset entry in a mix specification."""

    path: Path
    weight: float
    alias: str  # Short identifier for logging


def parse_dataset_mix(mix_spec: str) -> List[DatasetEntry]:
    """Parse a dataset mix specification string.

    Format: "path1:weight1 path2:weight2 ..." or just "path1" for single dataset.

    Examples:
        "data/wikiart:0.9 data/relaion:0.1"
        "data/wikiart"  # Single dataset, weight defaults to 1.0

    Args:
        mix_spec: Space-separated list of "path:weight" pairs

    Returns:
        List of DatasetEntry objects with normalized weights

    Raises:
        ValueError: If weights are invalid or paths are malformed
    """
    entries = []
    parts = mix_spec.strip().split()

    for part in parts:
        if ":" in part:
            path_str, weight_str = part.rsplit(":", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(f"Invalid weight '{weight_str}' in '{part}'")
            if weight <= 0:
                raise ValueError(f"Weight must be positive, got {weight} in '{part}'")
        else:
            path_str = part
            weight = 1.0

        path = Path(path_str)
        # Use directory name as alias (e.g., "wikiart-captions@256p")
        alias = path.name

        entries.append(DatasetEntry(path=path, weight=weight, alias=alias))

    if not entries:
        raise ValueError("Empty dataset mix specification")

    # Normalize weights to sum to 1.0
    total_weight = sum(e.weight for e in entries)
    for entry in entries:
        entry.weight = entry.weight / total_weight

    return entries


def load_mixed_dataset(
    entries: List[DatasetEntry],
    shuffle_seed: Optional[int] = None,
) -> Dataset:
    """Load and concatenate multiple datasets with dataset_id tracking.

    Each dataset gets a `dataset_id` column (integer index) added before
    concatenation for downstream tracking and weighted sampling.

    Args:
        entries: List of DatasetEntry objects from parse_dataset_mix
        shuffle_seed: If provided, shuffle each dataset before concatenation

    Returns:
        Concatenated dataset with `dataset_id` column
    """
    datasets = []

    for idx, entry in enumerate(entries):
        ds = load_from_disk(str(entry.path))

        # Add dataset_id column for tracking
        ds = ds.add_column("dataset_id", [idx] * len(ds))

        # Optional pre-shuffle for local randomness
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed + idx)

        datasets.append(ds)

    return concatenate_datasets(datasets)


def get_dataset_weights(entries: List[DatasetEntry]) -> List[float]:
    """Extract normalized weights from dataset entries.

    Args:
        entries: List of DatasetEntry objects

    Returns:
        List of weights (sums to 1.0)
    """
    return [e.weight for e in entries]


def get_dataset_aliases(entries: List[DatasetEntry]) -> List[str]:
    """Extract aliases from dataset entries.

    Args:
        entries: List of DatasetEntry objects

    Returns:
        List of alias strings
    """
    return [e.alias for e in entries]
