"""
Custom samplers and collate functions for bucket-based batch sampling.

Classes:
- ResolutionBucketSampler: Sampler that groups samples by resolution bucket
  with optional weighted multi-dataset mixing

Functions:
- collate_fn: Collate function for batching precomputed dataset samples
"""

import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class ResolutionBucketSampler(Sampler):
    """
    Sampler that groups samples by resolution bucket for efficient batching.

    Ensures all samples in a batch have the same resolution, which is required
    for efficient GPU processing without padding.

    Supports weighted multi-dataset mixing when `dataset_weights` is provided.
    In this mode, samples are grouped by (dataset_id, bucket_id) and batches
    are drawn from datasets according to the specified weights. When a dataset
    exhausts its batches, it is reshuffled and continues sampling to maintain
    stable ratios throughout training.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        dataset_weights: Optional[List[float]] = None,
    ):
        """Initialize the sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Number of samples per batch
            num_replicas: Number of distributed processes
            rank: Current process rank
            shuffle: Whether to shuffle indices and batches
            drop_last: Whether to drop incomplete batches
            dataset_weights: Optional list of weights for multi-dataset mixing.
                Index corresponds to dataset_id. If None, single-dataset mode.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset_weights = dataset_weights

        # Pre-fetch columns and compute rank indices (vectorized, done once)
        self._all_bucket_ids = np.asarray(self._get_column("resolution_bucket_id"))
        self._rank_indices = np.arange(len(self.dataset))[self.rank :: self.num_replicas]

        if self.dataset_weights is not None:
            self._all_dataset_ids = np.asarray(self._get_column("dataset_id"))
            # Pre-group indices by dataset for fast rebuilding
            self._dataset_indices = self._build_dataset_indices()

    def _get_column(self, column_name: str) -> List:
        """Efficiently fetch a column from the dataset."""
        try:
            return self.dataset[column_name]
        except (KeyError, TypeError):
            return [self.dataset[i][column_name] for i in range(len(self.dataset))]

    def _build_dataset_indices(self) -> Dict[int, np.ndarray]:
        """Build mapping from dataset_id to indices (for this rank)."""
        rank_dataset_ids = self._all_dataset_ids[self._rank_indices]
        unique_ids = np.unique(rank_dataset_ids)
        return {
            int(ds_id): self._rank_indices[rank_dataset_ids == ds_id]
            for ds_id in unique_ids
        }

    def _build_batches_for_dataset(self, dataset_id: int) -> deque:
        """Build batches for a single dataset using vectorized operations."""
        indices = self._dataset_indices[dataset_id].copy()

        if self.shuffle:
            np.random.shuffle(indices)

        # Get bucket IDs for these indices
        bucket_ids = self._all_bucket_ids[indices]

        # Group by bucket using numpy
        unique_buckets = np.unique(bucket_ids)
        batches = []

        for bucket_id in unique_buckets:
            bucket_mask = bucket_ids == bucket_id
            bucket_indices = indices[bucket_mask]

            # Create batches
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size].tolist()
                if self.drop_last:
                    if len(batch) == self.batch_size:
                        batches.append(batch)
                else:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        return deque(batches)

    def __iter__(self):
        if self.dataset_weights is not None:
            yield from self._iter_weighted()
        else:
            yield from self._iter_simple()

    def _iter_simple(self):
        """Single-dataset iteration with vectorized bucket grouping."""
        indices = self._rank_indices.copy()

        if self.shuffle:
            np.random.shuffle(indices)

        # Get bucket IDs for rank indices (vectorized)
        bucket_ids = self._all_bucket_ids[indices]

        # Group by bucket using numpy
        unique_buckets = np.unique(bucket_ids)
        all_batches = []

        for bucket_id in unique_buckets:
            bucket_mask = bucket_ids == bucket_id
            bucket_indices = indices[bucket_mask]

            # Create batches
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size].tolist()
                if self.drop_last:
                    if len(batch) == self.batch_size:
                        all_batches.append(batch)
                else:
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def _iter_weighted(self):
        """Weighted multi-dataset iteration with reshuffle on exhaustion."""
        # Build initial batches for each dataset
        dataset_batches: Dict[int, deque] = {
            ds_id: self._build_batches_for_dataset(ds_id)
            for ds_id in self._dataset_indices.keys()
        }

        # Filter to datasets with batches
        active_datasets = {
            ds_id: batches
            for ds_id, batches in dataset_batches.items()
            if len(batches) > 0
        }

        if not active_datasets:
            return

        # Pre-compute total batches for epoch length estimation
        # (sum of all datasets' batch counts)
        total_batches = sum(len(b) for b in dataset_batches.values())
        batches_yielded = 0

        while batches_yielded < total_batches:
            # Get weights for active datasets (use original weights, not renormalized)
            active_ids = list(active_datasets.keys())
            active_weights = [self.dataset_weights[ds_id] for ds_id in active_ids]

            # Normalize weights
            total_weight = sum(active_weights)
            if total_weight <= 0:
                break
            normalized_weights = [w / total_weight for w in active_weights]

            # Sample a dataset according to weights
            chosen_id = random.choices(active_ids, weights=normalized_weights, k=1)[0]

            # If chosen dataset is empty, reshuffle and rebuild its batches
            if len(active_datasets[chosen_id]) == 0:
                new_batches = self._build_batches_for_dataset(chosen_id)
                if len(new_batches) == 0:
                    # Dataset has no valid batches (e.g., too few samples)
                    del active_datasets[chosen_id]
                    continue
                active_datasets[chosen_id] = new_batches

            batch = active_datasets[chosen_id].popleft()
            yield batch
            batches_yielded += 1

    def __len__(self):
        if self.dataset_weights is not None:
            return self._len_weighted()
        return self._len_simple()

    def _len_simple(self):
        """Count batches for single-dataset mode (vectorized)."""
        bucket_ids = self._all_bucket_ids[self._rank_indices]
        unique_buckets, counts = np.unique(bucket_ids, return_counts=True)
        return int(np.sum(counts // self.batch_size))

    def _len_weighted(self):
        """Count batches for weighted multi-dataset mode (vectorized)."""
        rank_bucket_ids = self._all_bucket_ids[self._rank_indices]
        rank_dataset_ids = self._all_dataset_ids[self._rank_indices]

        # Create composite key for grouping
        # Shift dataset_id to avoid collision with bucket_id
        max_bucket = int(rank_bucket_ids.max()) + 1 if len(rank_bucket_ids) > 0 else 1
        composite_keys = rank_dataset_ids * max_bucket + rank_bucket_ids

        unique_keys, counts = np.unique(composite_keys, return_counts=True)
        return int(np.sum(counts // self.batch_size))


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching precomputed dataset samples.

    Handles variable-length text sequences by padding to max length in batch.

    Args:
        batch: List of dataset samples, each containing:
            - captions: str
            - resolution_bucket_id: int
            - latents: torch.Tensor of shape (C, H, W)
            - dataset_id: int (optional, for multi-dataset mixing)

    Returns:
        Batched dictionary with:
            - latents: torch.Tensor of shape (B, C, H, W)
            - captions: List[str] of length B
            - resolution_bucket_ids: torch.Tensor of shape (B,)
            - dataset_ids: torch.Tensor of shape (B,) if present in samples
    """
    # Extract components
    latents = [sample["latents"] for sample in batch]
    captions = [sample["captions"] for sample in batch]
    bucket_ids = [sample["resolution_bucket_id"] for sample in batch]

    # Stack latents (all same resolution in bucket)
    latents_batch = torch.stack(latents, dim=0)

    # Convert bucket IDs to tensor
    bucket_ids_batch = torch.tensor(bucket_ids, dtype=torch.long)

    batch_dict = {
        "latents": latents_batch,
        "captions": captions,
        "resolution_bucket_ids": bucket_ids_batch,
    }

    # Include dataset_id if present (multi-dataset mode)
    if "dataset_id" in batch[0]:
        dataset_ids = [sample["dataset_id"] for sample in batch]
        batch_dict["dataset_ids"] = torch.tensor(dataset_ids, dtype=torch.long)

    return batch_dict

