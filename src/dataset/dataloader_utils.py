"""DataLoader utilities for resolution bucket sampling and batching."""

from typing import List, Dict, Any
import random

import torch
from torch.utils.data import Sampler


class ResolutionBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas: int = 1, rank: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # 1. Shuffle all indices
        indices = list(range(len(self.dataset)))

        # Subsample for distributed training (shard by rank)
        indices = indices[self.rank :: self.num_replicas]

        random.shuffle(indices)

        # 2. Group by resolution bucket
        from collections import defaultdict

        buckets = defaultdict(list)

        # Optimization: Fetch all bucket IDs at once if possible
        try:
            # For Hugging Face datasets, this returns the column list/array efficiently
            all_bucket_ids = self.dataset["resolution_bucket_id"]
        except (KeyError, TypeError):
            # Fallback for datasets that don't support column access
            all_bucket_ids = [
                self.dataset[i]["resolution_bucket_id"]
                for i in range(len(self.dataset))
            ]

        for idx in indices:
            bucket_id = all_bucket_ids[idx]
            if isinstance(bucket_id, torch.Tensor):
                bucket_id = bucket_id.item()
            buckets[bucket_id].append(idx)

        # 3. Create batches per bucket (drop incomplete batches)
        all_batches = []
        for bucket_indices in buckets.values():
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # Only keep full batches
                    all_batches.append(batch)

        # 4. Shuffle the batches themselves (randomize bucket order)
        random.shuffle(all_batches)

        # 5. Yield batches in random order
        for batch in all_batches:
            yield batch

    def __len__(self):
        # Count only complete batches per bucket
        from collections import defaultdict

        buckets = defaultdict(list)

        # Optimization: Fetch all bucket IDs at once if possible
        try:
            all_bucket_ids = self.dataset["resolution_bucket_id"]
        except (KeyError, TypeError):
            all_bucket_ids = [
                self.dataset[i]["resolution_bucket_id"]
                for i in range(len(self.dataset))
            ]

        # Only consider indices for this rank
        indices = range(len(self.dataset))[self.rank :: self.num_replicas]

        for idx in indices:
            bucket_id = all_bucket_ids[idx]
            if isinstance(bucket_id, torch.Tensor):
                bucket_id = bucket_id.item()
            buckets[bucket_id].append(idx)

        total_batches = 0
        for bucket_indices in buckets.values():
            total_batches += len(bucket_indices) // self.batch_size
        return total_batches


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching precomputed dataset samples.

    Handles variable-length text sequences by padding to max length in batch.

    Args:
        batch: List of dataset samples, each containing:
            - captions: str
            - resolution_bucket_id: int
            - latents: torch.Tensor of shape (C, H, W)

    Returns:
        Batched dictionary with:
            - latents: torch.Tensor of shape (B, C, H, W)
            - captions: List[str] of length B
            - resolution_bucket_ids: torch.Tensor of shape (B,)
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

    return batch_dict
