"""DataLoader utilities for resolution bucket sampling and batching."""

from typing import List, Dict, Any
import random

import torch
from torch.utils.data import Sampler


class ResolutionBucketSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # 1. Shuffle all indices
        indices = list(range(len(self.dataset)))
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

        for idx in range(len(self.dataset)):
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


def test_resolution_bucket_sampler():
    """Test the ResolutionBucketSampler to visualize batch grouping and randomization."""
    from torch.utils.data import Dataset

    # Create a mock dataset with resolution buckets
    class MockDataset(Dataset):
        def __init__(self, size=100):
            # Create samples distributed across 7 buckets
            self.data = []
            for i in range(size):
                bucket_id = (i % 7) + 1  # Buckets 1-7
                self.data.append({"idx": i, "resolution_bucket_id": bucket_id})

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create dataset and sampler
    dataset = MockDataset(size=70)  # 10 samples per bucket
    batch_size = 4
    sampler = ResolutionBucketSampler(dataset, batch_size)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Expected batches: {len(sampler)}\n")

    # Test two epochs to show randomization
    for epoch in range(2):
        print(f"{'=' * 60}")
        print(f"EPOCH {epoch + 1}")
        print(f"{'=' * 60}")

        batch_num = 0
        for batch_indices in sampler:
            batch_num += 1
            # Get bucket IDs for this batch
            bucket_ids = [dataset[idx]["resolution_bucket_id"] for idx in batch_indices]
            sample_indices = [dataset[idx]["idx"] for idx in batch_indices]

            # Verify all samples in batch have same resolution
            assert len(set(bucket_ids)) == 1, "Batch contains mixed resolutions!"

            print(
                f"Batch {batch_num:2d} | Bucket {bucket_ids[0]} | "
                f"Size: {len(batch_indices)} | "
                f"Sample indices: {sample_indices}"
            )

        print()

    print(f"{'=' * 60}")
    print("Test passed! All batches contain same-resolution samples.")
    print("Notice how bucket order changes between epochs.")


def test_collate_fn():
    """Test the collate_fn to verify proper padding and batching."""
    print(f"{'=' * 60}")
    print("Testing collate_fn")
    print(f"{'=' * 60}\n")

    # Create mock samples with variable-length text embeddings
    batch_size = 4
    hidden_dim = 512
    latent_channels = 16
    latent_h, latent_w = 32, 32

    # Different sequence lengths to test padding
    seq_lengths = [10, 25, 15, 30]  # Max will be 30

    mock_batch = []
    for i, seq_len in enumerate(seq_lengths):
        sample = {
            "caption": f"Sample caption {i} with length {seq_len}",
            "resolution_bucket_id": 3,
            "latents": torch.randn(latent_channels, latent_h, latent_w),
            "text_embedding": torch.randn(seq_len, hidden_dim),
            "attention_mask": torch.ones(seq_len),
            "pooled_text_embedding": torch.randn(hidden_dim),
        }
        mock_batch.append(sample)

    print(
        f"Created {len(mock_batch)} mock samples with sequence lengths: {seq_lengths}"
    )
    print(f"Expected max_seq_len after padding: {max(seq_lengths)}\n")

    # Run collate_fn
    batched = collate_fn(mock_batch)

    # Verify shapes
    print("Verifying output shapes:")
    print(
        f"  latents: {batched['latents'].shape} (expected: [{batch_size}, {latent_channels}, {latent_h}, {latent_w}])"
    )
    assert batched["latents"].shape == (batch_size, latent_channels, latent_h, latent_w)

    max_seq_len = max(seq_lengths)
    print(
        f"  text_embeddings: {batched['text_embeddings'].shape} (expected: [{batch_size}, {max_seq_len}, {hidden_dim}])"
    )
    assert batched["text_embeddings"].shape == (batch_size, max_seq_len, hidden_dim)

    print(
        f"  attention_masks: {batched['attention_masks'].shape} (expected: [{batch_size}, {max_seq_len}])"
    )
    assert batched["attention_masks"].shape == (batch_size, max_seq_len)

    print(
        f"  pooled_text_embeddings: {batched['pooled_text_embeddings'].shape} (expected: [{batch_size}, {hidden_dim}])"
    )
    assert batched["pooled_text_embeddings"].shape == (batch_size, hidden_dim)

    print(
        f"  resolution_bucket_ids: {batched['resolution_bucket_ids'].shape} (expected: [{batch_size}])"
    )
    assert batched["resolution_bucket_ids"].shape == (batch_size,)

    print(f"  captions: {len(batched['captions'])} strings (expected: {batch_size})")
    assert len(batched["captions"]) == batch_size

    # Verify padding behavior
    print("\nVerifying padding behavior:")
    for i, seq_len in enumerate(seq_lengths):
        # Check that attention mask has correct pattern (1s followed by 0s)
        mask = batched["attention_masks"][i]
        actual_ones = mask[:seq_len].sum().item()
        actual_zeros = mask[seq_len:].sum().item()

        print(f"  Sample {i} (seq_len={seq_len}):")
        print(
            f"    First {seq_len} mask values sum: {actual_ones} (expected: {seq_len})"
        )
        print(
            f"    Remaining {max_seq_len - seq_len} mask values sum: {actual_zeros} (expected: 0)"
        )

        assert actual_ones == seq_len, f"Expected {seq_len} ones, got {actual_ones}"
        assert actual_zeros == 0, f"Expected 0 in padded region, got {actual_zeros}"

    # Test without pooled embeddings
    print("\nTesting without pooled embeddings:")
    batch_no_pooled = []
    for i, seq_len in enumerate(seq_lengths):
        sample = {
            "caption": f"Sample {i}",
            "resolution_bucket_id": 2,
            "latents": torch.randn(latent_channels, latent_h, latent_w),
            "text_embedding": torch.randn(seq_len, hidden_dim),
            "attention_mask": torch.ones(seq_len),
        }
        batch_no_pooled.append(sample)

    batched_no_pooled = collate_fn(batch_no_pooled)
    assert batched_no_pooled["pooled_text_embeddings"] is None
    print("  ✓ pooled_text_embeddings correctly None when not provided")

    print(f"\n{'=' * 60}")
    print("All collate_fn tests passed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Run tests
    test_collate_fn()
    print()
    test_resolution_bucket_sampler()
