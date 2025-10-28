"""DataLoader utilities for resolution bucket sampling and batching."""

from typing import List, Dict, Any, Optional
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
        for idx in indices:
            bucket_id = self.dataset[idx]['resolution_bucket_id']
            buckets[bucket_id].append(idx)

        # 3. Create batches per bucket (drop incomplete batches)
        all_batches = []
        for bucket_indices in buckets.values():
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i+self.batch_size]
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
        for idx in range(len(self.dataset)):
            bucket_id = self.dataset[idx]['resolution_bucket_id']
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
            - caption: str
            - resolution_bucket_id: int
            - latents: torch.Tensor of shape (C, H, W)
            - text_embedding: torch.Tensor of shape (seq_len, hidden_dim)
            - attention_mask: torch.Tensor of shape (seq_len,)
            - pooled_text_embedding: Optional[torch.Tensor] of shape (hidden_dim,)

    Returns:
        Batched dictionary with:
            - latents: torch.Tensor of shape (B, C, H, W)
            - text_embeddings: torch.Tensor of shape (B, max_seq_len, hidden_dim)
            - attention_masks: torch.Tensor of shape (B, max_seq_len)
            - pooled_text_embeddings: Optional[torch.Tensor] of shape (B, hidden_dim)
            - captions: List[str] of length B
            - resolution_bucket_ids: torch.Tensor of shape (B,)
    """
    batch_size = len(batch)

    # Extract components
    latents = [sample["latents"] for sample in batch]
    text_embeddings = [sample["text_embedding"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    pooled_embeddings = [sample.get("pooled_text_embedding") for sample in batch]
    captions = [sample["caption"] for sample in batch]
    bucket_ids = [sample["resolution_bucket_id"] for sample in batch]

    # Stack latents (all same resolution in bucket)
    latents_batch = torch.stack(latents, dim=0)

    # Find max sequence length for padding
    max_seq_len = max(emb.shape[0] for emb in text_embeddings)
    hidden_dim = text_embeddings[0].shape[1]

    # Pad text embeddings and attention masks
    padded_embeddings = []
    padded_masks = []

    for emb, mask in zip(text_embeddings, attention_masks):
        seq_len = emb.shape[0]
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            # Pad embeddings with zeros
            padded_emb = torch.cat([
                emb,
                torch.zeros(pad_len, hidden_dim, dtype=emb.dtype, device=emb.device)
            ], dim=0)
            # Pad mask with zeros (0 = ignore)
            padded_mask = torch.cat([
                mask,
                torch.zeros(pad_len, dtype=mask.dtype, device=mask.device)
            ], dim=0)
        else:
            padded_emb = emb
            padded_mask = mask

        padded_embeddings.append(padded_emb)
        padded_masks.append(padded_mask)

    text_embeddings_batch = torch.stack(padded_embeddings, dim=0)
    attention_masks_batch = torch.stack(padded_masks, dim=0)

    # Stack pooled embeddings if present
    pooled_batch = None
    if pooled_embeddings[0] is not None:
        pooled_batch = torch.stack([p for p in pooled_embeddings if p is not None], dim=0)

    # Convert bucket IDs to tensor
    bucket_ids_batch = torch.tensor(bucket_ids, dtype=torch.long)

    return {
        "latents": latents_batch,
        "text_embeddings": text_embeddings_batch,
        "attention_masks": attention_masks_batch,
        "pooled_text_embeddings": pooled_batch,
        "captions": captions,
        "resolution_bucket_ids": bucket_ids_batch,
    }

def test_resolution_bucket_sampler():
    """Test the ResolutionBucketSampler to visualize batch grouping and randomization."""
    from torch.utils.data import Dataset, DataLoader

    # Create a mock dataset with resolution buckets
    class MockDataset(Dataset):
        def __init__(self, size=100):
            # Create samples distributed across 7 buckets
            self.data = []
            for i in range(size):
                bucket_id = (i % 7) + 1  # Buckets 1-7
                self.data.append({
                    'idx': i,
                    'resolution_bucket_id': bucket_id
                })

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
        print(f"{'='*60}")
        print(f"EPOCH {epoch + 1}")
        print(f"{'='*60}")

        batch_num = 0
        for batch_indices in sampler:
            batch_num += 1
            # Get bucket IDs for this batch
            bucket_ids = [dataset[idx]['resolution_bucket_id'] for idx in batch_indices]
            sample_indices = [dataset[idx]['idx'] for idx in batch_indices]

            # Verify all samples in batch have same resolution
            assert len(set(bucket_ids)) == 1, "Batch contains mixed resolutions!"

            print(f"Batch {batch_num:2d} | Bucket {bucket_ids[0]} | "
                  f"Size: {len(batch_indices)} | "
                  f"Sample indices: {sample_indices}")

        print()

    print(f"{'='*60}")
    print("Test passed! All batches contain same-resolution samples.")
    print("Notice how bucket order changes between epochs.")

if __name__ == '__main__':
    test_resolution_bucket_sampler()