import unittest
import torch
from torch.utils.data import Dataset
from src.dataset.sampler import ResolutionBucketSampler, collate_fn


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


class TestDataloaderUtils(unittest.TestCase):
    def test_resolution_bucket_sampler(self):
        """Test the ResolutionBucketSampler to visualize batch grouping and randomization."""
        dataset = MockDataset(size=70)  # 10 samples per bucket
        batch_size = 4
        sampler = ResolutionBucketSampler(dataset, batch_size)

        # Test two epochs to show randomization
        for epoch in range(2):
            batch_num = 0
            for batch_indices in sampler:
                batch_num += 1
                # Get bucket IDs for this batch
                bucket_ids = [
                    dataset[idx]["resolution_bucket_id"] for idx in batch_indices
                ]

                # Verify all samples in batch have same resolution
                self.assertEqual(
                    len(set(bucket_ids)), 1, "Batch contains mixed resolutions!"
                )
                self.assertEqual(
                    len(batch_indices),
                    batch_size,
                    "Batch size mismatch (should drop incomplete)",
                )

    def test_collate_fn(self):
        """Test the collate_fn to verify proper padding and batching."""
        batch_size = 4
        latent_channels = 16
        latent_h, latent_w = 32, 32
        hidden_dim = 512

        # Different sequence lengths to test padding
        seq_lengths = [10, 25, 15, 30]  # Max will be 30

        mock_batch = []
        for i, seq_len in enumerate(seq_lengths):
            sample = {
                "captions": f"Sample caption {i} with length {seq_len}",
                "resolution_bucket_id": 3,
                "latents": torch.randn(latent_channels, latent_h, latent_w),
                # Extra fields that might be present but ignored by collate_fn
                "text_embedding": torch.randn(seq_len, hidden_dim),
                "attention_mask": torch.ones(seq_len),
                "pooled_text_embedding": torch.randn(hidden_dim),
            }
            mock_batch.append(sample)

        # Run collate_fn
        batched = collate_fn(mock_batch)

        # Verify shapes
        self.assertEqual(
            batched["latents"].shape, (batch_size, latent_channels, latent_h, latent_w)
        )
        self.assertEqual(batched["resolution_bucket_ids"].shape, (batch_size,))
        self.assertEqual(len(batched["captions"]), batch_size)

    def test_distributed_sampler(self):
        """Test the ResolutionBucketSampler with distributed settings."""

        class DistributedMockDataset(Dataset):
            def __init__(self, size=100):
                self.data = [{"idx": i, "resolution_bucket_id": 1} for i in range(size)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = DistributedMockDataset(size=20)
        batch_size = 2

        # Simulate 2 GPUs
        sampler_rank0 = ResolutionBucketSampler(
            dataset, batch_size, num_replicas=2, rank=0
        )
        sampler_rank1 = ResolutionBucketSampler(
            dataset, batch_size, num_replicas=2, rank=1
        )

        indices_rank0 = []
        for batch in sampler_rank0:
            indices_rank0.extend(batch)

        indices_rank1 = []
        for batch in sampler_rank1:
            indices_rank1.extend(batch)

        # Verification
        all_indices = sorted(indices_rank0 + indices_rank1)

        # Check 1: No overlap
        overlap = set(indices_rank0).intersection(set(indices_rank1))
        self.assertEqual(len(overlap), 0, f"Overlap found between ranks: {overlap}")

        # Check 2: All indices covered (assuming perfect division for this simple case)
        self.assertEqual(
            len(all_indices), 20, f"Expected 20 indices, got {len(all_indices)}"
        )

    def test_collate_fn_stage1(self):
        """Test collate_fn for Stage 1 (Raw Captions)."""
        batch_size = 4
        latent_channels = 16
        latent_h, latent_w = 32, 32

        mock_batch = []
        for i in range(batch_size):
            sample = {
                "captions": ["Caption 1", "Caption 2"],  # List of captions
                "resolution_bucket_id": 3,
                "latents": torch.randn(latent_channels, latent_h, latent_w),
            }
            mock_batch.append(sample)

        batched = collate_fn(mock_batch)

        self.assertIn("latents", batched)
        self.assertIn("captions", batched)
        self.assertIn("resolution_bucket_ids", batched)
        self.assertNotIn("text_embeddings", batched)

        self.assertIsInstance(batched["captions"], list)
        self.assertEqual(len(batched["captions"]), batch_size)
        self.assertIsInstance(batched["captions"][0], list)


if __name__ == "__main__":
    unittest.main()
