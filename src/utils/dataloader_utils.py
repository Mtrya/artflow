


import random

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