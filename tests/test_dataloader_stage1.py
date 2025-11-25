import torch
from src.dataset.dataloader_utils import collate_fn


def test_collate_fn_stage1():
    """Test collate_fn for Stage 1 (no text embeddings)."""
    print(f"{'=' * 60}")
    print("Testing collate_fn for Stage 1 (Raw Captions)")
    print(f"{'=' * 60}\n")

    batch_size = 4
    latent_channels = 16
    latent_h, latent_w = 32, 32

    mock_batch = []
    for i in range(batch_size):
        sample = {
            "caption": ["Caption 1", "Caption 2"],  # List of captions
            "resolution_bucket_id": 3,
            "latents": torch.randn(latent_channels, latent_h, latent_w),
            # No text_embedding
        }
        mock_batch.append(sample)

    batched = collate_fn(mock_batch)

    print("Verifying output keys...")
    assert "latents" in batched
    assert "captions" in batched
    assert "resolution_bucket_ids" in batched
    assert "text_embeddings" not in batched
    print("  ✓ Keys correct")

    print(f"  captions type: {type(batched['captions'])}")
    assert isinstance(batched["captions"], list)
    assert len(batched["captions"]) == batch_size
    assert isinstance(batched["captions"][0], list)
    print("  ✓ Captions structure correct (List[List[str]])")

    print(f"\n{'=' * 60}")
    print("Stage 1 collate_fn test passed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    test_collate_fn_stage1()
