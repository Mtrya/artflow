import unittest
from unittest.mock import MagicMock, patch
import torch
from datasets import Dataset
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/utils"))
)

from precompute_engine import clean_caption, precompute, sample_caption


class TestPrecomputeEngine(unittest.TestCase):
    def test_clean_caption(self):
        # Test basic quote removal
        self.assertEqual(clean_caption('"""Hello"""'), "Hello")
        self.assertEqual(clean_caption("'''Hello'''"), "Hello")

        # Test recursive/inner removal
        # User wanted to remove inner triple quotes too
        self.assertEqual(clean_caption('""" "The Blue Boy" """'), '"The Blue Boy"')
        self.assertEqual(
            clean_caption('"""Text with """inner""" quotes"""'),
            "Text with inner quotes",
        )

        # Test with user example
        user_example = '"""In this Thursday, Sept. 20, 2018, photo... (AP Photo/Damian Dovarganes)"""'
        expected = (
            "In this Thursday, Sept. 20, 2018, photo... (AP Photo/Damian Dovarganes)"
        )
        self.assertEqual(clean_caption(user_example), expected)

        # Test empty and None
        self.assertEqual(clean_caption(""), "")
        self.assertEqual(clean_caption('""""""'), "")

    def test_sample_caption(self):
        captions = [
            "Short",
            "Medium length caption",
            "Very long caption with many words",
        ]
        # Test sampling returns one of the captions
        sampled = sample_caption(captions, stage=0.5)
        self.assertIn(sampled, captions)

    @patch("diffusers.AutoencoderKLQwenImage")
    @patch("precompute_engine.encode_image")
    def test_precompute_flow(self, mock_encode, mock_vae_cls):
        # Setup mocks
        mock_vae = MagicMock()
        mock_vae_cls.from_pretrained.return_value = mock_vae

        # Mock encode_image to return random tensors matching batch size
        def side_effect_encode(images, vae):
            return torch.randn(len(images), 256)  # Mock latent shape

        mock_encode.side_effect = side_effect_encode

        # Create dummy dataset
        data = {
            "image": [Image.new("RGB", (100, 100)) for _ in range(5)],
            "caption_1": ["Cap 1A", "Cap 1B", "Cap 1C", "Cap 1D", "Cap 1E"],
            "caption_2": ["Cap 2A", "Cap 2B", "Cap 2C", "Cap 2D", "Cap 2E"],
            "other_col": [1, 2, 3, 4, 5],
        }
        dataset = Dataset.from_dict(data)

        resolution_buckets = {1: (100, 100)}

        # Run precompute
        processed_ds = precompute(
            dataset=dataset,
            image_field="image",
            caption_fields=["caption_1", "caption_2"],
            vae_path="dummy/path",
            resolution_buckets=resolution_buckets,
            text_fn=lambda x: x + "_cleaned",  # Simple text_fn
            batch_size=2,
        )

        # Verify output
        self.assertIn("latents", processed_ds.column_names)
        self.assertIn("captions", processed_ds.column_names)
        self.assertIn("resolution_bucket_id", processed_ds.column_names)
        self.assertNotIn("other_col", processed_ds.column_names)

        self.assertEqual(len(processed_ds), 5)

        # Verify content
        row = dict(processed_ds[0])
        # Captions should be a list of strings
        self.assertIsInstance(row["captions"], list)
        self.assertEqual(len(row["captions"]), 2)  # Cap 1A and Cap 2A
        self.assertIn("Cap 1A_cleaned", row["captions"])
        self.assertIn("Cap 2A_cleaned", row["captions"])

        # Verify latents are tensors
        self.assertTrue(torch.is_tensor(row["latents"]))


if __name__ == "__main__":
    unittest.main()
