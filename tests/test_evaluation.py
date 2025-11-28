import unittest
from unittest.mock import MagicMock, patch
import torch
import os
import sys
import shutil
import tempfile

# Add project root and src to path to mimic the environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.evaluation import run_evaluation_heavy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.test_dir, "checkpoint.pt")
        torch.save({"module": {}}, self.checkpoint_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.utils.evaluation.calculate_clip_score")
    @patch("src.utils.evaluation.calculate_fid")
    @patch("src.flow.solvers.sample_ode")
    @patch("src.utils.encode_text.encode_text")
    @patch("datasets.load_from_disk")
    @patch("transformers.AutoProcessor")
    @patch("transformers.Qwen3VLForConditionalGeneration")
    @patch("diffusers.AutoencoderKLQwenImage")
    @patch("src.models.artflow.ArtFlow")
    @patch("src.dataset.dataloader_utils.ResolutionBucketSampler")
    @patch("src.utils.evaluation.DataLoader")
    def test_run_evaluation_heavy(
        self,
        mock_dataloader_cls,
        mock_sampler,
        mock_artflow,
        mock_vae_cls,
        mock_text_encoder_cls,
        mock_processor_cls,
        mock_load_dataset,
        mock_encode_text,
        mock_sample_ode,
        mock_calc_fid,
        mock_calc_clip,
    ):
        # Setup Mocks

        # Mock ArtFlow
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_artflow.return_value = mock_model

        # Mock VAE
        mock_vae = MagicMock()
        mock_vae.to.return_value = mock_vae
        mock_vae_cls.from_pretrained.return_value = mock_vae
        # VAE decode output
        mock_vae_output = MagicMock()
        mock_vae_output.sample = torch.randn(2, 3, 256, 256)  # Batch size 2
        mock_vae.decode.return_value = mock_vae_output

        # Mock Text Encoder & Processor
        mock_text_encoder = MagicMock()
        mock_text_encoder_cls.from_pretrained.return_value = mock_text_encoder
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        # Mock Dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.column_names = ["latents", "captions", "resolution_bucket_id"]
        mock_dataset.select.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        # Mock DataLoader to return dummy batches
        def dummy_batch_generator():
            for _ in range(2):  # 2 batches
                yield {
                    "latents": torch.randn(2, 4, 32, 32),  # B, C, H, W
                    "captions": [["a photo of a cat"]]
                    * 2,  # List of lists (as per dataset format usually)
                    "resolution_bucket_id": torch.tensor([1, 1]),
                }

        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.side_effect = dummy_batch_generator
        mock_dataloader_cls.return_value = mock_dataloader

        # Mock encode_text
        mock_encode_text.return_value = (
            torch.randn(2, 10, 768),
            torch.ones(2, 10),
            torch.randn(2, 768),
        )

        # Mock sample_ode
        mock_sample_ode.return_value = torch.randn(2, 16, 32, 32)  # B, C, H, W

        # Mock Metrics
        mock_calc_fid.return_value = 12.34
        mock_calc_clip.return_value = 23.45

        # Args
        model_config = {"in_channels": 16}
        output_dir = os.path.join(self.test_dir, "output")

        # Run
        metrics = run_evaluation_heavy(
            checkpoint_path=self.checkpoint_path,
            model_config=model_config,
            vae_path="dummy_vae_path",
            text_encoder_path="dummy_text_encoder_path",
            pooling=True,
            output_dir=output_dir,
            dataset_path="dummy_dataset_path",
            num_fid_samples=4,
            num_clip_samples=4,
            batch_size=2,
            device="cpu",
        )

        # Assertions
        self.assertIn("fid", metrics)
        self.assertEqual(metrics["fid"], 12.34)
        self.assertIn("clip_score", metrics)
        self.assertEqual(metrics["clip_score"], 23.45)

        # Check if files were saved
        self.assertTrue(os.path.exists(os.path.join(output_dir, "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "generated_images")))

        # Verify calls
        mock_artflow.assert_called_once()
        mock_vae_cls.from_pretrained.assert_called_once()
        mock_load_dataset.assert_called_once()
        mock_sample_ode.assert_called()
        mock_calc_fid.assert_called_once()
        mock_calc_clip.assert_called_once()


if __name__ == "__main__":
    unittest.main()
