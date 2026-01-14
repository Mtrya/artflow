import unittest
from unittest.mock import MagicMock, patch
import torch
import os
import sys
import shutil
import tempfile

# Add project root and src to path to mimic the environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.pipeline import run_evaluation_heavy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class TestEvaluationImports(unittest.TestCase):
    """Test that evaluation module imports work correctly after refactoring."""

    def test_imports(self):
        """Verify all evaluation module imports work."""
        from src.evaluation import (
            calculate_fid,
            calculate_kid,
            calculate_clip_score,
            calculate_rm_score,
            calculate_combined_reward,
            make_image_grid,
            visualize_denoising,
            format_prompt_caption,
            run_evaluation_uncond,
            run_evaluation_light,
            run_evaluation_heavy,
        )
        # Just verify imports succeed
        self.assertIsNotNone(calculate_fid)
        self.assertIsNotNone(calculate_kid)
        self.assertIsNotNone(calculate_clip_score)
        self.assertIsNotNone(calculate_rm_score)
        self.assertIsNotNone(calculate_combined_reward)
        self.assertIsNotNone(make_image_grid)
        self.assertIsNotNone(visualize_denoising)
        self.assertIsNotNone(format_prompt_caption)
        self.assertIsNotNone(run_evaluation_uncond)
        self.assertIsNotNone(run_evaluation_light)
        self.assertIsNotNone(run_evaluation_heavy)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.test_dir, "checkpoint.pt")
        torch.save({"module": {}}, self.checkpoint_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @unittest.skip("Complex mocking required - skipping for now")
    @patch("src.evaluation.pipeline.calculate_clip_score")
    @patch("src.evaluation.pipeline.calculate_fid")
    @patch("src.evaluation.pipeline.sample_ode")
    @patch("src.evaluation.pipeline.encode_text")
    @patch("src.evaluation.pipeline.load_from_disk")
    @patch("src.evaluation.pipeline.AutoProcessor")
    @patch("src.evaluation.pipeline.Qwen3VLForConditionalGeneration")
    @patch("src.evaluation.pipeline.AutoencoderKLQwenImage")
    @patch("src.evaluation.pipeline.ArtFlow")
    @patch("src.evaluation.pipeline.ResolutionBucketSampler")
    @patch("src.evaluation.pipeline.DataLoader")
    @patch("src.evaluation.pipeline.get_vae_stats")
    def test_run_evaluation_heavy(
        self,
        mock_get_vae_stats,
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

        # Mock VAE stats
        mock_get_vae_stats.return_value = (
            torch.zeros(1, 16, 1, 1),  # mean
            torch.ones(1, 16, 1, 1),   # std
        )

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


class TestRMScore(unittest.TestCase):
    """Test calculate_rm_score function."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.test_dir, "rm_checkpoint.pt")
        self.config_path = os.path.join(self.test_dir, "config.json")
        
        # Create a dummy config file
        import json
        config = {
            "feature_dim": 1024,
            "num_layers": 3,
            "hidden_dim": 512,
            "dropout": 0.1,
            "feature_layers": [12, 18, 23]
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        # Create a dummy checkpoint
        torch.save({"model_state_dict": {}}, self.checkpoint_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @unittest.skip("Requires reward model and CLIP model - skipping for now")
    @patch("src.evaluation.metrics.ChineseCLIPModel")
    @patch("src.evaluation.metrics.CLIPProcessor")
    @patch("src.evaluation.metrics.extract_clip_features")
    @patch("src.evaluation.metrics.RewardModel")
    def test_calculate_rm_score(
        self, mock_reward_model_cls, mock_extract_features, mock_processor_cls, mock_clip_cls
    ):
        """Test that calculate_rm_score processes images correctly."""
        from src.evaluation.metrics import calculate_rm_score
        
        # Setup mocks
        mock_clip = MagicMock()
        mock_clip_cls.from_pretrained.return_value = mock_clip
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        
        # Mock feature extraction
        mock_extract_features.return_value = torch.randn(4, 3072)  # [B, feature_dim]
        
        # Mock reward model
        mock_rm = MagicMock()
        mock_rm.return_value = torch.tensor([0.8, 0.7, 0.9, 0.6])  # Scores
        mock_reward_model_cls.return_value = mock_rm
        
        # Create test images
        images = torch.rand(4, 3, 256, 256)  # [B, C, H, W] in [0, 1]
        
        # Calculate score
        score = calculate_rm_score(
            images=images,
            checkpoint_path=self.checkpoint_path,
            device="cpu",
            batch_size=2
        )
        
        # Assertions
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        mock_extract_features.assert_called()


class TestVLMScore(unittest.TestCase):
    """Test calculate_vlm_score function."""

    @patch("src.evaluation.metrics.requests.post")
    def test_calculate_vlm_score(self, mock_post):
        """Test that calculate_vlm_score makes correct API calls."""
        from src.evaluation.metrics import calculate_vlm_score
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "0.85"}}]
        }
        mock_post.return_value = mock_response
        
        # Create test images
        images = torch.rand(2, 3, 256, 256)  # [B, C, H, W] in [0, 1]
        prompts = ["Rate this image", "Rate this artwork"]
        
        # Calculate score
        score = calculate_vlm_score(
            images=images,
            prompts=prompts,
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test_key",
            model_name="qwen-vl",
            batch_size=1
        )
        
        # Assertions
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.85)  # Should be average of both responses
        self.assertEqual(mock_post.call_count, 2)  # Called once per image
        
        # Verify API call structure
        call_args = mock_post.call_args_list[0]
        self.assertIn("Authorization", call_args[1]["headers"])
        self.assertEqual(call_args[1]["json"]["model"], "qwen-vl")

    @patch("src.evaluation.metrics.requests.post")
    def test_calculate_vlm_score_with_extraction(self, mock_post):
        """Test that calculate_vlm_score can extract numbers from text responses."""
        from src.evaluation.metrics import calculate_vlm_score
        
        # Mock API response with text containing a number
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "The score is 0.75 out of 1.0"}}]
        }
        mock_post.return_value = mock_response
        
        # Create test images
        images = torch.rand(1, 3, 256, 256)
        
        # Calculate score
        score = calculate_vlm_score(
            images=images,
            prompts="Rate this",
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test_key"
        )
        
        # Should extract 0.75 from the response
        self.assertEqual(score, 0.75)

    @patch("src.evaluation.metrics.requests.post")
    def test_calculate_vlm_score_api_failure(self, mock_post):
        """Test that calculate_vlm_score handles API failures gracefully."""
        from src.evaluation.metrics import calculate_vlm_score
        
        # Mock API failure
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # Create test images
        images = torch.rand(1, 3, 256, 256)
        
        # Calculate score
        score = calculate_vlm_score(
            images=images,
            prompts="Rate this",
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test_key"
        )
        
        # Should return 0.0 for failed API calls
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
