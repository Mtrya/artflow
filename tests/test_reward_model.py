"""
Test suite for src/models/reward_model.py.

Tests:
- RewardModel: Reward model architecture and forward pass
- extract_clip_features: Multi-layer CLIP feature extraction
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Add project root and src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.reward_model import RewardModel, extract_clip_features


class TestRewardModel(unittest.TestCase):
    """Test RewardModel class."""
    
    def test_initialization(self):
        """Test reward model initialization."""
        model = RewardModel(
            feature_dim=768,
            num_layers=3,
            hidden_dim=256,
            dropout=0.1,
        )
        
        self.assertEqual(model.feature_dim, 768)
        self.assertEqual(model.num_layers, 3)
        self.assertIsInstance(model.head, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape and range."""
        model = RewardModel(
            feature_dim=768,
            num_layers=3,
            hidden_dim=256,
            dropout=0.1,
        )
        model.eval()
        
        batch_size = 4
        features = torch.randn(batch_size, 768 * 3)
        
        with torch.no_grad():
            scores = model(features)
        
        # Check shape
        self.assertEqual(scores.shape, (batch_size,))
        
        # Check range [0, 1] due to sigmoid
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))
    
    def test_parameter_counts(self):
        """Test parameter counting methods."""
        model = RewardModel(
            feature_dim=768,
            num_layers=3,
            hidden_dim=256,
            dropout=0.1,
        )
        
        num_trainable = model.get_num_trainable_params()
        num_total = model.get_num_total_params()
        
        # Both should be positive
        self.assertGreater(num_trainable, 0)
        self.assertGreater(num_total, 0)
        
        # For this model, all params are trainable
        self.assertEqual(num_trainable, num_total)
        
        # Verify counts are reasonable (rough estimate)
        # Input: 768*3 = 2304, Hidden: 256, Output: 1
        # Layer1: 2304*256 + 256 = 590080
        # Layer2: 256*128 + 128 = 32896
        # Layer3: 128*1 + 1 = 129
        # Plus LayerNorm params
        # Should be around 600k-700k params
        self.assertGreater(num_total, 500000)
        self.assertLess(num_total, 1000000)
    
    def test_different_configurations(self):
        """Test model with different configurations."""
        configs = [
            {"feature_dim": 512, "num_layers": 2, "hidden_dim": 128, "dropout": 0.0},
            {"feature_dim": 1024, "num_layers": 4, "hidden_dim": 512, "dropout": 0.2},
            {"feature_dim": 768, "num_layers": 1, "hidden_dim": 256, "dropout": 0.15},
        ]
        
        for config in configs:
            model = RewardModel(**config)
            model.eval()
            
            batch_size = 2
            input_dim = config["feature_dim"] * config["num_layers"]
            features = torch.randn(batch_size, input_dim)
            
            with torch.no_grad():
                scores = model(features)
            
            self.assertEqual(scores.shape, (batch_size,))
            self.assertTrue(torch.all(scores >= 0))
            self.assertTrue(torch.all(scores <= 1))
    
    def test_batch_size_flexibility(self):
        """Test model handles different batch sizes."""
        model = RewardModel(
            feature_dim=768,
            num_layers=3,
            hidden_dim=256,
            dropout=0.1,
        )
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, 768 * 3)
            
            with torch.no_grad():
                scores = model(features)
            
            self.assertEqual(scores.shape, (batch_size,))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = RewardModel(
            feature_dim=768,
            num_layers=3,
            hidden_dim=256,
            dropout=0.1,
        )
        model.train()
        
        batch_size = 4
        features = torch.randn(batch_size, 768 * 3, requires_grad=True)
        
        scores = model(features)
        loss = scores.mean()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(features.grad)
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestExtractCLIPFeatures(unittest.TestCase):
    """Test extract_clip_features function."""
    
    def create_mock_clip_model(self, hidden_size=768, num_layers=24):
        """Create a mock CLIP model for testing."""
        mock_model = MagicMock()
        mock_vision_model = MagicMock()
        
        # Create mock hidden states
        def mock_forward(pixel_values, output_hidden_states=False):
            batch_size = pixel_values.shape[0]
            seq_len = 77  # Typical for CLIP
            
            # Create tuple of hidden states for all layers
            hidden_states = tuple(
                torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
            )
            
            outputs = MagicMock()
            outputs.hidden_states = hidden_states
            return outputs
        
        mock_vision_model.side_effect = mock_forward
        mock_model.vision_model = mock_vision_model
        
        return mock_model
    
    def create_mock_processor(self):
        """Create a mock CLIP processor."""
        mock_processor = MagicMock()
        
        def mock_process(images, return_tensors="pt"):
            batch_size = len(images)
            # Return dummy pixel values
            return {"pixel_values": torch.randn(batch_size, 3, 336, 336)}
        
        mock_processor.side_effect = mock_process
        return mock_processor
    
    def test_basic_feature_extraction(self):
        """Test basic multi-layer feature extraction."""
        batch_size = 4
        images = torch.rand(batch_size, 3, 336, 336)
        
        mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
        mock_processor = self.create_mock_processor()
        feature_layers = [12, 18, 23]
        
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            feature_layers=feature_layers,
            device=torch.device("cpu"),
        )
        
        # Check output shape
        expected_dim = 768 * len(feature_layers)
        self.assertEqual(features.shape, (batch_size, expected_dim))
    
    def test_default_feature_layers(self):
        """Test that default feature layers work."""
        batch_size = 2
        images = torch.rand(batch_size, 3, 336, 336)
        
        mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
        mock_processor = self.create_mock_processor()
        
        # Don't specify feature_layers, should use default [12, 18, 23]
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            device=torch.device("cpu"),
        )
        
        # Default is 3 layers
        expected_dim = 768 * 3
        self.assertEqual(features.shape, (batch_size, expected_dim))
    
    def test_image_range_01(self):
        """Test feature extraction with images in [0, 1] range."""
        batch_size = 2
        images = torch.rand(batch_size, 3, 336, 336)  # [0, 1] range
        
        mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
        mock_processor = self.create_mock_processor()
        
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            feature_layers=[12, 18, 23],
            device=torch.device("cpu"),
        )
        
        self.assertEqual(features.shape, (batch_size, 768 * 3))
    
    def test_image_range_255(self):
        """Test feature extraction with images in [0, 255] range."""
        batch_size = 2
        images = torch.randint(0, 256, (batch_size, 3, 336, 336), dtype=torch.uint8)
        
        mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
        mock_processor = self.create_mock_processor()
        
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            feature_layers=[12, 18, 23],
            device=torch.device("cpu"),
        )
        
        self.assertEqual(features.shape, (batch_size, 768 * 3))
    
    def test_single_layer_extraction(self):
        """Test extracting features from a single layer."""
        batch_size = 2
        images = torch.rand(batch_size, 3, 336, 336)
        
        mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
        mock_processor = self.create_mock_processor()
        feature_layers = [23]  # Only last layer
        
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            feature_layers=feature_layers,
            device=torch.device("cpu"),
        )
        
        # Should be single layer dimension
        self.assertEqual(features.shape, (batch_size, 768))
    
    def test_many_layers_extraction(self):
        """Test extracting features from many layers."""
        batch_size = 2
        images = torch.rand(batch_size, 3, 336, 336)
        
        mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
        mock_processor = self.create_mock_processor()
        feature_layers = [6, 12, 18, 23]  # 4 layers
        
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            feature_layers=feature_layers,
            device=torch.device("cpu"),
        )
        
        expected_dim = 768 * len(feature_layers)
        self.assertEqual(features.shape, (batch_size, expected_dim))
    
    def test_different_hidden_sizes(self):
        """Test with different CLIP model hidden sizes."""
        hidden_sizes = [512, 768, 1024]
        batch_size = 2
        
        for hidden_size in hidden_sizes:
            images = torch.rand(batch_size, 3, 336, 336)
            
            mock_model = self.create_mock_clip_model(
                hidden_size=hidden_size, num_layers=24
            )
            mock_processor = self.create_mock_processor()
            feature_layers = [12, 18, 23]
            
            features = extract_clip_features(
                images=images,
                clip_model=mock_model,
                processor=mock_processor,
                feature_layers=feature_layers,
                device=torch.device("cpu"),
            )
            
            expected_dim = hidden_size * len(feature_layers)
            self.assertEqual(features.shape, (batch_size, expected_dim))
    
    def test_batch_size_flexibility(self):
        """Test that function handles different batch sizes."""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            images = torch.rand(batch_size, 3, 336, 336)
            
            mock_model = self.create_mock_clip_model(hidden_size=768, num_layers=24)
            mock_processor = self.create_mock_processor()
            
            features = extract_clip_features(
                images=images,
                clip_model=mock_model,
                processor=mock_processor,
                feature_layers=[12, 18, 23],
                device=torch.device("cpu"),
            )
            
            self.assertEqual(features.shape, (batch_size, 768 * 3))
    
    def test_cls_token_extraction(self):
        """Test that CLS token (first token) is correctly extracted."""
        batch_size = 2
        images = torch.rand(batch_size, 3, 336, 336)
        hidden_size = 768
        
        # Create a mock with controlled outputs
        mock_model = MagicMock()
        mock_vision_model = MagicMock()
        
        # Create specific hidden states where CLS token is identifiable
        def mock_forward(pixel_values, output_hidden_states=False):
            batch_size = pixel_values.shape[0]
            seq_len = 50
            
            hidden_states = []
            for layer_idx in range(25):
                layer_output = torch.randn(batch_size, seq_len, hidden_size)
                # Make CLS token (position 0) have a specific pattern
                layer_output[:, 0, :] = layer_idx * 10.0
                hidden_states.append(layer_output)
            
            outputs = MagicMock()
            outputs.hidden_states = tuple(hidden_states)
            return outputs
        
        mock_vision_model.side_effect = mock_forward
        mock_model.vision_model = mock_vision_model
        mock_processor = self.create_mock_processor()
        
        features = extract_clip_features(
            images=images,
            clip_model=mock_model,
            processor=mock_processor,
            feature_layers=[12, 18, 23],
            device=torch.device("cpu"),
        )
        
        # Features should be concatenation of CLS tokens
        # Check that values match expected pattern
        self.assertEqual(features.shape, (batch_size, hidden_size * 3))
        
        # First layer features should be ~12*10 = 120
        first_layer_mean = features[:, :hidden_size].mean().item()
        self.assertAlmostEqual(first_layer_mean, 12.0 * 10.0, places=0)


class TestRewardModelIntegration(unittest.TestCase):
    """Integration tests combining RewardModel and extract_clip_features."""
    
    def create_mock_clip_model(self, hidden_size=768, num_layers=24):
        """Create a mock CLIP model for testing."""
        mock_model = MagicMock()
        mock_vision_model = MagicMock()
        
        def mock_forward(pixel_values, output_hidden_states=False):
            batch_size = pixel_values.shape[0]
            seq_len = 50
            
            hidden_states = tuple(
                torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
            )
            
            outputs = MagicMock()
            outputs.hidden_states = hidden_states
            return outputs
        
        mock_vision_model.side_effect = mock_forward
        mock_model.vision_model = mock_vision_model
        
        return mock_model
    
    def create_mock_processor(self):
        """Create a mock CLIP processor."""
        mock_processor = MagicMock()
        
        def mock_process(images, return_tensors="pt"):
            batch_size = len(images)
            return {"pixel_values": torch.randn(batch_size, 3, 336, 336)}
        
        mock_processor.side_effect = mock_process
        return mock_processor
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from images to reward scores."""
        # Create reward model
        feature_dim = 768
        num_layers = 3
        reward_model = RewardModel(
            feature_dim=feature_dim,
            num_layers=num_layers,
            hidden_dim=256,
            dropout=0.1,
        )
        reward_model.eval()
        
        # Create mock CLIP components
        mock_clip_model = self.create_mock_clip_model(
            hidden_size=feature_dim, num_layers=24
        )
        mock_processor = self.create_mock_processor()
        
        # Test images
        batch_size = 4
        images = torch.rand(batch_size, 3, 336, 336)
        
        # Extract features
        with torch.no_grad():
            features = extract_clip_features(
                images=images,
                clip_model=mock_clip_model,
                processor=mock_processor,
                feature_layers=[12, 18, 23],
                device=torch.device("cpu"),
            )
            
            # Get reward scores
            scores = reward_model(features)
        
        # Verify output
        self.assertEqual(scores.shape, (batch_size,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))
    
    def test_dimension_mismatch_detection(self):
        """Test that dimension mismatch between features and model is detected."""
        # Create reward model expecting 768*3 = 2304 dims
        reward_model = RewardModel(
            feature_dim=768,
            num_layers=3,
            hidden_dim=256,
            dropout=0.1,
        )
        reward_model.eval()
        
        # Create mock CLIP with different hidden size
        mock_clip_model = self.create_mock_clip_model(
            hidden_size=1024, num_layers=24  # Different from reward model!
        )
        mock_processor = self.create_mock_processor()
        
        batch_size = 2
        images = torch.rand(batch_size, 3, 336, 336)
        
        with torch.no_grad():
            features = extract_clip_features(
                images=images,
                clip_model=mock_clip_model,
                processor=mock_processor,
                feature_layers=[12, 18, 23],
                device=torch.device("cpu"),
            )
            
            # This should raise an error due to dimension mismatch
            with self.assertRaises(RuntimeError):
                scores = reward_model(features)


if __name__ == "__main__":
    unittest.main()
