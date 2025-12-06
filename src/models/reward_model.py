"""
Reward model for artistic preference learning.

Architecture:
- Accepts pre-extracted multi-layer CLIP features
- Lightweight MLP regression head
- Output: [0, 1] artistic quality score

Usage:
    from src.models.reward_model import RewardModel, extract_clip_features
    
    # Create model
    model = RewardModel(
        feature_dim=1024,  # CLIP hidden size
        num_layers=3,      # Number of layers to use
        hidden_dim=512,
    )
    
    # Extract features externally
    features = extract_clip_features(images, clip_model, feature_layers=[12, 18, 23])
    
    # Score
    score = model(features)  # [B, 3*1024] -> [B]
"""

from typing import List, Optional

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    Artistic preference reward model.
    
    Accepts pre-extracted multi-layer CLIP features for robust
    style/texture perception, followed by lightweight MLP head.
    """

    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Dimension of features from each layer
            num_layers: Number of layers features are extracted from
            hidden_dim: Hidden dimension of MLP head
            dropout: Dropout rate in MLP
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        total_feature_dim = feature_dim * num_layers
        
        # MLP regression head
        self.head = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output [0, 1]
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [B, feature_dim * num_layers] pre-extracted features
            
        Returns:
            scores: [B] quality scores in [0, 1]
        """
        scores = self.head(features).squeeze(-1)  # [B]
        return scores
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def extract_clip_features(
    images: torch.Tensor,
    clip_model: nn.Module,
    processor: Optional[any] = None,
    feature_layers: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract multi-layer features from CLIP vision model.
    
    Args:
        images: [B, 3, H, W] images in [0, 1] or [0, 255] range
        clip_model: CLIP model (e.g., ChineseCLIPModel)
        processor: CLIP processor (optional, will use default normalization if None)
        feature_layers: List of layer indices to extract. If None, uses [12, 18, 23]
        device: Device to run on
        
    Returns:
        features: [B, hidden_size * num_layers] concatenated multi-layer features
    """
    if device is None:
        device = images.device
    
    if feature_layers is None:
        feature_layers = [12, 18, 23]
    
    # Check if images are in [0, 255] range
    if images.min() < 0:
        print(f"Warning: Images are not in [0, 1] or [0, 255] range. Max: {images.max()}, Min: {images.min()}")
    # Convert images to [0, 255] range
    if images.max() <= 1:
        images = (images * 255.0).to(torch.uint8)

    # Preprocess images
    # Note: CLIPProcessor expects images in range [0, 255] by default if no `do_rescale=False` flag is set
    pixel_values = processor(images=[img.cpu() for img in images], return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device)
    
    # Extract features
    outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of [B, seq_len, hidden_size]
    
    # Extract CLS token from specified layers
    features = []
    for layer_idx in feature_layers:
        layer_output = hidden_states[layer_idx]  # [B, seq_len, hidden_size]
        cls_token = layer_output[:, 0, :]  # [B, hidden_size]
        features.append(cls_token)
    
    # Concatenate
    features = torch.cat(features, dim=-1)  # [B, hidden_size * num_layers]
    return features


if __name__ == "__main__":
    def test_reward_model():
        """Test reward model instantiation and forward pass."""
        print("Testing RewardModel...")

        from transformers import ChineseCLIPModel, CLIPProcessor

        # Load CLIP model
        clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
        processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")

        # Create reward model
        model = RewardModel(
            feature_dim=1024,  # CLIP-Large hidden size
            num_layers=3,
            hidden_dim=512,
        )

        print(f"Trainable params: {model.get_num_trainable_params():,}")
        print(f"Total params: {model.get_num_total_params():,}")

        # Test forward pass
        batch_size = 4
        images = torch.randn(batch_size, 3, 336, 336)

        with torch.no_grad():
            # Extract features
            features = extract_clip_features(images, clip_model, processor, feature_layers=[12, 18, 23])
            print(f"Features shape: {features.shape}")
            
            # Score
            scores = model(features)
            print(f"Scores shape: {scores.shape}")
            print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")

        print("✓ RewardModel test passed!")
    test_reward_model()
