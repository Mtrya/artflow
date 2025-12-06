"""
Pure metric calculation functions for image evaluation.

Functions:
- calculate_fid: Frechet Inception Distance
- calculate_kid: Kernel Inception Distance
- calculate_clip_score: CLIP Score for text-image alignment
- calculate_reward_score: Reward model score for artistic preference
"""

import gc
from pathlib import Path
from typing import List, Optional, Union

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore


def calculate_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature: int = 2048,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Calculate FID score between real and fake images.

    Args:
        real_images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        fake_images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        feature: Inception feature dimension (64, 192, 768, 2048)
        device: Device to run calculation on
        batch_size: Batch size for processing images to avoid OOM

    Returns:
        FID score (float)
    """
    if device is None:
        device = real_images.device

    fid = FrechetInceptionDistance(feature=feature).to(device)

    def update_metric(images, is_real):
        nonlocal fid
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            # Ensure images are uint8 for torchmetrics FID
            if batch.dtype != torch.uint8:
                if batch.max() <= 1.0:
                    batch = (batch * 255).to(torch.uint8)
                else:
                    batch = batch.to(torch.uint8)
            batch = batch.to(device)
            fid.update(batch, real=is_real)

    update_metric(real_images, True)
    update_metric(fake_images, False)

    score = fid.compute().item()

    # Cleanup to save VRAM
    del fid
    gc.collect()
    torch.cuda.empty_cache()

    return score


def calculate_kid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature: int = 2048,
    subset_size: int = 50,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Calculate KID score between real and fake images.

    Args:
        real_images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        fake_images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        feature: Inception feature dimension (64, 192, 768, 2048)
        subset_size: Number of samples to use for the polynomial kernel estimation
        device: Device to run calculation on
        batch_size: Batch size for processing images to avoid OOM

    Returns:
        KID score (float)
    """
    if device is None:
        device = real_images.device

    # Adjust subset_size if we have fewer samples
    n_samples = min(real_images.shape[0], fake_images.shape[0])
    if subset_size > n_samples:
        subset_size = n_samples

    kid = KernelInceptionDistance(feature=feature, subset_size=subset_size).to(device)

    def update_metric(images, is_real):
        nonlocal kid
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            # Ensure images are uint8 for torchmetrics KID
            if batch.dtype != torch.uint8:
                if batch.max() <= 1.0:
                    batch = (batch * 255).to(torch.uint8)
                else:
                    batch = batch.to(torch.uint8)
            batch = batch.to(device)
            kid.update(batch, real=is_real)

    update_metric(real_images, True)
    update_metric(fake_images, False)

    # KID returns (mean, std), we take the mean
    score = kid.compute()[0].item()

    # Cleanup to save VRAM
    del kid
    gc.collect()
    torch.cuda.empty_cache()

    return score


def calculate_clip_score(
    images: torch.Tensor,
    prompts: Union[str, List[str]],
    model_name_or_path: str = "openai/clip-vit-base-patch16", # or "OFA-Sys/chinese-clip-vit-base-patch16"
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Calculate CLIP score for images and prompts.

    Args:
        images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        prompts: Single string or list of strings matching batch size
        model_name_or_path: CLIP model name
        device: Device to run calculation on
        batch_size: Batch size for processing images to avoid OOM

    Returns:
        CLIP score (float)
    """
    if device is None:
        device = images.device

    metric = CLIPScore(model_name_or_path=model_name_or_path).to(device)

    if isinstance(prompts, str):
        prompts = [prompts] * images.shape[0]

    for i in range(0, len(images), batch_size):
        img_batch = images[i : i + batch_size]
        prompt_batch = prompts[i : i + batch_size]

        # Ensure images are uint8 [0, 255] for CLIPScore
        if img_batch.dtype != torch.uint8:
            if img_batch.max() <= 1.0:
                img_batch = (img_batch * 255).to(torch.uint8)
            else:
                img_batch = img_batch.to(torch.uint8)

        img_batch = img_batch.to(device)
        metric.update(img_batch, prompt_batch)

    score = metric.compute()
    val = score.item()

    # Cleanup
    del metric
    gc.collect()
    torch.cuda.empty_cache()

    return val


def _calculate_rm_score(
    clip_features: torch.Tensor,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Calculate reward model score for artistic preference using pre-extracted features.
    
    Args:
        clip_features: [B, hidden_size * num_layers] pre-extracted CLIP features
        checkpoint_path: Path to trained reward model checkpoint
        device: Device to run calculation on
        batch_size: Batch size for processing
        
    Returns:
        Average reward score (float) across all images
    """
    if device is None:
        device = clip_features.device
    
    from ..models.reward_model import RewardModel
    
    # Load checkpoint config
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    import json
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create model
    model = RewardModel(
        feature_dim=config["feature_dim"],
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        dropout=config.get("dropout", 0.1),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Calculate scores
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(clip_features), batch_size):
            batch_features = clip_features[i : i + batch_size].to(device)
            scores = torch.clamp(model(batch_features), 0, 1)
            all_scores.append(scores)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return torch.cat(all_scores, dim=0).mean().item()


def calculate_combined_reward(
    images: torch.Tensor,
    prompts: Union[str, List[str]],
    reward_checkpoint: str,
    clip_model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14-336px",
    feature_layers: Optional[List[int]] = None,
    clip_weight: float = 0.7,
    rm_weight: float = 0.3,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Calculate combined reward: weighted sum of CLIP score and reward model.
    
    This is the reward function for Flow-GRPO training.
    Uses a single CLIP model for both CLIP score and reward model to avoid duplication.
    
    Args:
        images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        prompts: Single string or list of strings matching batch size
        reward_checkpoint: Path to trained reward model checkpoint
        clip_model_name: CLIP model for both text-image alignment and reward features
        feature_layers: Layers to extract features from (e.g., [12, 18, 23])
        clip_weight: Weight for CLIP score (default: 0.7)
        rm_weight: Weight for reward model score (default: 0.3)
        device: Device to run calculation on
        batch_size: Batch size for processing
        
    Returns:
        Combined reward score (float)
    """
    if device is None:
        device = images.device
    
    from transformers import ChineseCLIPModel, CLIPProcessor
    from ..models.reward_model import extract_clip_features
    
    # Load CLIP model once
    clip_model = ChineseCLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    
    # Load reward model config
    checkpoint_dir = Path(reward_checkpoint).parent
    config_path = checkpoint_dir / "config.json"
    
    if config_path.exists():
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            if feature_layers is None:
                feature_layers = config.get("feature_layers", [12, 18, 23])
    else:
        if feature_layers is None:
            feature_layers = [12, 18, 23]
    
    # Prepare prompts
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)

    # Check if images are in [0, 255] range
    if images.min() < 0:
        print(f"Warning: Images are not in [0, 1] or [0, 255] range. Max: {images.max()}, Min: {images.min()}")
    # Convert images to [0, 255] range
    if images.max() <= 1:
        images = (images * 255.0).to(torch.uint8)
    
    # Process in batches
    clip_scores = []
    rm_features_list = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            img_batch = images[i : i + batch_size].to(device)
            prompt_batch = prompts[i : i + batch_size]
            
            # Extract multi-layer image features
            img_features_multilayer = extract_clip_features(
                img_batch, clip_model, clip_processor, feature_layers, device
            )
            
            # Store for reward model
            rm_features_list.append(img_features_multilayer.cpu())
            
            # Extract last layer features for CLIP score
            # Note: This assumes the last layer (e.g., 23) is in feature_layers
            hidden_size = img_features_multilayer.shape[-1] // len(feature_layers)
            last_layer_features = img_features_multilayer[:, -hidden_size:]  # [B, hidden_size]
            
            # Apply visual projection to get final embeddings
            img_embeddings = clip_model.visual_projection(last_layer_features)
            img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)
            
            # Get text embeddings
            text_inputs = clip_processor(
                text=prompt_batch, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            txt_embeddings = clip_model.get_text_features(**text_inputs)
            txt_embeddings = txt_embeddings / txt_embeddings.norm(p=2, dim=-1, keepdim=True)
            
            # Calculate CLIP score: 100 * cosine similarity
            batch_clip_scores = 100 * (img_embeddings * txt_embeddings).sum(dim=-1)
            clip_scores.append(batch_clip_scores.cpu())
    
    # Average CLIP score
    all_clip_scores = torch.cat(clip_scores, dim=0)
    clip_score = all_clip_scores.mean().item()
    
    # Concatenate all features for reward model
    all_rm_features = torch.cat(rm_features_list, dim=0).to(device)
    
    # Calculate reward model score
    rm_score = _calculate_rm_score(
        all_rm_features, reward_checkpoint, device, batch_size
    )
    
    # Cleanup
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Normalize CLIP score to [0, 1] range
    clip_score_normalized = clip_score / 100.0
    
    # Combine with weights
    combined = clip_weight * clip_score_normalized + rm_weight * rm_score
    
    return combined

