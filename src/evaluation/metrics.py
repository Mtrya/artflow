"""
Pure metric calculation functions for image evaluation.

Functions:
- calculate_fid: Frechet Inception Distance
- calculate_kid: Kernel Inception Distance
- calculate_clip_score: CLIP Score for text-image alignment
"""

import gc
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore


def aspect_resize_crop(images: torch.Tensor, size: int = 299) -> torch.Tensor:
    """
    Aspect-preserving resize (short side -> size) + center crop to size x size.

    Replaces the old distorting squash to 299x299 so variable-aspect buckets
    keep their geometry. Input [B, C, H, W] float in [0, 1].
    """
    _, _, h, w = images.shape
    if h == size and w == size:
        return images
    scale = size / min(h, w)
    new_h = max(size, int(round(h * scale)))
    new_w = max(size, int(round(w * scale)))
    out = F.interpolate(
        images, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
    )
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return out[:, :, top : top + size, left : left + size]


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
    subset_size: int = 100,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    return_std: bool = False,
):
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

    # KID returns (mean, std)
    mean, std = kid.compute()
    score = mean.item()

    # Cleanup to save VRAM
    del kid
    gc.collect()
    torch.cuda.empty_cache()

    if return_std:
        return score, std.item()
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
