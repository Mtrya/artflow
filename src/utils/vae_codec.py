"""
Helper functions to encode image or decode latents using Qwen-VAE
"""

from typing import List

from diffusers import AutoencoderKLQwenImage
import numpy as np
import torch
from PIL import Image


def encode_image(
    images: List[Image.Image], model: AutoencoderKLQwenImage
) -> torch.Tensor:
    """
    Encode a batch of PIL Images to latents.

    Args:
        images: List of PIL Images
        model: Qwen VAE model

    Returns:
        Latents tensor of shape [batch_size, channel, height, width]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Convert PIL images to tensors and normalize to [-1, 1]
    tensors = []
    for img in images:
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(dtype)
        tensor = tensor / 127.5 - 1.0
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)  # [b, c, h, w]

    # Add temporal dimension [b, c, 1, h, w]
    batch = batch.unsqueeze(2)

    with torch.no_grad():
        latents = model.encode(batch).latent_dist.sample()

    # Remove temporal dimension [b, c, h, w]
    latents = latents.squeeze(2)

    return latents


def decode_latents(
    latents: torch.Tensor, model: AutoencoderKLQwenImage
) -> List[Image.Image]:
    """
    Decode latents to a batch of PIL Images.

    Args:
        latents: Latents tensor of shape [B, C, H, W]
        model: Qwen VAE model

    Returns:
        List of PIL Images
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).device

    # Add temporal dimension [b, c, 1, h, w]
    latents = latents.unsqueeze(2).to(device).to(dtype)

    with torch.no_grad():
        reconstructed = model.decode(latents).sample

    # Remove temporal dimension [b, c, h, w]
    reconstructed = reconstructed.squeeze(2)

    # Denormalize from [-1, 1] to [0, 255] and convert to PIL
    images = []
    for tensor in reconstructed:
        tensor = torch.clamp(tensor, -1.0, 1.0)
        tensor = (tensor + 1.0) * 127.5

        img_array = tensor.permute(1, 2, 0).cpu().to(torch.uint8).numpy()
        images.append(Image.fromarray(img_array))

    return images


if __name__ == "__main__":
    vae = AutoencoderKLQwenImage.from_pretrained(
        "REPA-E/e2e-qwenimage-vae", torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    test_image = Image.open("test_image.png")

    latents = encode_image([test_image], vae)
    print(f"Latents shape: {latents.shape}")
    print(f"Latents dtype: {latents.dtype}")
    print(f"Latents device: {latents.device}")

    # Decode back to images
    reconstructed_images = decode_latents(latents, vae)
    print(f"Reconstructed {len(reconstructed_images)} images")

    # Save reconstructed image
    reconstructed_images[0].save("test_reconstruction.png")
    print("Saved reconstructed image to test_reconstruction.png")


def get_vae_stats(
    vae_path: str, device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get latent mean and std from VAE config.

    Args:
        vae_path: Path to VAE model
        device: Device to put tensors on

    Returns:
        (mean, std) tuple of tensors with shape [1, C, 1, 1]
    """
    # Load config only to be fast
    from transformers import PretrainedConfig

    try:
        config = PretrainedConfig.from_pretrained(vae_path)
    except Exception:
        # Fallback to loading full model if config load fails (e.g. local path issues)
        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        config = vae.config
        del vae

    if hasattr(config, "latents_mean") and config.latents_mean is not None:
        mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1)
    else:
        mean = torch.zeros(1, 16, 1, 1)

    if hasattr(config, "latents_std") and config.latents_std is not None:
        std = torch.tensor(config.latents_std).view(1, -1, 1, 1)
    else:
        std = torch.ones(1, 16, 1, 1)

    if device:
        mean = mean.to(device)
        std = std.to(device)

    # print(
    #    mean
    # )  # [-0.0418, -0.0157, -0.0053, -0.0127, -0.0445, 0.0351, -0.0367, 0.0239, -0.0363, -0.0044, 0.0380, -0.0015, -0.0821, -0.1100, -0.0483, 0.0077]
    # print(
    #    std
    # )  # [2.3349, 2.3665, 2.3873, 2.3958, 2.3773, 2.4054, 2.3908, 2.3725, 2.3623, 2.3824, 2.4043, 2.3669, 2.3800, 2.3779, 2.3889, 2.3639]
    return mean, std
