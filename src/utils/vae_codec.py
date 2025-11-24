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
