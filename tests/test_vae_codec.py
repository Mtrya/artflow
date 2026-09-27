"""VAE decoding accepts solver latents independently of the VAE's dtype."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.utils.vae_codec import decode_latents


@pytest.mark.parametrize("latent_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vae_dtype", [torch.float32, torch.bfloat16])
def test_decode_latents_casts_for_vae_and_returns_rgb_images(latent_dtype, vae_dtype):
    class StubVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(16, 3, kernel_size=1, dtype=vae_dtype)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.bias.zero_()
                for channel in range(3):
                    self.conv.weight[channel, channel, 0, 0, 0] = 1

        def decode(self, latents):
            assert latents.shape == (2, 16, 1, 2, 3)
            return SimpleNamespace(sample=self.conv(latents))

    latents = torch.zeros(2, 16, 2, 3, dtype=latent_dtype)
    latents[0, :3] = torch.tensor([-2, 0, 2], dtype=latent_dtype)[:, None, None]
    latents[1, :3] = torch.tensor([1, -1, 0], dtype=latent_dtype)[:, None, None]

    images = decode_latents(latents, StubVAE())

    assert len(images) == 2
    for image, rgb in zip(images, ([0, 127, 255], [255, 0, 127])):
        assert image.mode == "RGB"
        assert image.size == (3, 2)
        expected = np.broadcast_to(np.array(rgb, dtype=np.uint8), (2, 3, 3))
        np.testing.assert_array_equal(np.asarray(image), expected)
