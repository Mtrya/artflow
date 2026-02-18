"""
ArtFlow Pipeline - Standalone text-to-image generation pipeline.
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from PIL import Image
import numpy as np


class ArtFlowPipelineOutput:
    """Output class for ArtFlow pipeline."""

    def __init__(self, images: List[Image.Image], nsfw_content_detected: Optional[List[bool]] = None):
        self.images = images
        self.nsfw_content_detected = nsfw_content_detected


class ArtFlowPipeline:
    """
    Pipeline for text-to-image generation using ArtFlow.

    This pipeline is self-contained and can be loaded entirely from HuggingFace Hub.

    Example:
        ```python
        pipe = ArtFlowPipeline.from_pretrained("username/artflow-stage2")
        image = pipe(
            prompt="impressionist landscape",
            height=640,
            width=640,
            guidance_scale=3.0
        ).images[0]
        ```
    """

    def __init__(
        self,
        transformer: "ArtFlow",
        vae: "AutoencoderKLQwenImage",
        text_encoder: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        vae_mean: Optional[torch.Tensor] = None,
        vae_std: Optional[torch.Tensor] = None,
        solver: str = "euler",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        offload: bool = True,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae_mean = vae_mean
        self.vae_std = vae_std
        self.solver = solver
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.offload = offload

        # Move to eval mode
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()

    def _get_autocast_context(self):
        """Get autocast context manager for inference."""
        device_type = "cuda" if "cuda" in self.device else "cpu"
        if self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type=device_type, dtype=self.dtype)
        return torch.no_grad()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "ArtFlowPipeline":
        """
        Load pipeline from HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: HF Hub repo ID (e.g., "username/artflow-stage2")
            dtype: Data type for model weights (default: bfloat16)
            device: Device to load models on (default: cuda if available)

        Returns:
            ArtFlowPipeline instance
        """
        from huggingface_hub import hf_hub_download, list_repo_files
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoencoderKLQwenImage

        dtype = kwargs.get("dtype", torch.bfloat16)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        offload = kwargs.get("offload", True)

        # Download all source files from the repo
        repo_files = list_repo_files(pretrained_model_name_or_path)
        source_files = [f for f in repo_files if f.startswith("artflow/")]

        # Create temp directory and download sources
        temp_dir = Path(tempfile.gettempdir()) / f"artflow_{pretrained_model_name_or_path.replace('/', '_')}"
        temp_dir.mkdir(exist_ok=True)

        for file in source_files:
            hf_hub_download(pretrained_model_name_or_path, file, local_dir=temp_dir)

        # Add to path and import
        sys.path.insert(0, str(temp_dir))

        # Import after adding to path
        from artflow.models.artflow import ArtFlow
        from artflow.utils.encode_text import encode_text
        from artflow.flow.solvers import sample_ode
        from artflow.utils.vae_codec import get_vae_stats

        # Load config
        config_path = hf_hub_download(pretrained_model_name_or_path, "transformer_config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Load transformer
        transformer = ArtFlow(**config)

        # Load weights (try safetensors first, then .pt)
        try:
            weights_path = hf_hub_download(pretrained_model_name_or_path, "model.safetensors")
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        except Exception:
            weights_path = hf_hub_download(pretrained_model_name_or_path, "ema_weights.pt")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "module" in state_dict:
                state_dict = state_dict["module"]

        transformer.load_state_dict(state_dict)

        # Load VAE
        vae_repo = config.get("vae_repo", "REPA-E/e2e-qwenimage-vae")
        vae = AutoencoderKLQwenImage.from_pretrained(vae_repo, torch_dtype=dtype)

        # Load text encoder
        text_encoder_repo = config.get("text_encoder_repo", "Qwen/Qwen3-0.6B")
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_repo,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_repo)

        # Get VAE stats for denormalization
        vae_mean, vae_std = get_vae_stats(vae_repo, device=device)
        vae_mean = vae_mean.to(device=device, dtype=dtype)
        vae_std = vae_std.to(device=device, dtype=dtype)

        # Load models to appropriate device based on offload setting
        if offload:
            # Keep on CPU, offload to GPU when needed
            transformer.to(dtype=dtype)
        else:
            # Load directly to GPU
            transformer.to(device=device, dtype=dtype)
            vae.to(device=device)
            text_encoder.to(device=device)

        # Create pipeline
        pipe = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae_mean=vae_mean,
            vae_std=vae_std,
            solver=config.get("solver", "euler"),
            dtype=dtype,
            device=device,
            offload=offload,
        )

        return pipe

    @staticmethod
    def _sanitize_resolution(value: int) -> int:
        """Ensure resolution is divisible by 16."""
        value = max(64, int(value))
        return (value // 16) * 16

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        solver: Optional[str] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        num_images_per_prompt: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[ArtFlowPipelineOutput, List[Image.Image]]:
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            height: Image height (default: 640, sanitized to multiple of 16)
            width: Image width (default: 640, sanitized to multiple of 16)
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: CFG scale (default: 3.0, 1.0 = no CFG)
            negative_prompt: Negative prompt for CFG (default: empty)
            seed: Random seed for reproducibility
            solver: "euler" or "heun" (default: pipeline default)
            output_type: "pil" or "latent"
            return_dict: Return PipelineOutput if True, else images only
            num_images_per_prompt: Batch size per prompt

        Returns:
            ArtFlowPipelineOutput or list of images
        """
        # Set defaults
        height = height or 640
        width = width or 640
        solver = solver or self.solver

        # Sanitize resolutions
        height = self._sanitize_resolution(height)
        width = self._sanitize_resolution(width)

        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # --- Stage 1: Text encoding (text_encoder on GPU) ---
        do_cfg = guidance_scale > 1.0

        if self.offload:
            self.text_encoder.to(self.device)
        if do_cfg:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            all_prompts = prompt + negative_prompt
            text_emb, text_mask, _ = self._encode_text(all_prompts)

            text_emb_cond = text_emb[:batch_size]
            text_emb_uncond = text_emb[batch_size:]
            text_mask_cond = text_mask[:batch_size]
            text_mask_uncond = text_mask[batch_size:]
        else:
            text_emb_cond, text_mask_cond = self._encode_text(prompt)[:2]
            text_emb_uncond = None
            text_mask_uncond = None
        if self.offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # --- Stage 2: Denoising (transformer on GPU) ---
        if self.offload:
            self.transformer.to(self.device)
        device = torch.device(self.device)
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        latents_shape = (
            batch_size * num_images_per_prompt,
            16,  # VAE channels
            height // 8,
            width // 8,
        )
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=self.dtype)

        if do_cfg:
            latents = torch.cat([latents, latents], dim=0)
            text_emb = torch.cat([text_emb_cond, text_emb_uncond], dim=0)
            text_mask = torch.cat([text_mask_cond, text_mask_uncond], dim=0)
        else:
            text_emb = text_emb_cond
            text_mask = text_mask_cond

        def model_fn(x, t):
            t_tensor = torch.as_tensor(t, device=x.device).expand(x.shape[0])
            return self.transformer(x, t_tensor, text_emb, txt_mask=text_mask)

        from artflow.flow.solvers import sample_ode

        with self._get_autocast_context():
            latents = sample_ode(
                model_fn,
                latents,
                steps=num_inference_steps,
                solver=solver,
                device=self.device,
                progress_callback=progress_callback,
            )

        if do_cfg:
            latents_cond, latents_uncond = latents.chunk(2)
            latents = latents_uncond + guidance_scale * (latents_cond - latents_uncond)
        if self.offload:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

        # --- Stage 3: VAE decode (vae on GPU) ---
        if output_type == "latent":
            images = latents
        else:
            if self.offload:
                self.vae.to(self.device)
            images = self._decode_latents(latents)
            if self.offload:
                self.vae.to("cpu")
                torch.cuda.empty_cache()

        if not return_dict:
            return images

        return ArtFlowPipelineOutput(images=images)

    def _encode_text(
        self, prompts: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode text prompts using the text encoder."""
        from artflow.utils.encode_text import encode_text

        pooling = self.transformer.conditioning_scheme == "fused"
        txt_emb, txt_mask, txt_pooled = encode_text(
            prompts, self.text_encoder, self.tokenizer, pooling=pooling
        )

        txt_emb = txt_emb.to(device=self.device, dtype=self.dtype)
        txt_mask = txt_mask.to(device=self.device)
        if txt_pooled is not None:
            txt_pooled = txt_pooled.to(device=self.device, dtype=self.dtype)

        return txt_emb, txt_mask, txt_pooled

    def _decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        """Decode VAE latents to PIL images."""
        latents = latents.to(device=self.device, dtype=self.dtype)

        # Denormalize
        if self.vae_mean is not None and self.vae_std is not None:
            latents = latents * self.vae_std + self.vae_mean

        # Add temporal dim for video VAE
        latents = latents.unsqueeze(2)

        # Decode
        with torch.no_grad():
            decoded = self.vae.decode(latents).sample

        decoded = decoded.squeeze(2)

        # Convert to PIL
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in decoded]
        return images
