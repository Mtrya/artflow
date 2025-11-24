"""
Implement evaluation methods:
1. FID score
2. CLIP score
3. Gather and plot generated images for human eval
"""

import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from typing import List, Optional, Union, Dict, Any
import os
import numpy as np
import gc
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
import swanlab

transformers.logging.set_verbosity_error()


def calculate_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature: int = 2048,
    device: Optional[torch.device] = None,
) -> float:
    """
    Calculate FID score between real and fake images.

    Args:
        real_images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        fake_images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255] (uint8)
        feature: Inception feature dimension (64, 192, 768, 2048)
        device: Device to run calculation on

    Returns:
        FID score (float)
    """
    if device is None:
        device = real_images.device

    fid = FrechetInceptionDistance(feature=feature).to(device)

    # Ensure images are uint8 for torchmetrics FID
    if real_images.dtype != torch.uint8:
        if real_images.max() <= 1.0:
            real_images = (real_images * 255).to(torch.uint8)
        else:
            real_images = real_images.to(torch.uint8)

    if fake_images.dtype != torch.uint8:
        if fake_images.max() <= 1.0:
            fake_images = (fake_images * 255).to(torch.uint8)
        else:
            fake_images = fake_images.to(torch.uint8)

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    score = fid.compute().item()

    # Cleanup to save VRAM
    del fid
    gc.collect()
    torch.cuda.empty_cache()

    return score


def calculate_clip_score(
    images: torch.Tensor,
    prompts: Union[str, List[str]],
    model_name_or_path: str = "openai/clip-vit-base-patch16",
    device: Optional[torch.device] = None,
) -> float:
    """
    Calculate CLIP score for images and prompts.

    Args:
        images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255]
        prompts: Single string or list of strings matching batch size
        model_name_or_path: CLIP model name
        device: Device to run calculation on

    Returns:
        CLIP score (float)
    """
    if device is None:
        device = images.device

    # Suppress the specific warning about use_fast if possible,
    # but we already set verbosity to error globally for transformers.

    metric = CLIPScore(model_name_or_path=model_name_or_path).to(device)

    # Ensure images are uint8 [0, 255] for CLIPScore
    if images.dtype != torch.uint8:
        if images.max() <= 1.0:
            images = (images * 255).to(torch.uint8)
        else:
            images = images.to(torch.uint8)

    if isinstance(prompts, str):
        prompts = [prompts] * images.shape[0]

    score = metric(images, prompts)

    val = score.item()

    # Cleanup
    del metric
    gc.collect()
    torch.cuda.empty_cache()

    return val


def make_image_grid(
    images: torch.Tensor,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    value_range: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Create a grid of images and optionally save it.

    Args:
        images: Tensor of shape [B, C, H, W]
        rows: Number of rows (optional)
        cols: Number of columns (optional)
        save_path: Path to save the grid image
        normalize: Whether to normalize images to [0, 1]
        value_range: Range of values in input images (min, max)

    Returns:
        Grid tensor
    """
    if rows is None and cols is None:
        nrow = int(np.ceil(np.sqrt(images.shape[0])))
    elif cols is not None:
        nrow = cols
    else:
        nrow = int(np.ceil(images.shape[0] / rows))

    grid = torchvision.utils.make_grid(
        images, nrow=nrow, normalize=normalize, value_range=value_range, padding=2
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(grid, save_path)

    return grid


def visualize_denoising(
    intermediate_steps: List[torch.Tensor], save_path: str, num_steps_to_show: int = 10
):
    """
    Visualize the denoising process by selecting a subset of steps.

    Args:
        intermediate_steps: List of tensors [B, C, H, W] from the sampling process
        save_path: Path to save the visualization
        num_steps_to_show: Number of steps to display
    """
    total_steps = len(intermediate_steps)
    if total_steps < num_steps_to_show:
        indices = list(range(total_steps))
    else:
        indices = np.linspace(0, total_steps - 1, num_steps_to_show, dtype=int).tolist()

    selected_steps = [intermediate_steps[i] for i in indices]

    # Take the first sample from the batch for visualization
    first_sample_steps = [step[0] for step in selected_steps]  # List of [C, H, W]

    # Stack them: [Num_steps, C, H, W]
    stacked = torch.stack(first_sample_steps)

    # Make grid: 1 row, Num_steps columns
    make_image_grid(
        stacked,
        rows=1,
        cols=len(selected_steps),
        save_path=save_path,
        normalize=True,
        value_range=(-1, 1),
    )


def run_evaluation_uncond(
    accelerator: Accelerator,
    model: torch.nn.Module,
    args: Any,
    epoch: int,
    train_dataloader: DataLoader,
    algorithm: Any,
    sample_ode_fn: Any,
    ScoreMatchingODE_cls: Any,
) -> dict:
    """
    Lightweight evaluation pipeline for unconditional generation:
    1. Generate samples
    2. Calculate FID and CLIP scores
    3. Visualize denoising process
    4. Return metrics for Wandb and save locally
    """
    print("Running evaluation...")
    model.eval()

    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(accelerator.device)

    with torch.no_grad():
        # --- 1. Generate Samples ---
        sample_z0 = torch.randn(
            args.num_eval_samples,
            16,
            args.eval_resolution // 8,
            args.eval_resolution // 8,
            device=accelerator.device,
        )

        # Define model wrapper for solver
        def model_fn(x, t):
            if isinstance(t, float):
                t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
            else:
                t_tensor = t
            return model(x, t_tensor)

        if args.algorithm == "fm-ot":
            samples = sample_ode_fn(
                model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0
            )
        elif args.algorithm == "sm-diffusion":
            solver = ScoreMatchingODE_cls(
                beta_min=algorithm.beta_min, beta_max=algorithm.beta_max
            )
            samples = sample_ode_fn(
                model_fn,
                sample_z0,
                steps=50,
                t_start=1.0,
                t_end=0.0,
                solver_instance=solver,
            )
        else:
            samples = sample_ode_fn(
                model_fn, sample_z0, steps=50, t_start=1.0, t_end=0.0
            )

        samples = samples.to(dtype=torch.bfloat16)
        samples = samples.unsqueeze(2)
        images = vae.decode(samples).sample
        images = images.squeeze(2)

        # Save locally
        save_path = os.path.join(
            args.output_dir, f"{args.run_name}/samples_epoch_{epoch + 1:03d}.png"
        )
        _ = make_image_grid(
            images, save_path=save_path, normalize=True, value_range=(-1, 1)
        )
        print(f"Saved samples to {save_path}")

        # Log to WandB
        if accelerator.is_main_process:
            accelerator.log({"samples": swanlab.Image(save_path)}, step=epoch + 1)

        # --- 2. Evaluation Metrics ---
        metrics = {}

        # FID
        # Get real images from dataloader (just one batch for stage0)
        real_batch = next(iter(train_dataloader))
        real_latents = (
            real_batch["latents"].to(accelerator.device).to(dtype=torch.bfloat16)
        )
        real_latents = real_latents.unsqueeze(2)
        real_images = vae.decode(real_latents).sample.squeeze(2)

        # Normalize to [0, 1]
        real_images_norm = (real_images + 1) / 2
        fake_images_norm = (images + 1) / 2

        fid_score = calculate_fid(
            real_images_norm, fake_images_norm, device=accelerator.device
        )
        print(f"Epoch {epoch + 1} | FID: {fid_score:.4f}")
        metrics["fid"] = fid_score

        # CLIP Score
        prompt = "A Monet painting"
        clip_score = calculate_clip_score(
            fake_images_norm, prompt, device=accelerator.device
        )
        print(f"Epoch {epoch + 1} | CLIP Score: {clip_score:.4f}")
        metrics["clip_score"] = clip_score

        if accelerator.is_main_process and metrics:
            accelerator.log(metrics, step=epoch + 1)

        # --- 3. Denoising Visualization ---
        try:
            vis_z0 = torch.randn(
                1,
                16,
                args.eval_resolution // 8,
                args.eval_resolution // 8,
                device=accelerator.device,
            )
            if args.algorithm == "fm-ot":
                _, intermediates = sample_ode_fn(
                    model_fn,
                    vis_z0,
                    steps=20,
                    t_start=0.0,
                    t_end=1.0,
                    return_intermediates=True,
                )
            elif args.algorithm == "sm-diffusion":
                solver = ScoreMatchingODE_cls(
                    beta_min=algorithm.beta_min, beta_max=algorithm.beta_max
                )
                _, intermediates = sample_ode_fn(
                    model_fn,
                    vis_z0,
                    steps=20,
                    t_start=1.0,
                    t_end=0.0,
                    solver_instance=solver,
                    return_intermediates=True,
                )
            else:
                _, intermediates = sample_ode_fn(
                    model_fn,
                    vis_z0,
                    steps=20,
                    t_start=1.0,
                    t_end=0.0,
                    return_intermediates=True,
                )

            decoded_intermediates = []
            for lat in intermediates:
                lat = lat.to(accelerator.device).to(dtype=torch.bfloat16)
                lat = lat.unsqueeze(2)
                img = vae.decode(lat).sample.squeeze(2)
                decoded_intermediates.append(img.cpu())

            vis_path = os.path.join(
                args.output_dir, f"{args.run_name}/denoising_epoch_{epoch + 1:03d}.png"
            )
            visualize_denoising(decoded_intermediates, vis_path)
            print(f"Saved denoising visualization to {vis_path}")
            if accelerator.is_main_process:
                accelerator.log({"denoising": swanlab.Image(vis_path)}, step=epoch + 1)

        except Exception as e:
            print(f"Denoising visualization failed: {e}")

    # Cleanup VAE
    del vae
    gc.collect()
    torch.cuda.empty_cache()


def run_evaluation_light(
    accelerator: Accelerator,
    model: torch.nn.Module,
    args: Any,
    current_step: int,
    text_encoder: Any,
    processor: Any,
    pooling: bool,
    sample_ode_fn: Any,
    resolution: int = 256,
) -> Dict[str, float]:
    """
    Lightweight evaluation for online use during Stage 1/2 training.

    Args:
        accelerator: Accelerate instance
        model: Conditional ArtFlow model
        args: Training arguments
        current_step: Current training step
        text_encoder: Frozen Qwen3-VL model
        processor: AutoProcessor
        pooling: Whether to use pooled text embeddings
        sample_ode_fn: Sampling function
        resolution: Target resolution for evaluation (default: 256)

    Returns:
        Dictionary of metrics
    """
    print(f"Running light evaluation at step {current_step}...")
    model.eval()

    # Import here to avoid circular imports
    try:
        from .encode_text import encode_text
    except ImportError:
        from encode_text import encode_text

    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(accelerator.device)

    metrics = {}

    with torch.no_grad():
        from datasets import load_dataset

        # Hardcode validation set
        os.environ["HF_HUB_OFFLINE"] = "1"
        dataset = load_dataset(
            "kaupane/wikiart-captions", split="train[:args.num_eval_samples]"
        )
        del os.environ["HF_HUB_OFFLINE"]

        real_images = []
        prompts = []

        for item in dataset:
            # Image
            img = item["image"]
            # Resize to fixed resolution
            img = img.resize((resolution, resolution), Image.BICUBIC)
            real_images.append(torchvision.transforms.functional.to_tensor(img))
            prompts.append(item["qwen-direct"])

        # 2. Encode prompts
        txt, txt_mask, txt_pooled = encode_text(
            prompts, text_encoder, processor, pooling
        )

        # 3. Generate Samples
        # Sample z0 (noise) at fixed resolution
        H = W = resolution
        sample_z0 = torch.randn(
            len(prompts), 16, H // 8, W // 8, device=accelerator.device
        )

        # Model wrapper for solver
        def model_fn(x, t):
            # Expand t to batch size
            if isinstance(t, float):
                t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
            else:
                t_tensor = t
            return model(x, t_tensor, txt, txt_pooled, txt_mask)

        # Sample using fm-ot defaults (t_start=0.0, t_end=1.0)
        samples = sample_ode_fn(model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0)

        # Decode
        samples = samples.to(dtype=torch.bfloat16)
        samples = samples.unsqueeze(2)
        images = vae.decode(samples).sample
        images = images.squeeze(2)  # [B, C, H, W]

        # Save locally
        save_path = os.path.join(
            args.output_dir, f"{args.run_name}/samples_step_{current_step:06d}.png"
        )
        _ = make_image_grid(
            images, save_path=save_path, normalize=True, value_range=(-1, 1)
        )
        print(f"Saved samples to {save_path}")

        # Log to WandB
        if accelerator.is_main_process:
            accelerator.log({"samples": swanlab.Image(save_path)}, step=current_step)

        # 4. Calculate Metrics
        fake_images_norm = (images + 1) / 2
        fake_images_norm = torch.clamp(fake_images_norm, 0, 1)

        # FID (if real images available)
        if real_images:
            real_images_tensor = torch.stack(real_images).to(accelerator.device)
            # Ensure real images are [0, 1]
            if real_images_tensor.max() > 1.0:
                real_images_tensor = real_images_tensor / 255.0

            fid_score = calculate_fid(
                real_images_tensor, fake_images_norm, device=accelerator.device
            )
            print(f"Step {current_step} | FID: {fid_score:.4f}")
            metrics["fid"] = fid_score

        # CLIP Score
        clip_score = calculate_clip_score(
            fake_images_norm, prompts, device=accelerator.device
        )
        print(f"Step {current_step} | CLIP Score: {clip_score:.4f}")
        metrics["clip_score"] = clip_score

        if accelerator.is_main_process and metrics:
            accelerator.log(metrics, step=current_step)

    # Cleanup
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def run_evaluation_heavy(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    dataset_name: str,
    vae_path: str,
    text_encoder_path: str,
    pooling: bool,
    output_dir: str,
    sample_ode_fn: Any,
    num_samples: int = 2000,
    batch_size: int = 32,
    split: str = "validation",
    device: str = "cuda:0",
) -> Dict[str, float]:
    """
    Large-scale evaluation for offline use.

    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration dictionary
        dataset_name: Dataset name
        vae_path: Path to VAE
        text_encoder_path: Path to text encoder
        pooling: Whether to use pooled text embeddings
        output_dir: Output directory
        sample_ode_fn: Sampling function
        num_samples: Number of samples to generate
        batch_size: Batch size
        split: Dataset split
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    print(f"Running heavy evaluation on {checkpoint_path}...")
    os.makedirs(output_dir, exist_ok=True)

    # Import here
    try:
        from .encode_text import encode_text
        from ..models.artflow import ArtFlow
    except ImportError:
        from encode_text import encode_text
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
        from src.models.artflow import ArtFlow

    # 1. Load Models
    print("Loading models...")
    # Load ArtFlow
    model = ArtFlow(**model_config)
    # Load state dict (handling potential accelerator wrapping)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load VAE
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)

    # Load Text Encoder
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        text_encoder_path,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(text_encoder_path)

    # 2. Load Dataset
    print(f"Loading dataset {dataset_name}...")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    # Select random subset if dataset is larger than num_samples
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    else:
        num_samples = len(dataset)

    # 3. Generation Loop
    print(f"Generating {num_samples} samples...")
    generated_images_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(generated_images_dir, exist_ok=True)

    all_prompts = []
    all_fake_paths = []

    # We need real images for FID
    # Save real images to a temp dir for torchmetrics FID if needed,
    # or keep in memory if fits. For 2000 images, memory is fine.
    real_images_list = []
    fake_images_list = []

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
            # Get prompts
            if "caption" in batch:
                prompts = batch["caption"]
            elif "text" in batch:
                prompts = batch["text"]
            else:
                # Fallback if no caption column found, though unlikely for this project
                prompts = [""] * len(batch["image"])

            all_prompts.extend(prompts)

            # Store real images
            # Assuming batch["image"] is list of PIL images or tensor
            if isinstance(batch["image"], list):
                # Convert PIL to tensor
                imgs = [
                    torchvision.transforms.functional.to_tensor(img)
                    for img in batch["image"]
                ]
                imgs = torch.stack(imgs)
            else:
                imgs = batch["image"]
            real_images_list.append(imgs)

            # Encode text
            txt, txt_mask, txt_pooled = encode_text(
                prompts, text_encoder, processor, pooling
            )

            # Sample
            # Use resolution from first real image or config
            # Assuming square 256 for now or from config if available
            # But ArtFlow supports variable resolution.
            # For evaluation, let's stick to a fixed resolution like 256x256 for consistency
            H = W = 256

            sample_z0 = torch.randn(len(prompts), 16, H // 8, W // 8, device=device)

            def model_fn(x, t):
                if isinstance(t, float):
                    t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
                else:
                    t_tensor = t
                return model(x, t_tensor, txt, txt_pooled, txt_mask)

            samples = sample_ode_fn(
                model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0
            )

            # Decode
            samples = samples.to(dtype=torch.bfloat16)
            samples = samples.unsqueeze(2)
            images = vae.decode(samples).sample
            images = images.squeeze(2)  # [B, C, H, W]

            # Normalize to [0, 1]
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)

            fake_images_list.append(images.cpu())

            # Save images
            for j, img in enumerate(images):
                idx = i * batch_size + j
                save_path = os.path.join(generated_images_dir, f"sample_{idx:05d}.png")
                torchvision.utils.save_image(img, save_path)
                all_fake_paths.append(save_path)

    # 4. Calculate Metrics
    print("Calculating metrics...")

    # Concatenate all images
    real_images_all = torch.cat(real_images_list, dim=0)
    fake_images_all = torch.cat(fake_images_list, dim=0)

    # Resize real images to match fake images if needed
    if real_images_all.shape[-2:] != fake_images_all.shape[-2:]:
        real_images_all = torch.nn.functional.interpolate(
            real_images_all,
            size=fake_images_all.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

    # FID
    # Process in chunks to avoid OOM if needed, but torchmetrics FID updates incrementally
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Update in batches
    eval_bs = 50
    for i in range(0, len(real_images_all), eval_bs):
        real_batch = real_images_all[i : i + eval_bs].to(device)
        # Ensure uint8
        real_batch = (real_batch * 255).to(torch.uint8)
        fid.update(real_batch, real=True)

    for i in range(0, len(fake_images_all), eval_bs):
        fake_batch = fake_images_all[i : i + eval_bs].to(device)
        fake_batch = (fake_batch * 255).to(torch.uint8)
        fid.update(fake_batch, real=False)

    fid_score = fid.compute().item()
    print(f"FID: {fid_score:.4f}")

    # CLIP Score
    # Calculate in batches
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(
        device
    )
    clip_scores = []

    for i in range(0, len(fake_images_all), eval_bs):
        fake_batch = fake_images_all[i : i + eval_bs].to(device)
        fake_batch = (fake_batch * 255).to(torch.uint8)
        batch_prompts = all_prompts[i : i + eval_bs]

        score = clip_metric(fake_batch, batch_prompts)
        clip_scores.append(score.item())

    avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
    print(f"CLIP Score: {avg_clip_score:.4f}")

    metrics = {
        "fid": fid_score,
        "clip_score": avg_clip_score,
        "num_samples": num_samples,
    }

    # Save metrics
    import json

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
