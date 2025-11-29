"""
Implement evaluation methods:
1. FID score
2. CLIP score
3. Gather and plot generated images for human eval
"""

import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from typing import List, Optional, Union, Dict, Any
import os
import sys
import numpy as np
import gc
import math
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
    model_name_or_path: str = "openai/clip-vit-base-patch16",
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Calculate CLIP score for images and prompts.

    Args:
        images: Tensor of shape [B, C, H, W], values in [0, 1] or [0, 255]
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


def format_prompt_caption(prompts: List[str], limit: int=32) -> str:
    if not prompts:
        return ""
    trimmed = [p.replace("\n", " ").strip() for p in prompts[:limit]]
    lines = [f"{idx + 1}. {text}" for idx, text in enumerate(trimmed)]
    remaining = len(prompts) - len(trimmed)
    if remaining > 0:
        lines.append(f"... (+{remaining} more)")
    return "\n\n".join(lines)


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

    # Get VAE stats
    from .vae_codec import get_vae_stats

    vae_mean, vae_std = get_vae_stats(args.vae_path, device=accelerator.device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

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

        # Denormalize
        samples = samples * vae_std + vae_mean

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

        kid_score = calculate_kid(
            real_images_norm, fake_images_norm, device=accelerator.device
        )
        print(f"Epoch {epoch + 1} | KID: {kid_score:.4f}")
        metrics["kid"] = kid_score

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

                # Denormalize
                lat = lat * vae_std + vae_mean

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


def gather_all(accelerator, tensor):
    if not accelerator.use_distributed:
        return tensor
    return accelerator.gather(tensor)

def run_evaluation_light(
    accelerator: Accelerator,
    model: torch.nn.Module,
    vae_path: str,
    save_path: str,
    current_step: int,
    text_encoder: Any,
    processor: Any,
    pooling: bool,
    dataset_path: str = "./precomputed_dataset/light-eval@256p",
    num_samples: int = 16,
    batch_size: int = 16,
    bucket_resolutions = {1: (256, 256),2: (336, 192),3: (192, 336),4: (288, 224),5: (224, 288)}
) -> Dict[str, float]:
    """Run the fast validation loop over the light-eval split."""
    print(f"Running light evaluation at step {current_step}...")
    # Imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
    from flow.solvers import sample_ode
    from .encode_text import encode_text
    from .vae_codec import get_vae_stats
    from diffusers import AutoencoderKLQwenImage
    from datasets import load_from_disk

    model.eval()

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(accelerator.device)

    # Get VAE stats for denormalization
    vae_mean, vae_std = get_vae_stats(vae_path, device=accelerator.device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    metrics = {}
    num_processes = getattr(accelerator, "num_processes", 1)
    process_index = getattr(accelerator, "process_index", 0)
    can_broadcast_objects = num_processes > 1 and hasattr(accelerator, "broadcast_object_list")
    can_gather_objects = num_processes > 1 and hasattr(accelerator, "gather_object")

    with torch.no_grad():
        real_images_list: List[torch.Tensor] = []
        bucket_prompts: Dict[int, List[str]] = {bid: [] for bid in bucket_resolutions}

        if accelerator.is_main_process:
            dataset = load_from_disk(dataset_path)
            if num_samples is not None:
                num_eval = min(num_samples, len(dataset))
                end_idx = current_step % (len(dataset) - num_eval) + num_eval
                indices = list(range(end_idx - num_eval, end_idx))
                dataset = dataset.select(indices)

            for item in dataset:
                latents = item["latents"].unsqueeze(0).unsqueeze(2).to(accelerator.device).to(torch.bfloat16)
                img = vae.decode(latents).sample.squeeze(2).squeeze(0)
                img = torch.clamp((img + 1) / 2, 0, 1)
                real_images_list.append(img.cpu().float())

                captions_list = item.get("captions", "")
                prompt = captions_list[0]

                bucket_id = int(item.get("resolution_bucket_id", 1))
                bucket_prompts.setdefault(bucket_id, []).append(prompt)
        
        bucket_payload = [bucket_prompts]
        if can_broadcast_objects:
            accelerator.broadcast_object_list(bucket_payload)
        bucket_prompts = bucket_payload[0]

        if accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)

        bucket_fake_images_local: Dict[int, List[torch.Tensor]] = {bid: [] for bid in bucket_resolutions}
        fake_images_local: List[torch.Tensor] = []
        generation_prompts_local: List[str] = []

        # Generation loop per bucket
        for bucket_id, prompts_list in bucket_prompts.items():
            total_prompts = len(prompts_list)
            if total_prompts == 0:
                continue # no prompts in this bucket

            chunk = math.ceil(total_prompts / max(1, num_processes))
            start = chunk * process_index
            end = min(start + chunk, total_prompts)
            if start >= end:
                continue

            local_prompts = prompts_list[start:end]
            H, W = bucket_resolutions[bucket_id]

            for batch_start in range(0, len(local_prompts), batch_size):
                batch_prompts = local_prompts[batch_start: batch_start + batch_size]
                txt, txt_mask, txt_pooled = encode_text(batch_prompts, text_encoder, processor, pooling)

                sample_z0 = torch.randn(len(batch_prompts), 16, H // 8, W // 8, device=accelerator.device)

                def model_fn(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask):
                    if isinstance(t, float):
                        t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
                    else:
                        t_tensor = t
                    return model(x, t_tensor, txt, txt_pooled, txt_mask)
                
                with accelerator.autocast():
                    samples = sample_ode(model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0)
                
                samples = samples.to(dtype=torch.bfloat16)

                # Denormalize: z = z_norm * std + mean
                samples = samples * vae_std + vae_mean
                samples = samples.unsqueeze(2)
                samples = vae.decode(samples).sample.squeeze(2)
                samples = torch.clamp((samples + 1) / 2, 0, 1)
                samples = samples.cpu().float()
                bucket_fake_images_local[bucket_id].append(samples)
                for img, prompt in zip(samples, batch_prompts):
                    fake_images_local.append(img)
                    generation_prompts_local.append(prompt)

        # Gather prompts and generated fake images
        if can_gather_objects:
            gathered_bucket_parts = accelerator.gather_object(bucket_fake_images_local)
            gathered_fake_lists = accelerator.gather_object(fake_images_local)
            gathered_prompt_lists = accelerator.gather_object(generation_prompts_local)
        else:
            gathered_bucket_parts = [bucket_fake_images_local]
            gathered_fake_lists = [fake_images_local]
            gathered_prompt_lists = [generation_prompts_local]

        if accelerator.is_main_process:
            bucket_fake_images: Dict[int, List[torch.Tensor]] = {bid: [] for bid in bucket_resolutions}
            for part in gathered_bucket_parts:
                if not part:
                    continue
                for bid, batches in part.items():
                    bucket_fake_images[bid].extend(batches)
            
            fake_images_list: List[torch.Tensor] = []
            for images in gathered_fake_lists:
                if images:
                    fake_images_list.extend(images)
            
            generation_prompts: List[str] = []
            for prompts in gathered_prompt_lists:
                if prompts:
                    generation_prompts.extend(prompts)

            # Save to local path and report generated images to swanlab
            bucket_grid_paths = []
            bucket_grid_captions: List[str] = []
            for bucket_id, image_batches in bucket_fake_images.items():
                if not image_batches:
                    continue
                bucket_images = torch.cat(image_batches, dim=0)
                H, W = bucket_resolutions[bucket_id]
                grid_path = os.path.join(save_path, f"samples/samples_step_{current_step:06d}_bucket{bucket_id}_{H}x{W}.png")
                _ = make_image_grid(bucket_images[:9], save_path=grid_path, normalize=True, value_range=(0,1))
                print(f"Saved samples to {grid_path}")
                bucket_grid_paths.append(grid_path)
                prompts_for_bucket = bucket_prompts.get(bucket_id, [])
                prompts_for_bucket = prompts_for_bucket[:bucket_images.shape[0]]
                bucket_grid_captions.append(format_prompt_caption(prompts_for_bucket[:9]))

            for idx, (path, caption) in enumerate(zip(bucket_grid_paths, bucket_grid_captions)):
                media = swanlab.Image(path, caption=caption) if caption else swanlab.Image(path)
                accelerator.log({f"samples_{idx}": media}, step=current_step)

            # Calculate metrics
            if real_images_list and fake_images_list:
                real = [torch.clamp(img, 0, 1) for img in real_images_list]
                fake = [torch.clamp(img, 0, 1) for img in fake_images_list]

                # Resize to 299x299 for metric calculation
                real = torch.stack([F.interpolate(img.unsqueeze(0), size=(299,299), mode="bicubic", align_corners=False).squeeze(0) for img in real])
                fake = torch.stack([F.interpolate(img.unsqueeze(0), size=(299,299), mode="bicubic", align_corners=False).squeeze(0) for img in fake])

                real = real.clamp(0,1)
                fake = fake.clamp(0,1)

                fid_score = calculate_fid(real, fake, device=accelerator.device, batch_size=batch_size)
                metrics["fid"] = fid_score
                print(f"Step {current_step} | FID: {fid_score:.4f}")

                kid_score = calculate_kid(real, fake, device=accelerator.device, batch_size=batch_size)
                metrics["kid"] = kid_score
                print(f"Step {current_step} | KID: {kid_score:.4f}")

                clip_score = calculate_clip_score(fake, generation_prompts, device=accelerator.device, batch_size=batch_size)
                metrics["clip_score"] = clip_score
                print(f"Step {current_step} | CLIP Score: {clip_score:.4f}")

                accelerator.log(metrics, step=current_step)

            else:
                print("Insufficient data for metric computation.")

    metrics_payload = [metrics]
    if can_broadcast_objects:
        accelerator.broadcast_object_list(metrics_payload)
    metrics = metrics_payload[0]

    # Cleanup
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    return metrics

def run_evaluation_heavy(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    vae_path: str,
    text_encoder_path: str,
    pooling: bool,
    save_path: str,
    dataset_path: str = "./precomputed_dataset/heavy-eval@256p",
    num_fid_samples: int = 2000,
    num_clip_samples: int = 2000,
    batch_size: int = 32,
    device: str = "cuda:0"
) -> Dict[str, float]:
    """Run the large-scale test loop over the heavy-eval split."""
    print(f"Running heavy evaluation on {checkpoint_path}...")
    # Imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ..flow.solvers import sample_ode
    from .encode_text import encode_text
    from ..models.artflow import ArtFlow
    from ..dataset.dataloader_utils import ResolutionBucketSampler, collate_fn
    from .vae_codec import get_vae_stats

    from datasets import load_from_disk
    from diffusers import AutoencoderKLQwenImage
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # 1. Load Models
    print("Loading models:\n"
          f"  diffusion model: {checkpoint_path},\n"
          f"  vae: {vae_path},\n"
          f"  text encoder: {text_encoder_path}...")
    model = ArtFlow(**model_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)

    # Get VAE stats for denormalization
    vae_mean, vae_std = get_vae_stats(vae_path, device=device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        text_encoder_path,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(text_encoder_path)

    # 2. Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    num_samples = max(num_fid_samples, num_clip_samples)

    if len(dataset) < num_samples:
        print(
            f"Warning: Dataset size ({len(dataset)}) is smaller than requested samples ({num_samples})"
        )
        num_samples = len(dataset)
        print(f"Warning: Dataset size ({num_samples}) is smaller than requested samples ({num_fid_samples}/{num_clip_samples})")
    else:
        print(f"")
        dataset = dataset.select(range(num_samples))


    # 3. Generation Loop
    print(f"Generating {num_samples} samples...")
    generated_images_dir = os.path.join(save_path, "generated_images")
    os.makedirs(generated_images_dir, exist_ok=True)

    all_prompts = []
    all_fake_paths = []
    real_images_list = []
    fake_images_list = []

    sampler = ResolutionBucketSampler(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
            prompts = [c[0] for c in batch["captions"]]

            # Captions
            all_prompts.extend(prompts)

            # Real images
            latents = batch["latents"].to(device).to(torch.bfloat16).unsqueeze(2)
            real_imgs = vae.decode(latents).sample.squeeze(2)
            real_imgs = torch.clamp((real_imgs + 1) / 2, 0, 1).cpu()
            real_images_list.append(real_imgs)

            # Fake images
            txt, txt_mask, txt_pooled = encode_text(prompts, text_encoder, processor, pooling)
            _, _, H_lat, W_lat = batch["latents"].shape
            H, W = H_lat * 8, W_lat * 8
            sample_z0 = torch.randn(len(prompts), 16, H // 8, W // 8, device=device)

            def model_fn(x, t):
                if isinstance(t, float):
                    t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
                else:
                    t_tensor = t
                return model(x, t_tensor, txt, txt_pooled, txt_mask)
            
            device_type = "cuda" if "cuda" in device else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                samples = sample_ode(model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0, device=device)
            
            samples = samples.to(dtype=torch.bfloat16)
            
            # Denormalize
            samples = samples * vae_std + vae_mean

            samples = samples.unsqueeze(2)
            fake_imgs = vae.decode(samples).sample.squeeze(2)
            fake_imgs = torch.clamp((fake_imgs + 1) / 2, 0, 1).cpu()
            fake_images_list.append(fake_imgs)

            # Save images
            for j, img in enumerate(fake_imgs):
                idx = len(all_fake_paths)
                save_path_img = os.path.join(generated_images_dir, f"sample_{idx:05d}.png")
                torchvision.utils.save_image(img, save_path_img)
                all_fake_paths.append(save_path_img)

    # 4. Calculate Metrics
    print("Calculating metrics...")

    # Concatenate all images
    first_shape = fake_images_list[0].shape[-2:]
    all_same_shape = all(t.shape[-2:] == first_shape for t in fake_images_list)

    if not all_same_shape:
        print("Variable resolutions detected. Resizing to 299x299 for metric calculation.")
        real_images_all = []
        fake_images_all = []
        for batch in real_images_list:
            real_images_all.append(
                torch.nn.functional.interpolate(
                    batch, size=(299, 299), mode="bicubic", align_corners=False
                )
            )
        for batch in fake_images_list:
            fake_images_all.append(
                torch.nn.functional.interpolate(
                    batch, size=(299, 299), mode="bicubic", align_corners=False
                )
            )

        real_images_all = torch.cat(real_images_all, dim=0)
        fake_images_all = torch.cat(fake_images_all, dim=0)
    else:
        real_images_all = torch.cat(real_images_list, dim=0)
        fake_images_all = torch.cat(fake_images_list, dim=0)

    metrics = {}

    # FID
    if num_fid_samples > 0:
        print(f"Calculating FID on {min(len(real_images_all), num_fid_samples)} samples...")
        real_subset = real_images_all[:num_fid_samples]
        fake_subset = fake_images_all[:num_fid_samples]

        fid_score = calculate_fid(
            real_subset,
            fake_subset,
            device=device,
            batch_size=batch_size,
        )
        print(f"FID: {fid_score:.4f}")
        metrics["fid"] = fid_score

    # CLIP score
    if num_clip_samples > 0:
        print(f"Calculating CLIP Score on {min(len(fake_images_all), num_clip_samples)} samples...")
        fake_subset = fake_images_all[:num_clip_samples]
        prompts_subset = all_prompts[:num_clip_samples]

        clip_score = calculate_clip_score(
            fake_subset,
            prompts_subset,
            device=device,
            batch_size=batch_size,
        )
        print(f"CLIP Score: {clip_score:.4f}")
        metrics["clip_score"] = clip_score

    metrics["num_samples"] = num_samples
    metrics["num_fid_samples"] = num_fid_samples
    metrics["num_clip_samples"] = num_clip_samples

    return metrics
            
            
