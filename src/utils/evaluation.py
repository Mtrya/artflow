"""
Implement evaluation methods:
1. FID score
2. CLIP score
3. Gather and plot generated images for human eval
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from typing import List, Optional, Union, Dict, Any
import os
import numpy as np
import logging
import warnings
import gc
from accelerate import Accelerator
from torch.utils.data import DataLoader

import transformers
transformers.logging.set_verbosity_error()
import wandb

def calculate_fid(
    real_images: torch.Tensor, 
    fake_images: torch.Tensor, 
    feature: int = 2048, 
    device: Optional[torch.device] = None
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
    device: Optional[torch.device] = None
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
    value_range: Optional[tuple] = None
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
        images, 
        nrow=nrow, 
        normalize=normalize, 
        value_range=value_range,
        padding=2
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(grid, save_path)
        
    return grid

def visualize_denoising(
    intermediate_steps: List[torch.Tensor], 
    save_path: str,
    num_steps_to_show: int = 10
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
    first_sample_steps = [step[0] for step in selected_steps] # List of [C, H, W]
    
    # Stack them: [Num_steps, C, H, W]
    stacked = torch.stack(first_sample_steps)
    
    # Make grid: 1 row, Num_steps columns
    make_image_grid(
        stacked, 
        rows=1, 
        cols=len(selected_steps), 
        save_path=save_path, 
        normalize=True, 
        value_range=(-1, 1)
    )

def run_evaluation_uncond(
    accelerator: Accelerator,
    model: torch.nn.Module,
    args: Any,
    epoch: int,
    train_dataloader: DataLoader,
    algorithm: Any,
    sample_ode_fn: Any,
    ScoreMatchingODE_cls: Any
)->dict:
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
        args.vae_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(accelerator.device)
    
    with torch.no_grad():
        # --- 1. Generate Samples ---
        sample_z0 = torch.randn(args.num_eval_samples, 16, args.resolution // 8, args.resolution // 8, device=accelerator.device)
        
        # Define model wrapper for solver
        def model_fn(x, t):
            if isinstance(t, float):
                t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
            else:
                t_tensor = t
            return model(x, t_tensor)
        
        if args.algorithm == "fm-ot":
            samples = sample_ode_fn(model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0)
        elif args.algorithm == "sm-diffusion":
            solver = ScoreMatchingODE_cls(beta_min=algorithm.beta_min, beta_max=algorithm.beta_max)
            samples = sample_ode_fn(model_fn, sample_z0, steps=50, t_start=1.0, t_end=0.0, solver_instance=solver)
        else:
            samples = sample_ode_fn(model_fn, sample_z0, steps=50, t_start=1.0, t_end=0.0)
        
        samples = samples.to(dtype=torch.bfloat16)
        samples = samples.unsqueeze(2)
        images = vae.decode(samples).sample
        images = images.squeeze(2)
        
        # Save locally
        save_path = os.path.join(args.output_dir, f"{args.run_name}/samples_epoch_{epoch+1:03d}.png")
        grid = make_image_grid(images, save_path=save_path, normalize=True, value_range=(-1, 1))
        print(f"Saved samples to {save_path}")
        
        # Log to WandB
        if accelerator.is_main_process:
            accelerator.log({"samples": wandb.Image(save_path)}, step=epoch+1)

        # --- 2. Evaluation Metrics ---
        metrics = {}
        
        # FID
        # Get real images from dataloader (just one batch for stage0)
        real_batch = next(iter(train_dataloader))
        real_latents = real_batch["latents"].to(accelerator.device).to(dtype=torch.bfloat16)
        real_latents = real_latents.unsqueeze(2)
        real_images = vae.decode(real_latents).sample.squeeze(2)
        
        # Normalize to [0, 1]
        real_images_norm = (real_images + 1) / 2
        fake_images_norm = (images + 1) / 2
        
        fid_score = calculate_fid(real_images_norm, fake_images_norm, device=accelerator.device)
        print(f"Epoch {epoch+1} | FID: {fid_score:.4f}")
        metrics["fid"] = fid_score

        # CLIP Score
        prompt = "A Monet painting"
        clip_score = calculate_clip_score(fake_images_norm, prompt, device=accelerator.device)
        print(f"Epoch {epoch+1} | CLIP Score: {clip_score:.4f}")
        metrics["clip_score"] = clip_score

        if accelerator.is_main_process and metrics:
            accelerator.log(metrics, step=epoch+1)

        # --- 3. Denoising Visualization ---
        try:
            vis_z0 = torch.randn(1, 16, args.resolution // 8, args.resolution // 8, device=accelerator.device)
            if args.algorithm == "fm-ot":
                    _, intermediates = sample_ode_fn(model_fn, vis_z0, steps=20, t_start=0.0, t_end=1.0, return_intermediates=True)
            elif args.algorithm == "sm-diffusion":
                    solver = ScoreMatchingODE_cls(beta_min=algorithm.beta_min, beta_max=algorithm.beta_max)
                    _, intermediates = sample_ode_fn(model_fn, vis_z0, steps=20, t_start=1.0, t_end=0.0, solver_instance=solver, return_intermediates=True)
            else:
                    _, intermediates = sample_ode_fn(model_fn, vis_z0, steps=20, t_start=1.0, t_end=0.0, return_intermediates=True)
            
            decoded_intermediates = []
            for lat in intermediates:
                lat = lat.to(accelerator.device).to(dtype=torch.bfloat16)
                lat = lat.unsqueeze(2)
                img = vae.decode(lat).sample.squeeze(2)
                decoded_intermediates.append(img.cpu())
            
            vis_path = os.path.join(args.output_dir, f"{args.run_name}/denoising_epoch_{epoch+1:03d}.png")
            visualize_denoising(decoded_intermediates, vis_path)
            print(f"Saved denoising visualization to {vis_path}")
            if accelerator.is_main_process:
                accelerator.log({"denoising": wandb.Image(vis_path)}, step=epoch+1)
                
        except Exception as e:
            print(f"Denoising visualization failed: {e}")
    
    # Cleanup VAE
    del vae
    gc.collect()
    torch.cuda.empty_cache()

def run_evaluation_light():
    pass

def run_evaluation_heavy():
    pass
    