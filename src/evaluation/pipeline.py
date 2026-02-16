"""
Evaluation pipelines for ArtFlow models.

Functions:
- run_evaluation_uncond: Evaluation for unconditional generation (Stage 0)
- run_evaluation_light: Fast validation loop for conditional generation
- run_evaluation_heavy: Large-scale evaluation for final metrics
"""

import gc
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import swanlab
import transformers

from .metrics import calculate_fid, calculate_kid, calculate_clip_score
from .visualize import make_image_grid, visualize_denoising, format_prompt_caption

transformers.logging.set_verbosity_error()


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
    from ..utils.vae_codec import get_vae_stats

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
    tokenizer: Any,
    pooling: bool,
    dataset_path: str = "./precomputed_dataset/light-eval@256p",
    num_samples: int = 16,
    batch_size: int = 16,
    compute_metrics: bool = False,
    bucket_resolutions={
        1: (256, 256),
        2: (336, 192),
        3: (192, 336),
        4: (288, 224),
        5: (224, 288),
    },
) -> Dict[str, float]:
    """Run the fast validation loop over the light-eval split."""
    print(f"Running light evaluation at step {current_step}...")
    # Imports
    from ..flow.solvers import sample_ode
    from ..utils.encode_text import encode_text
    from ..utils.vae_codec import get_vae_stats
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
    can_broadcast_objects = num_processes > 1 and hasattr(
        accelerator, "broadcast_object_list"
    )
    can_gather_objects = num_processes > 1 and hasattr(accelerator, "gather_object")

    with torch.no_grad():
        real_images_list: List[torch.Tensor] = []
        bucket_prompts: Dict[int, List[str]] = {bid: [] for bid in bucket_resolutions}

        if accelerator.is_main_process:
            dataset = load_from_disk(dataset_path).shuffle(seed=42)
            if num_samples is not None:
                num_eval = min(num_samples, len(dataset))
                end_idx = current_step % (len(dataset) - num_eval) + num_eval
                indices = list(range(end_idx - num_eval, end_idx))
                dataset = dataset.select(indices)

            for item in dataset:
                latents = (
                    item["latents"]
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .to(accelerator.device)
                    .to(torch.bfloat16)
                )
                img = vae.decode(latents).sample.squeeze(2).squeeze(0)
                img = torch.clamp((img + 1) / 2, 0, 1)
                real_images_list.append(img.cpu().float())

                captions_list = item.get("captions", "")
                prompt = captions_list[1] if len(captions_list) > 1 else captions_list[0]

                bucket_id = int(item.get("resolution_bucket_id", 1))
                bucket_prompts.setdefault(bucket_id, []).append(prompt)

        bucket_payload = [bucket_prompts]
        if can_broadcast_objects:
            accelerator.broadcast_object_list(bucket_payload)
        bucket_prompts = bucket_payload[0]

        if accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)

        bucket_fake_images_local: Dict[int, List[torch.Tensor]] = {
            bid: [] for bid in bucket_resolutions
        }
        fake_images_local: List[torch.Tensor] = []
        generation_prompts_local: List[str] = []

        # Generation loop per bucket
        for bucket_id, prompts_list in bucket_prompts.items():
            total_prompts = len(prompts_list)
            if total_prompts == 0:
                continue  # no prompts in this bucket

            chunk = math.ceil(total_prompts / max(1, num_processes))
            start = chunk * process_index
            end = min(start + chunk, total_prompts)
            if start >= end:
                continue

            local_prompts = prompts_list[start:end]
            H, W = bucket_resolutions[bucket_id]

            for batch_start in range(0, len(local_prompts), batch_size):
                batch_prompts = local_prompts[batch_start : batch_start + batch_size]
                txt, txt_mask, txt_pooled = encode_text(
                    batch_prompts, text_encoder, tokenizer, pooling
                )

                sample_z0 = torch.randn(
                    len(batch_prompts), 16, H // 8, W // 8, device=accelerator.device
                )

                def model_fn(
                    x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask
                ):
                    if isinstance(t, float):
                        t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])
                    else:
                        t_tensor = t
                    return model(x, t_tensor, txt, txt_pooled, txt_mask)

                with accelerator.autocast():
                    samples = sample_ode(
                        model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0
                    )

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
            bucket_fake_images: Dict[int, List[torch.Tensor]] = {
                bid: [] for bid in bucket_resolutions
            }
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
                grid_path = os.path.join(
                    save_path,
                    f"samples/samples_step_{current_step:06d}_bucket{bucket_id}_{H}x{W}.png",
                )
                _ = make_image_grid(
                    bucket_images[:4], save_path=grid_path, normalize=True, value_range=(0, 1)
                )
                print(f"Saved samples to {grid_path}")
                bucket_grid_paths.append(grid_path)
                prompts_for_bucket = bucket_prompts.get(bucket_id, [])
                prompts_for_bucket = prompts_for_bucket[: bucket_images.shape[0]]
                bucket_grid_captions.append(format_prompt_caption(prompts_for_bucket[:4]))

            for idx, (path, caption) in enumerate(
                zip(bucket_grid_paths, bucket_grid_captions)
            ):
                media = (
                    swanlab.Image(path, caption=caption)
                    if caption
                    else swanlab.Image(path)
                )
                accelerator.log({f"samples_{idx}": media}, step=current_step)

            # Calculate metrics (optional, controlled by compute_metrics flag)
            if compute_metrics and real_images_list and fake_images_list:
                real = [torch.clamp(img, 0, 1) for img in real_images_list]
                fake = [torch.clamp(img, 0, 1) for img in fake_images_list]

                # Resize to 299x299 for metric calculation
                real = torch.stack(
                    [
                        F.interpolate(
                            img.unsqueeze(0),
                            size=(299, 299),
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze(0)
                        for img in real
                    ]
                )
                fake = torch.stack(
                    [
                        F.interpolate(
                            img.unsqueeze(0),
                            size=(299, 299),
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze(0)
                        for img in fake
                    ]
                )

                real = real.clamp(0, 1)
                fake = fake.clamp(0, 1)

                fid_score = calculate_fid(
                    real, fake, device=accelerator.device, batch_size=batch_size
                )
                metrics["fid"] = fid_score
                print(f"Step {current_step} | FID: {fid_score:.4f}")

                kid_score = calculate_kid(
                    real, fake, device=accelerator.device, batch_size=batch_size
                )
                metrics["kid"] = kid_score
                print(f"Step {current_step} | KID: {kid_score:.4f}")

                clip_score = calculate_clip_score(
                    fake,
                    generation_prompts,
                    device=accelerator.device,
                    batch_size=batch_size,
                )
                metrics["clip_score"] = clip_score
                print(f"Step {current_step} | CLIP Score: {clip_score:.4f}")

                accelerator.log(metrics, step=current_step)

            elif compute_metrics:
                print("Insufficient data for metric computation.")

    if compute_metrics:
        metrics_payload = [metrics]
        if can_broadcast_objects:
            accelerator.broadcast_object_list(metrics_payload)
        metrics = metrics_payload[0]

    # Cleanup
    del vae
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

    return {}


def run_evaluation_heavy(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    vae_path: str,
    text_encoder_path: str,
    pooling: bool,
    save_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_path: str = "./precomputed_dataset/heavy-eval@256p",
    num_samples: int = 2000,
    num_fid_samples: Optional[int] = None,
    num_clip_samples: Optional[int] = None,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Run the large-scale test loop over the heavy-eval split."""
    print(f"Running heavy evaluation on {checkpoint_path}...")
    # Imports
    from ..flow.solvers import sample_ode
    from ..utils.encode_text import encode_text
    from ..models.artflow import ArtFlow
    from ..dataset.sampler import ResolutionBucketSampler, collate_fn
    from ..utils.vae_codec import get_vae_stats

    from datasets import load_from_disk
    from diffusers import AutoencoderKLQwenImage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Load Models
    print(
        "Loading models:\n"
        f"  diffusion model: {checkpoint_path},\n"
        f"  vae: {vae_path},\n"
        f"  text encoder: {text_encoder_path}..."
    )
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

    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    # 2. Load Dataset
    target_dir = output_dir if output_dir is not None else save_path
    if target_dir is None:
        raise ValueError("Either save_path or output_dir must be provided.")
    save_path = target_dir
    os.makedirs(save_path, exist_ok=True)

    fid_limit = num_fid_samples if num_fid_samples is not None else num_samples
    clip_limit = num_clip_samples if num_clip_samples is not None else num_samples
    eval_target = max(num_samples, fid_limit, clip_limit)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    available = len(dataset)
    if available < eval_target:
        print(
            f"Warning: Dataset size ({available}) is smaller than requested samples ({eval_target})"
        )
        effective_sample_count = available
    else:
        dataset = dataset.select(range(eval_target))
        effective_sample_count = eval_target

    # 3. Generation Loop
    print(f"Generating {effective_sample_count} samples...")
    generated_images_dir = os.path.join(save_path, "generated_images")
    os.makedirs(generated_images_dir, exist_ok=True)

    all_prompts = []
    all_fake_paths = []
    real_images_list = []
    fake_images_list = []
    grid_idx = 0

    sampler = ResolutionBucketSampler(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
            prompts = [c[1] or c[0] for c in batch["captions"]]

            # Captions
            all_prompts.extend(prompts)

            # Real images
            latents = batch["latents"].to(device).to(torch.bfloat16).unsqueeze(2)
            real_imgs = vae.decode(latents).sample.squeeze(2)
            real_imgs = torch.clamp((real_imgs + 1) / 2, 0, 1).cpu()
            real_images_list.append(real_imgs)

            # Fake images
            txt, txt_mask, txt_pooled = encode_text(
                prompts, text_encoder, tokenizer, pooling
            )
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
                samples = sample_ode(
                    model_fn, sample_z0, steps=50, t_start=0.0, t_end=1.0, device=device
                )

            samples = samples.to(dtype=torch.bfloat16)

            # Denormalize
            samples = samples * vae_std + vae_mean

            samples = samples.unsqueeze(2)
            fake_imgs = vae.decode(samples).sample.squeeze(2)
            fake_imgs = torch.clamp((fake_imgs + 1) / 2, 0, 1).cpu()
            fake_images_list.append(fake_imgs)

            # Save 4x4 grids per batch to avoid mixing resolutions
            batch_size_curr = fake_imgs.shape[0]
            for start_idx in range(0, batch_size_curr, 16):
                chunk = fake_imgs[start_idx : start_idx + 16]
                save_path_img = os.path.join(
                    generated_images_dir, f"grid_{grid_idx:05d}.png"
                )
                make_image_grid(
                    chunk, rows=4, cols=4, save_path=save_path_img, normalize=False
                )
                all_fake_paths.append(save_path_img)
                grid_idx += 1
    # 4. Calculate Metrics
    print("Calculating metrics...")

    # Concatenate all images
    # Always resize to 299x299 for metric calculation to ensure consistency with InceptionV3
    print("Resizing images to 299x299 for metric calculation...")
    real_images_all = []
    fake_images_all = []

    # Process in chunks to save memory if needed, but here we just loop
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

    # Sanity check: ensure real and fake are not identical
    if real_images_all.shape == fake_images_all.shape:
        diff = (real_images_all - fake_images_all).abs().mean()
        print(f"Mean absolute difference between real and fake images: {diff:.6f}")
        if diff < 1e-6:
            print(
                "WARNING: Real and fake images appear to be identical! Check evaluation logic."
            )

    metrics = {}

    # FID
    fid_count = min(len(real_images_all), fid_limit)
    if fid_count > 0:
        print(f"Calculating FID on {fid_count} samples...")
        real_subset = real_images_all[:fid_count]
        fake_subset = fake_images_all[:fid_count]

        fid_score = calculate_fid(
            real_subset,
            fake_subset,
            device=device,
            batch_size=batch_size,
        )
        print(f"FID: {fid_score:.4f}")
        metrics["fid"] = fid_score
        metrics["num_fid_samples"] = fid_count

    # CLIP score
    clip_count = min(len(fake_images_all), len(all_prompts), clip_limit)
    if clip_count > 0:
        print(f"Calculating CLIP Score on {clip_count} samples...")
        fake_subset = fake_images_all[:clip_count]
        prompts_subset = all_prompts[:clip_count]

        clip_score = calculate_clip_score(
            fake_subset,
            prompts_subset,
            device=device,
            batch_size=batch_size,
        )
        print(f"CLIP Score: {clip_score:.4f}")
        metrics["clip_score"] = clip_score
        metrics["num_clip_samples"] = clip_count

    metrics["num_samples"] = effective_sample_count

    metrics_path = os.path.join(save_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
