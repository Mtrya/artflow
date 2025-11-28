"""
Main training script for ArtFlow.
Stage 0: unconditional image generation with WikiArt Monet subset, for algorithm ablation
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from datasets import load_from_disk

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.artflow_uncond import ArtFlowUncond
from src.flow.paths import ScoreMatchingDiffusion, FlowMatchingDiffusion, FlowMatchingOT
from src.flow.solvers import sample_ode, ScoreMatchingODE
from src.utils.evaluation import run_evaluation_uncond
from src.utils.vae_codec import get_vae_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train ArtFlow Stage 0")
    parser.add_argument(
        "--run_name", type=str, default="artflow_run", help="Name of the run"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--precomputed_dataset_path",
        type=str,
        required=True,
        help="Path to precomputed dataset",
    )
    parser.add_argument(
        "--range", type=int, default=-1, help="Range of dataset to use (for debugging)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per device"
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sm-diffusion", "fm-diffusion", "fm-ot"],
        default="fm-ot",
        help="Algorithm to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=20, help="Epochs between checkpoints"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=20, help="Epochs between evaluation"
    )
    parser.add_argument(
        "--eval_resolution",
        type=int,
        default=256,
        help="Resolution for evaluation generation",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=9,
        help="Number of samples to generate during evaluation",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="REPA-E/e2e-qwenimage-vae",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--model_hidden_size", type=int, default=512, help="Hidden size of the model"
    )
    parser.add_argument("--model_depth", type=int, default=8, help="Depth of the model")
    parser.add_argument(
        "--model_num_heads", type=int, default=8, help="Number of heads in the model"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="swanlab")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="artflow-stage0",
            config=vars(args),
            init_kwargs={"swanlab": {"experiment_name": args.run_name}},
        )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Starting run: {args.run_name}")
        print(f"Algorithm: {args.algorithm}")

    # 1. Load and Preprocess Data
    # For Stage 0, we use fixed resolution and no text conditioning (unconditional)
    if accelerator.is_main_process:
        print(f"Loading precomputed dataset from {args.precomputed_dataset_path}...")

    dataset = load_from_disk(args.precomputed_dataset_path)

    if args.range > 0:
        if accelerator.is_main_process:
            print(f"Selecting first {args.range} examples...")
        dataset = dataset.select(range(min(args.range, len(dataset))))

    # Set format for pytorch
    dataset.set_format("torch", columns=["latents"])

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Load VAE Stats
    if accelerator.is_main_process:
        print(f"Loading VAE Stats from {args.vae_path}")
    vae_mean, vae_std = get_vae_stats(args.vae_path, device=accelerator.device)
    if args.mixed_precision == "bf16":
        vae_mean = vae_mean.to(dtype=torch.bfloat16)
        vae_std = vae_std.to(dtype=torch.bfloat16)
    elif args.mixed_precision == "fp16":
        vae_mean = vae_mean.to(dtype=torch.float16)

    # 2. Model Setup
    model = ArtFlowUncond(
        in_channels=16,
        hidden_size=args.model_hidden_size,
        depth=args.model_depth,
        num_heads=args.model_num_heads,
        patch_size=2,
    )

    # 3. Algorithm Setup
    if args.algorithm == "sm-diffusion":
        algorithm = ScoreMatchingDiffusion()
    elif args.algorithm == "fm-diffusion":
        algorithm = FlowMatchingDiffusion()
    elif args.algorithm == "fm-ot":
        algorithm = FlowMatchingOT()
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 5. Prepare with Accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # 6. Training Loop
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}",
        )

        total_loss = 0.0
        for batch in progress_bar:
            latents = batch["latents"]  # [B, 16, H, W]

            # Normalize latents
            latents = (latents - vae_mean) / vae_std

            # Sample t
            t = torch.rand(latents.shape[0], device=latents.device)

            # Sample z0 (noise) and z1 (data)
            z1 = latents
            z0 = torch.randn_like(z1)

            # Sample z_t
            z_t = algorithm.sample_zt(z0, z1, t)

            # Model prediction
            model_output = model(z_t, t)

            # Compute loss
            loss = algorithm.compute_loss(model_output, z0, z1, t)

            # Backward
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")
            accelerator.log({"train_loss": avg_loss}, step=epoch)

            # Save checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir, f"{args.run_name}/checkpoint_epoch_{epoch + 1:03d}"
                )
                accelerator.save_state(checkpoint_dir)

            # Evaluation
            if (epoch + 1) % args.eval_interval == 0:
                run_evaluation_uncond(
                    accelerator=accelerator,
                    model=model,
                    args=args,
                    epoch=epoch,
                    train_dataloader=train_dataloader,
                    algorithm=algorithm,
                    sample_ode_fn=sample_ode,
                    ScoreMatchingODE_cls=ScoreMatchingODE,
                )

    accelerator.end_training()
    print("Training finished.")


if __name__ == "__main__":
    main()
