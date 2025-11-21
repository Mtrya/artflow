"""
Main training script for ArtFlow.
Stage 0: unconditional image generation with WikiArt Monet subset, for algorithm ablation
"""

import argparse
import os
import sys
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.artflow_uncond import ArtFlowUncond
from src.flow.paths import ScoreMatchingDiffusion, FlowMatchingDiffusion, FlowMatchingOT
from src.flow.solvers import sample_ode, ScoreMatchingODE
from src.utils.precompute_engine import precompute
from src.utils.evaluation import run_evaluation_stage0
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser(description="Train ArtFlow model")
    parser.add_argument("--run_name", type=str, default="artflow_run", help="Name of the run")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="kaupane/wikiart-captions-monet", help="Dataset name")
    parser.add_argument("--resolution", type=int, default=128, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--algorithm", type=str, choices=["sm-diffusion", "fm-diffusion", "fm-ot"], default="fm-ot", help="Algorithm to use")
    parser.add_argument("--vae_path", type=str, default="REPA-E/e2e-qwenimage-vae", help="Path to VAE")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Epochs between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=20, help="Epochs between evaluation")
    parser.add_argument("--num_eval_samples", type=int, default=9, help="Number of samples to generate during evaluation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--model_hidden_size", type=int, default=512, help="Hidden size of the model")
    parser.add_argument("--model_depth", type=int, default=8, help="Depth of the model")
    parser.add_argument("--model_num_heads", type=int, default=8, help="Number of heads in the model")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="wandb")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="artflow", 
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}}
        )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Starting run: {args.run_name}")
        print(f"Algorithm: {args.algorithm}")

    # 1. Load and Preprocess Data
    # For Stage 0, we use fixed resolution and no text conditioning (unconditional)
    if accelerator.is_main_process:
        print("Loading dataset...")
    
    # Load raw dataset
    dataset = load_dataset(args.dataset_name, split="train")
    
    if accelerator.is_main_process:
        print("Precomputing dataset (VAE encoding)...")
    
    precomputed_dataset = precompute(
        dataset=dataset,
        stage=0.5, # Irrelevant for unconditional
        caption_fields=[], 
        text_encoder_path="", # Not used
        pooling=False,
        vae_path=args.vae_path,
        resolution_buckets={1: (args.resolution, args.resolution)}, # Fixed resolution
        resolution_probs=[1.0],
        do_caption_scheduling=False,
        do_data_augmentation=False,
        preprocessing_batch_size=32,
        vae_batch_size=16,
        text_batch_size=1
    )
    
    # Set format for pytorch
    precomputed_dataset.set_format("torch", columns=["latents"])
    
    train_dataloader = DataLoader(
        precomputed_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 2. Model Setup
    model = ArtFlowUncond(
        in_channels=16,
        hidden_size=args.model_hidden_size,
        depth=args.model_depth,
        num_heads=args.model_num_heads,
        patch_size=2
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
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")
        
        total_loss = 0.0
        for batch in progress_bar:
            latents = batch["latents"] # [B, 16, H, W]
            
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
            print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
            accelerator.log({"train_loss": avg_loss}, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"{args.run_name}/checkpoint_epoch_{epoch+1:03d}")
                accelerator.save_state(checkpoint_dir)

            # Evaluation
            if (epoch + 1) % args.eval_interval == 0:
                run_evaluation_stage0(
                    accelerator=accelerator,
                    model=model,
                    args=args,
                    epoch=epoch,
                    train_dataloader=train_dataloader,
                    algorithm=algorithm,
                    sample_ode_fn=sample_ode,
                    ScoreMatchingODE_cls=ScoreMatchingODE
                )

    accelerator.end_training()
    print("Training finished.")

if __name__ == "__main__":
    main()
