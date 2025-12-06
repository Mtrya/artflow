"""
Training script for reward model.

Usage:
    python -m src.train.train_rm \\
        --dataset_path ./scored_images/scored_dataset \\
        --output_dir ./output/reward_model \\
        --encoder_name OFA-Sys/chinese-clip-vit-large-patch14-336px \\
        --batch_size 32 \\
        --epochs 50 \\
        --lr 1e-4
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_from_disk
from PIL import Image
from tqdm.auto import tqdm
import json

from src.models.reward_model import RewardModel, extract_clip_features
from transformers import ChineseCLIPModel, CLIPProcessor


class ScoredImageDataset(Dataset):
    """Dataset for scored images."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_path"]
        score = item["score"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Convert to tensor [0, 1]
        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms as T
            def to_uint8(image):
                tensor = T.ToTensor()(image) * 255.0
                tensor = tensor.clamp(0, 255).to(torch.uint8)
                return tensor
                
            image = to_uint8(image)

        # Normalize score to [0, 1]
        score = torch.tensor(score / 5.0, dtype=torch.float32)

        return {"image": image, "score": score}


def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([item["image"] for item in batch])
    scores = torch.stack([item["score"] for item in batch])
    return {"images": images, "scores": scores}


def train_one_epoch(
    model: nn.Module,
    clip_model: nn.Module,
    clip_processor: any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    feature_layers: list,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    clip_model.eval()  # CLIP stays frozen

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["images"].to(device)
        scores = batch["scores"].to(device)

        # Extract features (no grad for CLIP)
        with torch.no_grad():
            features = extract_clip_features(
                images, clip_model, clip_processor, feature_layers, device
            )

        # Forward (only MLP head has gradients)
        pred_scores = model(features)
        loss = criterion(pred_scores, scores)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        mae = (pred_scores - scores).abs().mean()

        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mae": f"{mae.item():.4f}"})

    return {
        "train_loss": total_loss / num_batches,
        "train_mae": total_mae / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    clip_model: nn.Module,
    clip_processor: any,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    feature_layers: list,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    clip_model.eval()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["images"].to(device)
        scores = batch["scores"].to(device)

        # Extract features
        features = extract_clip_features(
            images, clip_model, clip_processor, feature_layers, device
        )

        # Forward
        pred_scores = model(features)
        loss = criterion(pred_scores, scores)

        # Metrics
        mae = (pred_scores - scores).abs().mean()

        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1

    return {
        "val_loss": total_loss / num_batches,
        "val_mae": total_mae / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to scored dataset (output of export-scores)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/reward_model",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP encoder name",
    )
    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs="+",
        default=[12, 18, 23],
        help="Layer indices to extract features from",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of MLP head",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config (will be extended with feature_dim and num_layers after model creation)
    config_dict = vars(args).copy()

    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset size: {len(dataset)}")

    # Create torch dataset
    torch_dataset = ScoredImageDataset(dataset)

    # Split into train/val
    val_size = int(len(torch_dataset) * args.val_split)
    train_size = len(torch_dataset) - val_size
    train_dataset, val_dataset = random_split(
        torch_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Load CLIP model (frozen)
    print(f"Loading CLIP model: {args.encoder_name}")
    clip_model = ChineseCLIPModel.from_pretrained(args.encoder_name)
    clip_processor = CLIPProcessor.from_pretrained(args.encoder_name)
    clip_model = clip_model.to(args.device)
    clip_model.eval()
    
    # Freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Get feature dimension from CLIP config
    if hasattr(clip_model.config, 'vision_config'):
        feature_dim = clip_model.config.vision_config.hidden_size
    else:
        feature_dim = clip_model.config.hidden_size
    
    print(f"CLIP feature dimension: {feature_dim}")
    print(f"Extracting from layers: {args.feature_layers}")
    
    # Create reward model (lightweight MLP head only)
    print("Creating reward model...")
    model = RewardModel(
        feature_dim=feature_dim,
        num_layers=len(args.feature_layers),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = model.to(args.device)

    print(f"Trainable params (MLP head only): {model.get_num_trainable_params():,}")
    print(f"Total CLIP params (frozen): {sum(p.numel() for p in clip_model.parameters()):,}")
    
    # Update and save config with feature dimensions
    config_dict["feature_dim"] = feature_dim
    config_dict["num_layers"] = len(args.feature_layers)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Training loop
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, clip_model, clip_processor, train_loader, 
            optimizer, criterion, args.device, epoch, args.feature_layers
        )

        # Validate
        val_metrics = validate(
            model, clip_model, clip_processor, val_loader, 
            criterion, args.device, args.feature_layers
        )

        # Update scheduler
        scheduler.step()

        # Combine metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        history.append(metrics)

        # Print
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train MAE: {train_metrics['train_mae']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val MAE: {val_metrics['val_mae']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "config": vars(args),
                },
                output_dir / "best_model.pt",
            )
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "config": config_dict,
                },
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
