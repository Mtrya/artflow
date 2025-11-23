"""
Script to verify precomputed datasets.
Calculates statistics and generates samples.
"""

import argparse
import os
import sys
import random
import torch
import numpy as np
from datasets import load_from_disk
from PIL import Image
from collections import Counter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="Verify precomputed dataset")
    parser.add_argument("dataset_path", type=str, help="Path to precomputed dataset")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE model for image reconstruction")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="output/verification_samples", help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for VAE")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist.")
        return

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    
    print(f"\n=== Basic Info ===")
    print(f"Number of examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    
    # Check required columns
    required_columns = ["latents", "captions", "resolution_bucket_id"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    # Statistics
    print(f"\n=== Statistics ===")
    
    # Resolution Buckets
    if "resolution_bucket_id" in dataset.column_names:
        bucket_ids = dataset["resolution_bucket_id"]
        # Convert to python integers if they are tensors
        if hasattr(bucket_ids[0], "item"):
             bucket_ids = [x.item() for x in bucket_ids]
        
        bucket_counts = Counter(bucket_ids)
        print("Resolution Bucket Distribution:")
        for bucket_id, count in sorted(bucket_counts.items()):
            print(f"  Bucket {bucket_id}: {count} ({count/len(dataset)*100:.2f}%)")
            
    # Captions
    if "captions" in dataset.column_names:
        num_captions = [len(x) for x in dataset["captions"]]
        avg_captions = sum(num_captions) / len(num_captions)
        print(f"Average captions per image: {avg_captions:.2f}")
        print(f"Min captions: {min(num_captions)}")
        print(f"Max captions: {max(num_captions)}")
        
    # Latents
    if "latents" in dataset.column_names:
        # Check a subset for latent stats to avoid OOM/slow processing
        print("Calculating latent statistics (on first 1000 samples)...")
        subset_size = min(1000, len(dataset))
        
        # Iterate through individual examples to handle torch format properly
        all_values = []
        for i in range(subset_size):
            try:
                example = dataset[i]
                latent = example.get("latents")
                
                if latent is None:
                    continue
                    
                if isinstance(latent, torch.Tensor):
                    all_values.append(latent.flatten())
                elif isinstance(latent, list):
                    all_values.append(torch.tensor(latent).flatten())
                elif isinstance(latent, np.ndarray):
                    all_values.append(torch.from_numpy(latent).flatten())
            except Exception as e:
                # Skip problematic examples
                continue
        
        if all_values:
            latents_subset = torch.cat(all_values)
        else:
            latents_subset = torch.tensor([])
             
        if latents_subset.numel() > 0:
            if latents_subset.dim() > 1:
                print(f"Latent shape (sample): {latents_subset[0].shape}")
            else:
                print(f"Latent stats calculated on flattened subset of {subset_size} samples.")
                
            print(f"Latent Min: {latents_subset.min().item():.4f}")
            print(f"Latent Max: {latents_subset.max().item():.4f}")
            print(f"Latent Mean: {latents_subset.mean().item():.4f}")
            print(f"Latent Std: {latents_subset.std().item():.4f}")
            
            if torch.isnan(latents_subset).any():
                print("WARNING: NaNs found in latents!")
            if torch.isinf(latents_subset).any():
                print("WARNING: Infs found in latents!")
        else:
            print("Warning: Latents subset is empty.")

    # Visualization
    if args.num_samples > 0:
        print(f"\n=== Visualization ({args.num_samples} samples) ===")
        os.makedirs(args.output_dir, exist_ok=True)
        
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        
        # Load VAE if provided
        vae = None
        if args.vae_path:
            try:
                from diffusers import AutoencoderKLQwenImage
                from src.utils.vae_codec import decode_latents
                print(f"Loading VAE from {args.vae_path}...")
                vae = AutoencoderKLQwenImage.from_pretrained(
                    args.vae_path,
                    torch_dtype=torch.bfloat16,
                    device_map=args.device
                )
                vae.eval()
            except Exception as e:
                print(f"Failed to load VAE: {e}")
        
        for i, idx in enumerate(indices):
            example = dataset[idx]
            print(f"\nSample {i+1} (Index {idx}):")
            print(f"  Bucket ID: {example.get('resolution_bucket_id', 'N/A')}")
            print(f"  Captions: {example.get('captions', [])}")
            
            if vae and "latents" in example:
                latent = torch.tensor(example["latents"]).to(args.device).to(torch.bfloat16)
                # Add batch dim
                latent = latent.unsqueeze(0) 
                
                try:
                    images = decode_latents(latent, vae)
                    img_path = os.path.join(args.output_dir, f"sample_{idx}.png")
                    images[0].save(img_path)
                    print(f"  Saved image reconstruction to {img_path}")
                except Exception as e:
                    print(f"  Failed to decode image: {e}")

if __name__ == "__main__":
    main()
