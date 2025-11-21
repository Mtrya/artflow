import torch
import sys
import os
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.evaluation import calculate_fid, calculate_clip_score, make_image_grid, visualize_denoising

def test_fid():
    print("Testing FID...")
    # Create dummy images [B, C, H, W] in [0, 1]
    real_images = torch.rand(10, 3, 64, 64)
    fake_images = torch.rand(10, 3, 64, 64)
    
    # FID requires at least 2 images, but usually more. 
    # torchmetrics FID might warn or error if too few, let's see.
    # Actually torchmetrics FID works with features. 
    # We need to make sure we can run it.
    
    try:
        score = calculate_fid(real_images, fake_images, feature=64) # Use small feature dim for speed/memory
        print(f"FID Score: {score}")
    except Exception as e:
        print(f"FID failed: {e}")

def test_clip():
    print("Testing CLIP Score...")
    images = torch.rand(4, 3, 64, 64)
    prompts = ["a random image"] * 4
    
    try:
        score = calculate_clip_score(images, prompts)
        print(f"CLIP Score: {score}")
    except Exception as e:
        print(f"CLIP Score failed: {e}")

def test_grid():
    print("Testing Image Grid...")
    images = torch.rand(16, 3, 64, 64)
    save_path = "output/test_grid.png"
    
    try:
        make_image_grid(images, save_path=save_path)
        if os.path.exists(save_path):
            print(f"Grid saved to {save_path}")
        else:
            print("Grid file not found")
    except Exception as e:
        print(f"Grid failed: {e}")

def test_denoising():
    print("Testing Denoising Visualization...")
    steps = []
    for i in range(10):
        steps.append(torch.rand(2, 3, 64, 64)) # 2 samples, 10 steps
        
    save_path = "output/test_denoising.png"
    try:
        visualize_denoising(steps, save_path)
        if os.path.exists(save_path):
            print(f"Denoising visualization saved to {save_path}")
        else:
            print("Denoising file not found")
    except Exception as e:
        print(f"Denoising failed: {e}")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    test_fid()
    test_clip()
    test_grid()
    test_denoising()
    print("Done.")
