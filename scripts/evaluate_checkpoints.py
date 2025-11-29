"""
Evaluate checkpoints using run_evaluation_heavy.
Thin wrapper that discovers checkpoints and calls run_evaluation_heavy.
"""

import argparse
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.evaluation import run_evaluation_heavy


def discover_checkpoints(checkpoint_pattern: str) -> list:
    """Discover checkpoint files matching pattern."""
    checkpoints = sorted(glob(checkpoint_pattern))
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        required=True,
        help="Glob pattern for checkpoints, e.g., 'output/baseline/checkpoint_step_*/ema_weights.pt'",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help='Model config as JSON string, e.g., \'{"in_channels":16,"hidden_size":768,...}\'',
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="REPA-E/e2e-qwenimage-vae",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Qwen/Qwen3-VL-2B",
        help="Path to text encoder",
    )
    parser.add_argument(
        "--pooling",
        action="store_true",
        default=False,
        help="Use pooled text embeddings",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./precomputed_dataset/heavy-eval@256p",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--num_fid_samples",
        type=int,
        default=2000,
        help="Number of samples for FID",
    )
    parser.add_argument(
        "--num_clip_samples",
        type=int,
        default=2000,
        help="Number of samples for CLIP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip checkpoints with existing results",
    )
    args = parser.parse_args()

    # Parse model config
    model_config = json.loads(args.model_config)

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoint_pattern)
    print(f"Found {len(checkpoints)} checkpoints")

    for checkpoint_path in checkpoints:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        results_file = os.path.join(checkpoint_dir, "evaluation_results.json")

        # Skip if results exist
        if args.skip_existing and os.path.exists(results_file):
            print(f"Skipping {checkpoint_path} (results exist)")
            continue

        print(f"\nEvaluating {checkpoint_path}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            results = run_evaluation_heavy(
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                vae_path=args.vae_path,
                text_encoder_path=args.text_encoder_path,
                pooling=args.pooling,
                save_path=checkpoint_dir,
                dataset_path=args.dataset_path,
                num_fid_samples=args.num_fid_samples,
                num_clip_samples=args.num_clip_samples,
                batch_size=args.batch_size,
                device=args.device,
            )

            # Save results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results: {results}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
