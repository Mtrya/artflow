"""
Upload ArtFlow model to HuggingFace Hub as a standalone repo.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload ArtFlow model to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to ema_weights.pt")
    parser.add_argument("--repo_id", required=True, help="HF Hub repo ID")
    parser.add_argument("--config", help="Optional config JSON to override inferred")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--vae_repo", default="REPA-E/e2e-qwenimage-vae", help="VAE repo ID")
    parser.add_argument("--text_encoder_repo", default="Qwen/Qwen3-0.6B", help="Text encoder repo ID")
    parser.add_argument("--solver", default="euler", help="Default solver (euler/heun)")
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    checkpoint_path = Path(args.checkpoint).resolve()
    src_dir = Path(__file__).parent.parent / "src"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Copy only inference-necessary source files
        artflow_dir = tmpdir / "artflow"
        artflow_dir.mkdir()

        # Define minimal set of files needed for inference
        inference_files = [
            "models/__init__.py",
            "models/artflow.py",
            "models/dit_blocks.py",
            "pipeline/__init__.py",
            "pipeline/artflow_pipeline.py",
            "utils/__init__.py",
            "utils/encode_text.py",
            "utils/vae_codec.py",
            "flow/__init__.py",
            "flow/solvers.py",
            "flow/paths.py",
        ]

        for rel_path in inference_files:
            src_file = src_dir / rel_path
            if src_file.exists():
                dest = artflow_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_file, dest)

        # Root __init__.py
        (artflow_dir / "__init__.py").write_text(
            '"""ArtFlow - Flow Matching DiT for Artistic Image Generation"""\n\n'
            "from .models.artflow import ArtFlow\n"
            "from .pipeline.artflow_pipeline import ArtFlowPipeline, ArtFlowPipelineOutput\n\n"
            '__version__ = "0.1.0"\n'
            '__all__ = ["ArtFlow", "ArtFlowPipeline", "ArtFlowPipelineOutput"]\n'
        )

        # pyproject.toml for pip install
        (tmpdir / "pyproject.toml").write_text(
            '[build-system]\n'
            'requires = ["setuptools>=61.0", "wheel"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            '[project]\n'
            'name = "artflow"\n'
            'version = "0.1.0"\n'
            'description = "Flow Matching DiT for Artistic Image Generation"\n'
            'readme = "README.md"\n'
            'license = {text = "MIT"}\n'
            'requires-python = ">=3.9"\n'
            'dependencies = [\n'
            '    "torch>=2.0",\n'
            '    "diffusers>=0.30",\n'
            '    "transformers>=4.40",\n'
            '    "huggingface-hub",\n'
            '    "numpy",\n'
            '    "Pillow",\n'
            ']\n'
        )

        # MANIFEST.in
        (tmpdir / "MANIFEST.in").write_text(
            "recursive-include artflow *.py\n"
            "include transformer_config.json\n"
            "include model.safetensors\n"
            "include ema_weights.pt\n"
        )

        # Save checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "module" in checkpoint:
            state_dict = checkpoint["module"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        try:
            from safetensors.torch import save_file
            save_file(state_dict, tmpdir / "model.safetensors")
            weights_file = "model.safetensors"
        except ImportError:
            torch.save(state_dict, tmpdir / "ema_weights.pt")
            weights_file = "ema_weights.pt"

        print(f"Saved weights: {weights_file}")

        # Save config
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
        else:
            config = infer_config_from_state_dict(state_dict)

        #config["vae_repo"] = args.vae_repo
        #config["text_encoder_repo"] = args.text_encoder_repo
        #config["solver"] = args.solver
        #config["model_type"] = "artflow"

        with open(tmpdir / "transformer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Minimal README (edit manually on HF Hub)
        (tmpdir / "README.md").write_text(
            f"# ArtFlow\n\n"
            f"Configuration: {config.get('hidden_size')} hidden size, "
            f"{config.get('double_stream_depth', 0) + config.get('single_stream_depth', 0)} blocks\n\n"
            f"See source code in `artflow/` directory.\n"
        )

        # Upload
        print(f"\nUploading to https://huggingface.co/{args.repo_id}...")
        api.upload_folder(folder_path=str(tmpdir), repo_id=args.repo_id, repo_type="model")

    print(f"\nUploaded to https://huggingface.co/{args.repo_id}")
    print(f"\nInstall: pip install --no-cache-dir 'git+https://huggingface.co/{args.repo_id}'")
    print(f"Use: from artflow import ArtFlowPipeline")


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model configuration from checkpoint state dict."""
    config = {}

    config["patch_size"] = state_dict["x_embedder.weight"].shape[2]
    config["in_channels"] = state_dict["x_embedder.weight"].shape[1]
    config["hidden_size"] = state_dict["x_embedder.weight"].shape[0]

    hidden_size = config["hidden_size"]

    if hidden_size >= 1152:
        config["num_heads"] = 16
    elif hidden_size >= 768:
        config["num_heads"] = 12
    elif hidden_size >= 640:
        config["num_heads"] = 10
    else:
        config["num_heads"] = 8

    block_indices = set()
    for key in state_dict.keys():
        if key.startswith("blocks."):
            idx = int(key.split(".")[1])
            block_indices.add(idx)
    total_blocks = max(block_indices) + 1 if block_indices else 0

    has_txt_mlp = any("txt_mlp" in k for k in state_dict.keys())
    if has_txt_mlp:
        double_blocks = set()
        for key in state_dict.keys():
            if "txt_mlp" in key:
                idx = int(key.split(".")[1])
                double_blocks.add(idx)
        config["double_stream_depth"] = len(double_blocks)
        config["single_stream_depth"] = total_blocks - len(double_blocks)
    else:
        config["double_stream_depth"] = 0
        config["single_stream_depth"] = total_blocks

    c_mlp_input = state_dict["c_mlp.0.weight"].shape[1]
    config["conditioning_scheme"] = "fused" if c_mlp_input > hidden_size else "pure"

    config["qkv_bias"] = (
        "blocks.0.attn.qkv.bias" in state_dict
        or "blocks.0.attn.qkv_img.bias" in state_dict
    )

    if "blocks.0.mlp.up_proj.weight" in state_dict or "blocks.0.mlp_img.up_proj.weight" in state_dict:
        config["ffn_type"] = "gated"
    else:
        config["ffn_type"] = "standard"

    config["mlp_ratio"] = 2.67
    config["rope_scaling_type"] = "none"
    config["rope_scaling_factor"] = 1.0

    return config


if __name__ == "__main__":
    main()
