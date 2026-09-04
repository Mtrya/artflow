"""
Fixed-prompt sample grids for cross-arm visual comparability.

Every eval uses the same prompt suite and per-prompt deterministic seeds, so
grid images are directly comparable across steps and across ablation arms.
Bucket shapes are derived from the eval dataset itself (never hand-written),
which keeps generation resolutions consistent with training data.
"""

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

import swanlab

from ..flow.solvers import sample_ode
from ..utils.encode_text import encode_text
from ..utils.vae_codec import get_vae_stats
from .visualize import make_image_grid, format_prompt_caption

# Module-level cache: bucket_id -> (h_lat, w_lat), keyed by dataset path
_BUCKET_SHAPE_CACHE: Dict[str, Dict[int, Tuple[int, int]]] = {}


def load_prompt_suite(path: str) -> List[Dict[str, Any]]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def prompt_seed(prompt_id: str) -> int:
    """Deterministic per-prompt noise seed."""
    return int(hashlib.md5(prompt_id.encode("utf-8")).hexdigest()[:8], 16)


def bucket_shapes_from_dataset(dataset_path: str, scan_rows: int = 2000) -> Dict[int, Tuple[int, int]]:
    """Map resolution_bucket_id -> (h_lat, w_lat) from actual eval data."""
    if dataset_path in _BUCKET_SHAPE_CACHE:
        return _BUCKET_SHAPE_CACHE[dataset_path]

    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    scan = dataset.select(range(min(scan_rows, len(dataset))))
    shapes: Dict[int, Tuple[int, int]] = {}
    for item in scan:
        bid = int(item["resolution_bucket_id"])
        if bid not in shapes:
            z = item["latents"]
            shapes[bid] = (int(z.shape[-2]), int(z.shape[-1]))
    _BUCKET_SHAPE_CACHE[dataset_path] = shapes
    return shapes


def _aspect(prompt: Dict[str, Any]) -> float:
    """Prompt aspect as w/h (e.g. '3:4' -> 0.75)."""
    spec = prompt.get("aspect", "1:1")
    w, h = spec.split(":")
    return float(w) / float(h)


def assign_bucket(prompt: Dict[str, Any], shapes: Dict[int, Tuple[int, int]]) -> int:
    pa = _aspect(prompt)
    best, best_dist = None, float("inf")
    for bid, (h_lat, w_lat) in shapes.items():
        dist = abs(math.log((w_lat / h_lat) / pa))
        if dist < best_dist:
            best, best_dist = bid, dist
    return best


@torch.no_grad()
def run_prompt_grid_eval(
    accelerator,
    model: torch.nn.Module,
    vae_path: str,
    save_path: str,
    current_step: int,
    text_encoder,
    tokenizer,
    pooling: bool,
    exit_layer: Optional[int] = None,
    prompts_path: str = "assets/eval/prompts_v1.jsonl",
    eval_dataset_path: str = "./precomputed_dataset/light-eval@256p",
    batch_size: int = 8,
    ode_steps: int = 50,
) -> None:
    """Generate fixed-seed samples for the prompt suite and log grids."""
    from diffusers import AutoencoderKLQwenImage

    print(f"Running prompt-grid evaluation at step {current_step}...")
    was_training = model.training
    model.eval()

    device = accelerator.device
    num_processes = getattr(accelerator, "num_processes", 1)
    process_index = getattr(accelerator, "process_index", 0)

    prompts = load_prompt_suite(prompts_path)
    shapes = bucket_shapes_from_dataset(eval_dataset_path)
    for p in prompts:
        p["_bucket"] = assign_bucket(p, shapes)

    # Shard prompts across ranks
    local_prompts = [
        p for i, p in enumerate(prompts) if i % max(1, num_processes) == process_index
    ]

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    vae_mean, vae_std = get_vae_stats(vae_path, device=device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    # Group local prompts by bucket for batched generation
    by_bucket: Dict[int, List[Dict[str, Any]]] = {}
    for p in local_prompts:
        by_bucket.setdefault(p["_bucket"], []).append(p)

    results: List[Tuple[Dict[str, Any], torch.Tensor]] = []  # (prompt, image cpu float)
    for bid, plist in sorted(by_bucket.items()):
        h_lat, w_lat = shapes[bid]
        for start in range(0, len(plist), batch_size):
            chunk = plist[start : start + batch_size]
            txt, txt_mask, txt_pooled = encode_text(
                [p["text"] for p in chunk],
                text_encoder,
                tokenizer,
                pooling,
                exit_layer=exit_layer,
            )
            noise = torch.stack(
                [
                    torch.randn(
                        (16, h_lat, w_lat),
                        generator=torch.Generator(device="cpu").manual_seed(
                            prompt_seed(p["id"])
                        ),
                    )
                    for p in chunk
                ]
            ).to(device, torch.bfloat16)

            def model_fn(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask):
                if isinstance(t, float):
                    t = torch.tensor(t, device=x.device).expand(x.shape[0])
                return model(x, t, txt, txt_pooled, txt_mask)

            with accelerator.autocast():
                samples = sample_ode(
                    model_fn, noise, steps=ode_steps, t_start=0.0, t_end=1.0
                )

            samples = samples.to(dtype=torch.bfloat16)
            samples = samples * vae_std + vae_mean
            images = vae.decode(samples.unsqueeze(2)).sample.squeeze(2)
            images = torch.clamp((images + 1) / 2, 0, 1).cpu().float()
            for p, img in zip(chunk, images):
                results.append((p, img))

    # Gather across ranks
    if num_processes > 1 and hasattr(accelerator, "gather_object"):
        gathered = accelerator.gather_object(results)
    else:
        gathered = [results]

    if accelerator.is_main_process:
        os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
        all_results = [r for part in gathered for r in part]
        # Regroup by bucket for grids
        grids: Dict[int, List[Tuple[Dict[str, Any], torch.Tensor]]] = {}
        for p, img in all_results:
            grids.setdefault(p["_bucket"], []).append((p, img))

        for bid, items in sorted(grids.items()):
            h_lat, w_lat = shapes[bid]
            images = torch.stack([img for _, img in items])
            captions = [p["text"] for p, _ in items]
            grid_path = os.path.join(
                save_path,
                "samples",
                f"grid_step_{current_step:06d}_bucket{bid}_{h_lat * 8}x{w_lat * 8}.png",
            )
            _ = make_image_grid(images, save_path=grid_path, normalize=True, value_range=(0, 1))
            caption = format_prompt_caption(captions[:4]) if captions else ""
            media = (
                swanlab.Image(grid_path, caption=caption)
                if caption
                else swanlab.Image(grid_path)
            )
            accelerator.log({f"grid/bucket{bid}_{h_lat * 8}x{w_lat * 8}": media}, step=current_step)
        print(f"Logged {len(grids)} prompt grids at step {current_step}")

    del vae
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if was_training:
        model.train()
