"""
Fixed-prompt sample grids for cross-run visual comparability.

Every eval uses the same prompt suite and per-prompt deterministic seeds, so
grid images are directly comparable across steps and across ablation runs.
Bucket shapes are derived from the eval dataset itself (never hand-written),
which keeps generation resolutions consistent with training data.
"""

import json
import math
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from accelerate.utils import gather_object

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
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    for prompt in prompts:
        resolved_prompt_seed(prompt)
        if "scene_id" in prompt:
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", prompt["scene_id"]):
                raise ValueError("scene_id must be a safe filename component")
            if prompt.get("lang") not in ("zh", "en") or prompt.get("variant") not in ("short", "long"):
                raise ValueError("scene prompts require lang=zh/en and variant=short/long")
    scenes = {}
    for prompt in prompts:
        if "scene_id" in prompt:
            scenes.setdefault(prompt["scene_id"], []).append(prompt)
    for scene, records in scenes.items():
        variants = {(p["lang"], p["variant"]) for p in records}
        if len(records) != 4 or len(variants) != 4:
            raise ValueError(f"scene {scene} requires exactly four language/length variants")
        if len({resolved_prompt_seed(p) for p in records}) != 1:
            raise ValueError(f"scene {scene} variants must share a seed")
        if len({_aspect(p) for p in records}) != 1:
            raise ValueError(f"scene {scene} variants must share an aspect ratio")
    return prompts


def resolved_prompt_seed(prompt: Dict[str, Any]) -> int:
    """Require explicit noise seeds so prompt edits cannot change the noise."""
    seed = prompt.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("prompt seed must be an integer in [0, 2**63)")
    return seed


def grid_due(step: int, interval: int, extra_steps: List[int]) -> bool:
    """Global-step triggers stay fixed across crash resumes; overlaps run once."""
    return step in extra_steps or (interval > 0 and step > 0 and step % interval == 0)


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


def load_prompt_plan(
    prompts_path: str, eval_dataset_path: str
) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[int, int]]]:
    """Load the fixed suite and attach each prompt's nearest resolution bucket.

    Returns the prompt records (each carrying ``_bucket``) and the bucket
    shapes read from the eval dataset, so a caller can either sample the whole
    suite or shard it across processes and still sample each prompt exactly as
    training does.
    """
    prompts = load_prompt_suite(prompts_path)
    shapes = bucket_shapes_from_dataset(eval_dataset_path)
    for prompt in prompts:
        prompt["_bucket"] = assign_bucket(prompt, shapes)
    return prompts, shapes


@torch.no_grad()
def sample_prompt_images(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    vae_mean: torch.Tensor,
    vae_std: torch.Tensor,
    text_encoder,
    tokenizer,
    prompts: List[Dict[str, Any]],
    shapes: Dict[int, Tuple[int, int]],
    *,
    batch_size: int,
    ode_steps: int,
    pooling: bool,
    device: torch.device,
    exit_layer: Optional[int] = None,
    amp_context: Optional[Callable[[], Any]] = None,
) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
    """Sample one deterministic image per prompt, grouped by resolution bucket.

    This is the per-prompt sampling shared by the training-time prompt grid and
    the offline blind-panel generator, so the same checkpoint and prompt yield
    the same noise in both: the seed is explicit in the prompt record, the
    sampler is ``sample_ode`` (Euler) from t=0 to t=1, text conditioning is
    ``encode_text`` and latents are decoded with the VAE's own statistics.

    Each prompt must carry its ``_bucket`` (see ``load_prompt_plan``) and every
    bucket must appear in ``shapes``.

    Args:
        vae_mean, vae_std: latent statistics from ``get_vae_stats``, already
            cast by the caller to the dtype used for decoding.
        amp_context: zero-argument context-manager factory wrapped around the
            solver call (e.g. ``accelerator.autocast``); ``None`` disables
            autocasting.

    Returns:
        ``(prompt, image)`` pairs in bucket order, image a ``[3, H, W]``
        float32 CPU tensor in [0, 1].
    """
    context_factory = amp_context if amp_context is not None else nullcontext

    # Group prompts by bucket for batched generation
    by_bucket: Dict[int, List[Dict[str, Any]]] = {}
    for prompt in prompts:
        by_bucket.setdefault(prompt["_bucket"], []).append(prompt)

    results: List[Tuple[Dict[str, Any], torch.Tensor]] = []
    for bucket_id, bucket_prompts in sorted(by_bucket.items()):
        h_lat, w_lat = shapes[bucket_id]
        for start in range(0, len(bucket_prompts), batch_size):
            chunk = bucket_prompts[start : start + batch_size]
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
                            resolved_prompt_seed(p)
                        ),
                    )
                    for p in chunk
                ]
            ).to(device, torch.bfloat16)

            def model_fn(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask):
                if isinstance(t, float):
                    t = torch.tensor(t, device=x.device).expand(x.shape[0])
                return model(x, t, txt, txt_pooled, txt_mask)

            with context_factory():
                samples = sample_ode(
                    model_fn, noise, steps=ode_steps, t_start=0.0, t_end=1.0
                )

            samples = samples.to(dtype=torch.bfloat16)
            samples = samples * vae_std + vae_mean
            images = vae.decode(samples.unsqueeze(2)).sample.squeeze(2)
            images = torch.clamp((images + 1) / 2, 0, 1).cpu().float()
            for prompt, image in zip(chunk, images):
                results.append((prompt, image))
    return results


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
    prompts_path: str,
    exit_layer: Optional[int] = None,
    eval_dataset_path: str = "./precomputed_dataset/light-eval@256p",
    batch_size: int = 8,
    ode_steps: int = 50,
    weights: str = "unspecified",
) -> None:
    """Generate fixed-seed samples for the prompt suite and log grids."""
    from diffusers import AutoencoderKLQwenImage

    print(f"Running prompt-grid evaluation at step {current_step}...")
    was_training = model.training
    model.eval()

    device = accelerator.device
    num_processes = getattr(accelerator, "num_processes", 1)
    process_index = getattr(accelerator, "process_index", 0)

    prompts, shapes = load_prompt_plan(prompts_path, eval_dataset_path)

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

    results = sample_prompt_images(
        model,
        vae,
        vae_mean,
        vae_std,
        text_encoder,
        tokenizer,
        local_prompts,
        shapes,
        batch_size=batch_size,
        ode_steps=ode_steps,
        pooling=pooling,
        device=device,
        exit_layer=exit_layer,
        amp_context=accelerator.autocast,
    )

    # Accelerate exposes this as a utility, not an Accelerator method.
    # It concatenates rank-local lists and is a no-op on a single process.
    all_results = gather_object(results)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
        # Gathering groups by rank; restore suite order so changing GPU count
        # does not rearrange the images within each bucket.
        prompt_order = {p["id"]: i for i, p in enumerate(prompts)}
        all_results.sort(key=lambda result: prompt_order[result[0]["id"]])
        # New panels pair language/length variants per scene. Old suites keep
        # their bucket grids. Include the bucket in the key to avoid stacking
        # differently shaped images from malformed scene records.
        grids = {}
        for p, img in all_results:
            key = (p.get("scene_id", ""), p["_bucket"])
            grids.setdefault(key, []).append((p, img))

        for (scene, bid), items in sorted(grids.items()):
            h_lat, w_lat = shapes[bid]
            if scene:
                order = {("zh", "short"): 0, ("en", "short"): 1,
                         ("zh", "long"): 2, ("en", "long"): 3}
                items.sort(key=lambda item: order[(item[0]["lang"], item[0]["variant"])])
            images = torch.stack([img for _, img in items])
            captions = [
                f"{p['lang']} / {p['variant']} / seed={resolved_prompt_seed(p)}: "
                f"{p['text'][:240]}" if scene else p["text"]
                for p, _ in items
            ]
            label = f"{scene}_bucket{bid}" if scene else f"bucket{bid}"
            grid_path = os.path.join(
                save_path,
                "samples",
                f"grid_step_{current_step:06d}_{label}_{h_lat * 8}x{w_lat * 8}.png",
            )
            _ = make_image_grid(images, cols=2 if scene else None,
                                save_path=grid_path, normalize=True, value_range=(0, 1))
            caption = format_prompt_caption(captions, limit=len(captions))
            media = (
                swanlab.Image(grid_path, caption=caption)
                if caption
                else swanlab.Image(grid_path)
            )
            accelerator.log({f"grid/{label}_{h_lat * 8}x{w_lat * 8}": media}, step=current_step)
        # Preserve the full text and exact sampling identity, not only the
        # shortened grid caption visible in SwanLab.
        with open(os.path.join(save_path, "samples", f"panel_step_{current_step:06d}.json"), "w") as handle:
            json.dump({"step": current_step, "solver": "euler", "ode_steps": ode_steps,
                       "cfg_scale": 1.0, "precision": "bf16",
                       "solver_precision": "fp32", "timestep_precision": "fp32",
                       "exit_layer": exit_layer,
                       "weights": weights, "pooling": pooling,
                       "eval_dataset_path": eval_dataset_path,
                       "prompts": [{**p, "seed": resolved_prompt_seed(p),
                                    "latent_shape": shapes[p["_bucket"]]} for p in prompts]},
                      handle, ensure_ascii=False, indent=2)
        print(f"Logged {len(grids)} prompt grids at step {current_step}")

    del vae
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if was_training:
        model.train()
