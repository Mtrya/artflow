"""
Stateless dataset precomputation utilities.

Functions:
- precompute: Precompute image latents using a VAE
"""

import concurrent.futures
import gc
import io
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import requests
import torch
from datasets import Dataset
from PIL import Image

from .buckets import get_resolution_bucket
from .captions import clean_caption, format_artist_name


def _fetch_image(image_data: Union[str, Image.Image]) -> Optional[Image.Image]:
    """
    Fetch image from URL or return PIL Image.
    """
    if isinstance(image_data, Image.Image):
        return image_data

    if isinstance(image_data, str):
        try:
            response = requests.get(image_data, timeout=(2, 7))
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image.load()  # Verify it's a valid image
            return image.convert("RGB")
        except Exception:
            return None

    return None


def precompute(
    dataset: Dataset,
    image_field: str,
    caption_fields: List[str],
    vae_path: str,
    resolution_buckets: Dict[int, Tuple[int, int]],
    text_fn: Callable[[str], str] = clean_caption,
    batch_size: int = 50,
    device: str = "cuda",
    non_zh_drop_prob: float = 0.0,
    resolution_tolerence: float = 1.0,
    max_caption_tokens: int = 1024,
) -> Dataset:
    """
    Stateless precomputation of image latents and caption preparation.

    Args:
        dataset: Input dataset.
        image_field: Name of the column containing image objects or URLs.
        caption_fields: List of column names containing captions.
        vae_path: Path to the VAE model.
        resolution_buckets: Dictionary mapping bucket IDs to (width, height).
        text_fn: Function to process a single caption string.
                 If None, no processing is applied.
        batch_size: Batch size for processing.
        device: Device to run VAE on.
        non_zh_drop_prob: Probability of dropping samples where LANGUAGE field
                          is not "zh". Only applies if dataset has LANGUAGE field.
                          Default 0.0 means no language filtering.
        max_caption_tokens: Maximum estimated tokens for any single caption.
                            Samples with longer captions will be dropped.

    Returns:
        Processed dataset with 'latents', 'captions', and 'resolution_bucket_id'.
        'captions' will be a List[str] for each example.
    """
    from diffusers import AutoencoderKLQwenImage
    from ..utils.vae_codec import encode_image
    from .captions import _estimate_token_counts

    # Load VAE
    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, device_map=device
    )
    vae.eval()

    def _process_batch(batch: Dict) -> Dict:
        nonlocal vae
        images_raw = batch[image_field]
        batch_len = len(images_raw)

        # --- Pre-fetch filtering ---
        skip_indices = set()
        processed_captions_map = {}

        # 1. Language filter
        languages = batch.get("LANGUAGE", [None] * batch_len)
        if non_zh_drop_prob > 0.0:
            for idx, lang in enumerate(languages):
                if lang is not None and lang != "zh":
                    if random.random() < non_zh_drop_prob:
                        skip_indices.add(idx)
        dropped_lang = len(skip_indices)

        # 2. Resolution filter (if metadata available)
        widths = batch.get("WIDTH") or batch.get("width", [None] * batch_len)
        heights = batch.get("HEIGHT") or batch.get("height", [None] * batch_len)
        for idx in range(batch_len):
            if idx in skip_indices:
                continue
            width, height = widths[idx], heights[idx]
            if isinstance(width, int) and isinstance(height, int):
                try:
                    _, resolution = get_resolution_bucket(width, height, resolution_buckets)
                    if width < resolution[0] * resolution_tolerence or height < resolution[1] * resolution_tolerence:
                        skip_indices.add(idx)
                except Exception:
                    skip_indices.add(idx)
        dropped_resolution_pre = len(skip_indices) - dropped_lang

        # 3. Caption length filter (drop too short or too long captions)
        for idx in range(batch_len):
            if idx in skip_indices:
                continue

            current_captions = []
            for field in caption_fields:
                if field not in batch or batch[field][idx] is None:
                    continue
                val = batch[field][idx]
                # Simplified caption extraction logic
                items_to_process = []
                if field == "human_caption_hq" and isinstance(val, list) and len(val) > 1 and "value" in val[1]:
                    items_to_process.append(val[1]["value"])
                elif field == "artist":
                    items_to_process.append(format_artist_name(val))
                elif isinstance(val, str):
                    items_to_process.append(val)
                elif isinstance(val, list):
                    items_to_process.extend(item for item in val if isinstance(item, str))

                for item in items_to_process:
                    if text_fn:
                        current_captions.append(text_fn(item))
                    else:
                        current_captions.append(item)

            token_counts = _estimate_token_counts(current_captions)
            if not current_captions or not any(count > 0 for count in token_counts):
                # Drop samples with effectively empty captions
                skip_indices.add(idx)
            elif any(count > max_caption_tokens for count in token_counts):
                # Drop samples with at least one overly long caption
                skip_indices.add(idx)
            else:
                processed_captions_map[idx] = current_captions

        dropped_caption_length = len(skip_indices) - dropped_lang - dropped_resolution_pre
        
        # --- Image Fetching and Processing ---
        dropped_fetch_timeout = 0
        dropped_fetch_error = 0
        dropped_invalid_image = 0
        dropped_resolution_post = 0
        dropped_bucket_error = 0
        dropped_vae = 0

        def _log_dropped_samples():
            total_dropped = len(skip_indices) + dropped_fetch_timeout + dropped_fetch_error + dropped_invalid_image + dropped_resolution_post + dropped_bucket_error + dropped_vae
            if total_dropped > 0:
                print(
                    f"Batch: {total_dropped}/{batch_len} dropped "
                    f"(lang={dropped_lang}, res_pre={dropped_resolution_pre}, caption={dropped_caption_length}, "
                    f"fetch_timeout={dropped_fetch_timeout}, fetch_err={dropped_fetch_error}, invalid_img={dropped_invalid_image}, "
                    f"res_post={dropped_resolution_post}, bucket_err={dropped_bucket_error}, vae={dropped_vae})"
                )

        batches_by_bucket = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_len, 50)) as executor:
            futures = {
                idx: executor.submit(_fetch_image, img_data)
                for idx, img_data in enumerate(images_raw)
                if idx not in skip_indices
            }

            for idx, future in futures.items():
                try:
                    image = future.result(timeout=10.0)
                except concurrent.futures.TimeoutError:
                    dropped_fetch_timeout += 1
                    continue
                except Exception:
                    dropped_fetch_error += 1
                    continue

                if image is None:
                    dropped_invalid_image += 1
                    continue

                width, height = image.size
                try:
                    bucket_id, resolution = get_resolution_bucket(width, height, resolution_buckets)
                    if width < resolution[0] * resolution_tolerence or height < resolution[1] * resolution_tolerence:
                        dropped_resolution_post += 1
                        continue
                    
                    resized_image = image.resize(resolution, Image.BICUBIC)
                    key = (bucket_id, resolution)
                    if key not in batches_by_bucket:
                        batches_by_bucket[key] = []
                    batches_by_bucket[key].append((idx, resized_image))
                except Exception:
                    dropped_bucket_error += 1
                    continue

        if not batches_by_bucket:
            _log_dropped_samples()
            return {"latents": [], "captions": [], "resolution_bucket_id": []}

        latents_map = {}
        bucket_ids_map = {}
        for (bucket_id, resolution), items in batches_by_bucket.items():
            batch_indices = [item[0] for item in items]
            batch_images = [item[1] for item in items]
            try:
                with torch.no_grad():
                    batch_latents = encode_image(batch_images, vae).detach().cpu()
                for i, original_idx in enumerate(batch_indices):
                    latents_map[original_idx] = batch_latents[i]
                    bucket_ids_map[original_idx] = int(bucket_id)
            except Exception as e:
                print(f"Error encoding batch for resolution {resolution}: {e}")
                dropped_vae += len(batch_indices)
        del batches_by_bucket

        if not latents_map:
            _log_dropped_samples()
            return {"latents": [], "captions": [], "resolution_bucket_id": []}

        valid_latents = []
        valid_captions_list = []
        valid_bucket_ids = []
        sorted_indices = sorted(latents_map.keys())

        for original_idx in sorted_indices:
            valid_latents.append(latents_map[original_idx])
            valid_bucket_ids.append(bucket_ids_map[original_idx])
            valid_captions_list.append(processed_captions_map[original_idx])

        _log_dropped_samples()
        del latents_map, bucket_ids_map, processed_captions_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "latents": valid_latents,
            "captions": valid_captions_list,
            "resolution_bucket_id": valid_bucket_ids,
        }

    columns_to_keep = {"latents", "captions", "resolution_bucket_id"}
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

    print("Starting precomputation...")
    processed_dataset = dataset.map(
        _process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=columns_to_remove,
        desc="Precomputing latents and captions",
    )

    print("Cleaning up VAE from GPU memory...")
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processed_dataset = processed_dataset.with_format("torch")
    return processed_dataset