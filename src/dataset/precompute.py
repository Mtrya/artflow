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

    Returns:
        Processed dataset with 'latents', 'captions', and 'resolution_bucket_id'.
        'captions' will be a List[str] for each example.
    """
    from diffusers import AutoencoderKLQwenImage
    from ..utils.vae_codec import encode_image

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

        # Language filter: randomly drop non-zh samples if LANGUAGE field exists
        languages = batch.get("LANGUAGE", [None] * batch_len)
        skip_indices = set()
        if non_zh_drop_prob > 0.0:
            for idx, lang in enumerate(languages):
                if lang is not None and lang != "zh":
                    if random.random() < non_zh_drop_prob:
                        skip_indices.add(idx)

        # Track drop counts for summary
        dropped_lang = len(skip_indices)
        dropped_download = 0
        dropped_resolution = 0

        # Group by resolution bucket: (bucket_id, resolution) -> list of (original_idx, image)
        batches_by_bucket = {}

        # Parallel fetch with timeout per image (skip filtered indices)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(batch_len, 50)
        ) as executor:
            # Submit tasks only for non-skipped samples
            futures = {
                idx: executor.submit(_fetch_image, img_data)
                for idx, img_data in enumerate(images_raw)
                if idx not in skip_indices
            }

            for idx, future in futures.items():
                try:
                    # Enforce strict timeout on the entire fetch operation
                    # If this times out, the sample is dropped
                    image = future.result(timeout=10.0)
                except concurrent.futures.TimeoutError:
                    dropped_download += 1
                    image = None
                except Exception:
                    dropped_download += 1
                    image = None

                if image is None:
                    continue

                # 2. Find bucket and resize if original resolution is sufficient
                width, height = image.size
                try:
                    bucket_id, resolution = get_resolution_bucket(
                        width, height, resolution_buckets
                    )

                    if width < resolution[0] or height < resolution[1]:
                        dropped_resolution += 1
                        continue

                    resized_image = image.resize(resolution, Image.BICUBIC)

                    key = (bucket_id, resolution)
                    if key not in batches_by_bucket:
                        batches_by_bucket[key] = []
                    batches_by_bucket[key].append((idx, resized_image))
                except Exception:
                    continue

        # Print batch summary if any samples were dropped
        total_dropped = dropped_lang + dropped_download + dropped_resolution
        if total_dropped > 0:
            print(
                f"Batch: {total_dropped}/{batch_len} dropped "
                f"(lang={dropped_lang}, download={dropped_download}, resolution={dropped_resolution})"
            )

        if not batches_by_bucket:
            return {"latents": [], "captions": [], "resolution_bucket_id": []}

        # 3. Compute latents per bucket
        latents_map = {}  # original_idx -> latent
        bucket_ids_map = {}  # original_idx -> bucket_id

        for (bucket_id, resolution), items in batches_by_bucket.items():
            batch_indices = [item[0] for item in items]
            batch_images = [item[1] for item in items]

            try:
                with torch.no_grad():
                    # encode_image expects a list of PIL images
                    batch_latents = encode_image(batch_images, vae).detach().cpu()

                for i, original_idx in enumerate(batch_indices):
                    latents_map[original_idx] = batch_latents[i]
                    bucket_ids_map[original_idx] = bucket_id

                # Explicitly delete intermediate tensors and images
                del batch_latents
                del batch_images
                del batch_indices

            except Exception as e:
                print(f"Error encoding batch for resolution {resolution}: {e}")
                continue

        # Clear the batches_by_bucket dict to free PIL images
        del batches_by_bucket

        if not latents_map:
            return {"latents": [], "captions": [], "resolution_bucket_id": []}

        # 4. Gather results in order
        valid_latents = []
        valid_captions_list = []
        valid_bucket_ids = []

        # Sort by original index to maintain relative order (optional but good for determinism)
        sorted_indices = sorted(latents_map.keys())

        for original_idx in sorted_indices:
            valid_latents.append(latents_map[original_idx])
            valid_bucket_ids.append(bucket_ids_map[original_idx])

            # Gather items from caption fields
            current_captions = []
            for field in caption_fields:
                if field not in batch or batch[field][original_idx] is None:
                    continue

                val = batch[field][original_idx]
                processed_items = []

                if field == "human_caption_hq":
                    item = val[1]["value"]
                    if text_fn and isinstance(item, str):
                        processed_items.append(text_fn(item))
                    elif isinstance(item, str):
                        processed_items.append(item)
                elif field == "artist":
                    if text_fn:
                        val = text_fn(val)
                    processed_items.append(format_artist_name(val))
                elif isinstance(val, str):
                    if text_fn:
                        processed_items.append(text_fn(val))
                    else:
                        processed_items.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            if text_fn:
                                processed_items.append(text_fn(item))
                            else:
                                processed_items.append(item)

                current_captions.extend(processed_items)

            valid_captions_list.append(current_captions)

        # Clear the maps to free memory before returning
        del latents_map
        del bucket_ids_map

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "latents": valid_latents,
            "captions": valid_captions_list,
            "resolution_bucket_id": valid_bucket_ids,
        }

    # Determine columns to remove
    columns_to_keep = {"latents", "captions", "resolution_bucket_id"}
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]

    print("Starting precomputation...")
    processed_dataset = dataset.map(
        _process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=columns_to_remove,
        desc="Precomputing latents and captions",
    )

    # Clean up VAE from GPU memory
    print("Cleaning up VAE from GPU memory...")
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set format to torch
    # We don't specify columns so that all columns are returned.
    processed_dataset = processed_dataset.with_format("torch")

    return processed_dataset

