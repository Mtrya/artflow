"""Utilities for stateless dataset precomputation.

This module provides functions to:
- Precompute image latents using a VAE.
- Sample and clean captions on-the-fly.
- Handle resolution bucketing and resizing.
"""

from typing import Dict, List, Optional, Tuple, Callable, Union
import random
import io
import gc
import requests
import concurrent.futures
import numpy as np
import torch
from datasets import Dataset
from PIL import Image

try:
    from vae_codec import encode_image
except ImportError:
    from .vae_codec import encode_image


def _estimate_token_counts(texts: List[str]) -> List[int]:
    """Approximate token counts assuming ~1.3 tokens per word for English prose."""
    return [int(len(text.split()) * 1.3) for text in texts]


def clean_caption(text: str) -> str:
    """
    Remove triple quotes and other wrapper artifacts from the caption.
    Remove "The image shows a painting of " opening

    Args:
        text: The input caption string.

    Returns:
        The cleaned caption string.
    """
    if not isinstance(text, str):
        return ""

    if text.startswith("The image shows a painting of "):
        text = text[len("The image shows a painting of ") :]
    if text.startswith("The image shows a drawing of "):
        text = text[len("The image shows a drawing of ") :]
    if text.startswith("The image shows "):
        text = text[len("The image shows ") :]

    text = text.replace('"""', "").replace("'''", "")

    return text.strip().capitalize()


def format_artist_name(text: str) -> str:
    return text.replace("-", " ").title()


def sample_caption(
    captions: List[str], stage: float, min_prob: float = 0.15, max_prob: float = 0.80
) -> str:
    """
    Sample a caption using stage-controlled symmetric preference scores.

    This method creates a smooth curriculum from short to long captions by computing
    deviation from mean token count and applying stage-dependent preference:

    - stage=0.0: Strongly favor short captions (below-mean length)
    - stage=0.5: No preference (uniform distribution)
    - stage=1.0: Strongly favor long captions (above-mean length)

    Args:
        captions: Available captions to choose from.
        stage: Training stage in [0, 1] that interpolates between short- and long-caption preferences.
        min_prob: Minimum probability assigned to any caption after clipping.
        max_prob: Maximum probability assigned to any caption after clipping.

    Returns:
        A single caption sampled according to the curriculum distribution.

    Raises:
        ValueError: If no captions are provided.
    """
    if not captions:
        raise ValueError("sample_caption requires at least one caption")

    token_counts = _estimate_token_counts(captions)
    total_tokens = sum(token_counts)

    if total_tokens == 0 or len(captions) == 1:
        probabilities = [1.0 / len(captions)] * len(captions)
    else:
        # Compute deviation from mean token count (symmetric around 0)
        mean_tokens = total_tokens / len(captions)
        deviations = [(count - mean_tokens) / mean_tokens for count in token_counts]
        preference_strength = 2.0
        alpha = float(np.clip(stage, 0.0, 1.0))

        scores = [1.0 + preference_strength * (alpha - 0.5) * dev for dev in deviations]

        # Ensure positive scores
        scores = [max(score, 1e-6) for score in scores]

        # Apply min/max probability clipping
        max_prob = (
            min(max_prob, 1.0 - min_prob * (len(captions) - 1))
            if len(captions) > 1
            else 1.0
        )
        if max_prob < min_prob:
            max_prob = min_prob

        # Normalize to probabilities
        total_score = sum(scores)
        probabilities = [s / total_score for s in scores]

        # Clip probabilities
        clipped_probs = [np.clip(p, min_prob, max_prob) for p in probabilities]
        prob_sum = sum(clipped_probs)
        probabilities = [p / prob_sum for p in clipped_probs]

    sampled_idx = random.choices(range(len(captions)), weights=probabilities, k=1)[0]

    return captions[sampled_idx]


def _get_resolution_bucket(
    width: int, height: int, resolution_buckets: Dict[int, Tuple[int, int]]
) -> Tuple[int, Tuple[int, int]]:
    """Return the closest resolution bucket to the original aspect ratio."""

    if not resolution_buckets:
        raise ValueError("resolution_buckets must contain at least one entry")
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")

    original_aspect = width / height
    distances = {
        bucket_id: abs((w / h) - original_aspect)
        for bucket_id, (w, h) in resolution_buckets.items()
        if h > 0
    }

    if not distances:
        raise ValueError("resolution_buckets contain invalid entries")

    closest_bucket_id = min(distances, key=distances.__getitem__)
    return closest_bucket_id, resolution_buckets[closest_bucket_id]


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
        except Exception as e:
            print(f"Failed to fetch image from {image_data[:10]}: {str(e)[:10]}")
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

    Returns:
        Processed dataset with 'latents', 'captions', and 'resolution_bucket_id'.
        'captions' will be a List[str] for each example.
    """

    # Load VAE
    from diffusers import AutoencoderKLQwenImage

    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, device_map=device
    )
    vae.eval()

    def _process_batch(batch: Dict) -> Dict:
        nonlocal vae
        images_raw = batch[image_field]
        batch_len = len(images_raw)

        # Group by resolution bucket: (bucket_id, resolution) -> list of (original_idx, image)
        batches_by_bucket = {}

        # Parallel fetch with timeout per image
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(batch_len, 50)
        ) as executor:
            # Submit all tasks
            futures = [
                executor.submit(_fetch_image, img_data) for img_data in images_raw
            ]

            for idx, future in enumerate(futures):
                try:
                    # Enforce strict timeout on the entire fetch operation
                    # If this times out, the sample is dropped
                    image = future.result(timeout=10.0)
                except concurrent.futures.TimeoutError:
                    print(f"Sample {idx}: Dropped due to download timeout")
                    image = None
                except Exception as e:
                    print(f"Sample {idx}: Dropped due to error: {str(e)[:10]}")
                    image = None

                if image is None:
                    continue

                # 2. Find bucket and resize if original resolution is sufficient
                width, height = image.size
                try:
                    bucket_id, resolution = _get_resolution_bucket(
                        width, height, resolution_buckets
                    )

                    if width < resolution[0] or height < resolution[1]:
                        print(
                            f"Skipping sample {idx}: resolution {width}x{height} below bucket {resolution[0]}x{resolution[1]}"
                        )
                        continue

                    resized_image = image.resize(resolution, Image.BICUBIC)

                    key = (bucket_id, resolution)
                    if key not in batches_by_bucket:
                        batches_by_bucket[key] = []
                    batches_by_bucket[key].append((idx, resized_image))
                except Exception:
                    continue

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
