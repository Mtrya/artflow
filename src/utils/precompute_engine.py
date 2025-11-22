"""Utilities for caching dataset encodings ahead of each training epoch.

This module centralizes the image-and-text pre-processing logic required by the training pipeline.
The :class:`PrecomputeEngine`:
- resizes images into resolution bins,
- samples captions according to a curriculum schedule,
- materializes both VAE latents and frozen text encoder embeddings.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import random
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from PIL import Image
try:
    from encode_text import encode_text
    from vae_codec import encode_image
except ImportError:
    from .encode_text import encode_text
    from .vae_codec import encode_image

def _estimate_token_counts(texts: List[str]) -> List[int]:
    """Approximate token counts assuming ~1.3 tokens per word for English prose."""
    return [int(len(text.split()) * 1.3) for text in texts]

def _sample_caption(captions: List[str], stage: float, min_prob, max_prob) -> str:
    """
    Sample a caption using stage-controlled symmetric preference scores.

    This method creates a smooth curriculum from short to long captions by computing
    deviation from mean token count and applying stage-dependent preference:

    - stage=0.0: Strongly favor short captions (below-mean length)
    - stage=0.5: No preference (uniform distribution)
    - stage=1.0: Strongly favor long captions (above-mean length)

    Args:
        stage: Training stage in [0, 1] that interpolates between short- and long-caption preferences.
        captions: Available captions to choose from.
        min_prob: Minimum probability assigned to any caption after clipping.
        max_prob: Maximum probability assigned to any caption after clipping.

    Returns:
        A single caption sampled according to the curriculum distribution.

    Raises:
        ValueError: If no captions are provided.
    """
    if not captions:
        raise ValueError("_sample_caption requires at least one caption")

    token_counts = _estimate_token_counts(captions)
    total_tokens = sum(token_counts)

    if total_tokens == 0 or len(captions) == 1:
        probabilities = [1.0 / len(captions)] * len(captions)
    else:
        # Compute deviation from mean token count (symmetric around 0)
        mean_tokens = total_tokens / len(captions)
        deviations = [(count - mean_tokens) / mean_tokens for count in token_counts]

        # Stage controls preference direction with symmetric push/pull
        # preference_strength controls how aggressive the curriculum is
        preference_strength = 2.0
        alpha = float(np.clip(stage, 0.0, 1.0))

        # At stage=0.5, (alpha - 0.5) = 0 → all scores equal → uniform
        # At stage=0.0, negative deviations (short) get boosted
        # At stage=1.0, positive deviations (long) get boosted
        scores = [1.0 + preference_strength * (alpha - 0.5) * dev for dev in deviations]

        # Ensure positive scores
        scores = [max(score, 1e-6) for score in scores]

        # Apply min/max probability clipping
        max_prob = min(max_prob, 1.0 - min_prob * (len(captions) - 1)) if len(captions) > 1 else 1.0
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

def _sample_resolution_bucket(
    width: int,
    height: int,
    resolution_buckets: Dict[int, Tuple[int, int]],
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

class PrecomputeEngine:
    def __init__(
        self,
        caption_fields: List[str],
        text_encoder_path: str,
        pooling: bool,
        vae_path: str,
        resolution_buckets: Optional[Dict[int, Tuple[int, int]]]=None,
        do_caption_scheduling: Optional[bool]=None,
        caption_scheduling_args: Optional[Dict[str, Any]]=None,
        cfg_drop_prob: Optional[float]=None,
    ) -> None:
        self.caption_fields = list(caption_fields)

        if self.caption_fields and not text_encoder_path:
            raise ValueError("text_encoder_path is required when caption_fields is not empty")
        self.text_encoder_path = text_encoder_path
        self.pooling = pooling
        self.vae_path = vae_path

        self.resolution_buckets = resolution_buckets or {
            1: (384, 384),
            2: (512, 288),
            3: (288, 512),
            4: (448, 336),
            5: (336, 448),
            6: (480, 320),
            7: (320, 480),
        }

        self.do_caption_scheduling = do_caption_scheduling or True
        self.caption_scheduling_args = caption_scheduling_args or {
            "min_prob": 0.15,
            "max_prob": 0.80
        }

        self.cfg_drop_prob = cfg_drop_prob or 0.1

    def _pass1_preprocessing(
        self,
        dataset: Dataset,
        stage: float,
        batch_size: int
    ) -> Dataset:
        """Pass 1: Assign resolution buckets and sample captions."""

        min_prob = float(self.caption_scheduling_args["min_prob"])
        max_prob = float(self.caption_scheduling_args["max_prob"])

        def _process(batch: Dict) -> Dict:
            images = batch["image"]
            batch_size = len(images)

            processed_images: List[Image.Image] = []
            bucket_ids: List[int] = []
            captions: List[str] = []

            for idx in range(batch_size):
                image = images[idx]
                width, height = image.size

                bucket_id, resolution = _sample_resolution_bucket(
                    width, height, self.resolution_buckets
                )

                resized_image = image.resize(resolution, Image.BICUBIC)

                if self.caption_fields:
                    caption_candidates = [
                        batch[field][idx] for field in self.caption_fields
                    ]

                    if self.do_caption_scheduling:
                        sampled_caption = _sample_caption(caption_candidates, stage, min_prob, max_prob)
                    else:
                        sampled_caption = _sample_caption(caption_candidates, 0.5, 0.0, 1.0)

                    if random.random() < self.cfg_drop_prob:
                        sampled_caption = ""
                else:
                    sampled_caption = ""

                processed_images.append(resized_image)
                bucket_ids.append(bucket_id)
                captions.append(sampled_caption)

            return {
                "image": processed_images,
                "caption": captions,
                "resolution_bucket_id": bucket_ids
            }
        
        columns_to_keep = {"image", "caption", "resolution_bucket_id"}
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        return dataset.map(
            _process,
            batched=True,
            batch_size=batch_size,
            remove_columns=columns_to_remove,
            keep_in_memory=False,
            num_proc=os.cpu_count(),
            desc="Pass 1: Preprocessing"
        )

    def _pass2_vae_encoding(
        self,
        dataset: Dataset,
        batch_size: int
    ) -> Dataset:
        """Pass 2: Encode images bucket-by-bucket for efficient encoding."""
        bucket_ids = set(dataset['resolution_bucket_id'])
        encoded_datasets = []

        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_pretrained(
            self.vae_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )

        for bucket_id in bucket_ids:
            bucket_ds = dataset.filter(
                lambda batch: [bid==bucket_id for bid in batch["resolution_bucket_id"]],
                batched=True,
                batch_size=2560,
                keep_in_memory=False,
                num_proc=os.cpu_count(),
                desc=f"Filtering bucket {bucket_id}"
            )

            def _encode(batch):
                images = batch["image"]
                latents_batch = encode_image(images, vae).detach().cpu()

                return {
                    'latents': [latents_batch[i] for i in range(len(images))],
                    'caption': batch['caption'],
                    'resolution_bucket_id': batch['resolution_bucket_id'],
                }
            
            columns_to_keep = {"caption", "resolution_bucket_id", "latents"}
            columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
            encoded_bucket = bucket_ds.map(
                _encode,
                batched=True,
                batch_size=batch_size,
                keep_in_memory=False,
                remove_columns=columns_to_remove,
                desc=f"Encoding bucket {bucket_id}"
            )

            encoded_datasets.append(encoded_bucket)

        return concatenate_datasets(encoded_datasets)
    
    def _pass3_text_encoding(self, dataset: Dataset, batch_size: int):
        """Pass 3: Encode text"""

        # Skip text encoding if no caption fields
        if not self.caption_fields:
            return dataset
        
        # Text encoding logic when caption_fields exists
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            self.text_encoder_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        processor=AutoProcessor.from_pretrained(self.text_encoder_path)

        def _encode(batch):
            captions = batch["caption"]

            embeddings, masks, pooled = encode_text(
                captions, text_encoder, processor, self.pooling
            )

            if self.pooling:
                return {
                    "latents": batch["latents"],
                    "text_embedding": embeddings,
                    "attention_mask": masks,
                    "pooled_embedding": pooled
                }
            else:
                return {
                    "latents": batch["latents"],
                    "text_embedding": embeddings,
                    "attention_mask": masks
                }

        return dataset.map(
            _encode,
            batched=True,
            batch_size=batch_size,
            keep_in_memory=False,
            desc="Pass 3: Text encoding"
        )

    def run(
        self,
        dataset: Dataset,
        stage: float,
        preprocessing_batch_size: int = 128,
        vae_batch_size: int = 32,
        text_batch_size: int = 4,
    ) -> Dataset:
        """Run all three precomputation passes in sequence.

        Args:
            dataset: Input dataset with image and caption fields.
            stage: Training stage in [0, 1] for caption curriculum scheduling.
            preprocessing_batch_size: Batch size for pass 1 (preprocessing).
            vae_batch_size: Batch size for pass 2 (VAE encoding).
            text_batch_size: Batch size for pass 3 (text encoding).

        Returns:
            Fully precomputed dataset ready for training.
        """
        dataset = self._pass1_preprocessing(dataset, stage, preprocessing_batch_size)
        dataset = self._pass2_vae_encoding(dataset, vae_batch_size)
        dataset = self._pass3_text_encoding(dataset, text_batch_size)
        return dataset

def precompute(
    dataset: Dataset,
    stage: float,
    caption_fields: List[str],
    text_encoder_path: str,
    pooling: bool,
    vae_path: str,
    resolution_buckets: Optional[Dict[int, Tuple[int, int]]] = None,
    do_caption_scheduling: Optional[bool] = None,
    caption_scheduling_args: Optional[Dict[str, Any]] = None,
    cfg_drop_prob: Optional[float] = None,
    preprocessing_batch_size: int = 128,
    vae_batch_size: int = 32,
    text_batch_size: int = 4,
) -> Dataset:
    """Convenience function for stateless precomputation.

    Creates a PrecomputeEngine instance and runs all three passes.

    Args:
        dataset: Input dataset with image and caption fields.
        stage: Training stage in [0, 1] for caption curriculum scheduling.
        caption_fields: List of caption field names to sample from.
        text_encoder_path: Path to Qwen3-VL text encoder.
        pooling: Whether to return pooled text embeddings.
        vae_path: Path to Qwen-Image VAE.
        resolution_buckets: Optional resolution bucket definitions.
        do_caption_scheduling: Whether to use caption curriculum scheduling.
        caption_scheduling_args: Arguments for caption scheduling.
        cfg_drop_prob: Probability of dropping captions for classifier-free guidance.
        preprocessing_batch_size: Batch size for pass 1.
        vae_batch_size: Batch size for pass 2.
        text_batch_size: Batch size for pass 3.

    Returns:
        Fully precomputed dataset ready for training.
    """
    engine = PrecomputeEngine(
        caption_fields=caption_fields,
        text_encoder_path=text_encoder_path,
        pooling=pooling,
        vae_path=vae_path,
        resolution_buckets=resolution_buckets,
        do_caption_scheduling=do_caption_scheduling,
        caption_scheduling_args=caption_scheduling_args,
        cfg_drop_prob=cfg_drop_prob,
    )
    return engine.run(
        dataset=dataset,
        stage=stage,
        preprocessing_batch_size=preprocessing_batch_size,
        vae_batch_size=vae_batch_size,
        text_batch_size=text_batch_size,
    )


def _test_sample_caption(
    output_path: str="./test_sample_caption.png",
    captions=None, 
    stages_count=101, 
    sample_count=1000
):
    """Test _sample_caption function by sampling multiple times and plotting probabilities."""
    import matplotlib.pyplot as plt
    
    # Test data: 4 captions with different lengths
    if captions is None:
        test_captions = [
            "A",  # Short caption
            "A beautiful landscape",  # Medium caption
            "A beautiful landscape with mountains and rivers under blue sky",  # Long caption
            "Art"  # Very short caption
        ]
    else:
        test_captions = captions
    
    print("Testing _sample_caption function...")
    print(f"Test captions: {test_captions}")
    
    # Sample points from 0.0 to 1.0
    stages = [i/(stages_count-1) for i in range(stages_count)] if stages_count > 1 else [0.0]
    
    # Initialize frequency tracking
    caption_frequencies = {caption: [] for caption in test_captions}
    
    min_prob, max_prob = 0.15, 0.80
    
    # For each stage, sample multiple times and record frequencies
    for stage in stages:
        # Count occurrences of each caption
        counts = {caption: 0 for caption in test_captions}
        
        # Sample multiple times
        for _ in range(sample_count):
            sampled_caption = _sample_caption(test_captions, stage, min_prob, max_prob)
            counts[sampled_caption] += 1
        
        # Record frequencies
        for caption in test_captions:
            frequency = counts[caption] / sample_count
            caption_frequencies[caption].append(frequency)
            print(f"Stage {stage:.2f}: '{caption}' frequency = {frequency:.2f}")
    
    # Plot the results if matplotlib is available
    plt.figure(figsize=(10, 6))
    for caption in test_captions:
        plt.plot(stages, caption_frequencies[caption], marker='o', label=f"'{caption[:20]}'")
    
    plt.xlabel('Training Stage')
    plt.ylabel('Sampling Frequency')
    plt.title('_sample_caption Sampling Frequency vs Training Stage')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved plot to {output_path}")
    print("Test completed.\n")

def _test_sample_resolution_bucket(
    resolution_buckets=None,
    test_images=None,
    sample_count=500
):
    """Test _sample_resolution_bucket function by sampling multiple times and printing frequencies."""
    # Test data: resolution buckets
    if resolution_buckets is None:
        resolution_buckets = {
            1: (336, 336),  # 1:1 aspect ratio
            2: (448, 252),  # 16:9 landscape
            3: (252, 448),  # 9:16 portrait
            4: (388, 291),  # 4:3 landscape
            5: (291, 388),  # 3:4 portrait
        }
    
    print("Testing _sample_resolution_bucket function...")
    print(f"Resolution buckets: {resolution_buckets}")
    
    # Test with different image sizes
    if test_images is None:
        test_images = [
            (300, 300),  # Square image
            (600, 400),  # Landscape image
            (400, 600),  # Portrait image
        ]
    
    for width, height in test_images:
        print(f"\nTesting with image size: {width}x{height}")
        
        # Count occurrences of each bucket
        counts = {bucket_id: 0 for bucket_id in resolution_buckets.keys()}
        
        # Sample multiple times
        for _ in range(sample_count):
            bucket_id, (bucket_width, bucket_height) = _sample_resolution_bucket(width, height, resolution_buckets)
            # Find the bucket_id for this resolution
            if bucket_id is not None:
                counts[bucket_id] += 1
        
        # Print frequencies
        print("Bucket frequencies:")
        for bucket_id, (w, h) in resolution_buckets.items():
            frequency = counts[bucket_id] / sample_count
            aspect_ratio = w / h
            print(f"  Bucket {bucket_id} ({w}x{h}, {aspect_ratio:.2f}): {frequency:.2f}")

def _test_precompute():
    """Test the full precomputation pipeline using PrecomputeEngine.run()"""
    from datasets import load_dataset

    print("Loading dataset...")
    dataset = load_dataset("kaupane/wikiart-captions-monet")["train"]
    print(f"Dataset size: {len(dataset)}\n")

    pooling = True
    engine = PrecomputeEngine(
        caption_fields=["mistral-caption"],
        text_encoder_path="Qwen/Qwen3-VL-2B-Instruct",
        pooling=pooling,
        vae_path="REPA-E/e2e-qwenimage-vae"
    )

    print("Running full precomputation pipeline...")
    dataset = engine.run(
        dataset=dataset,
        stage=0.5,
        preprocessing_batch_size=128,
        vae_batch_size=24,
        text_batch_size=32
    )

    print("\nPrecomputation complete!")
    print(f"Final dataset: {dataset}\n")

    # Set format for torch tensors
    if pooling:
        dataset = dataset.with_format("torch", ["latents", "text_embedding", "attention_mask", "pooled_embedding"])
    else:
        dataset = dataset.with_format("torch", ["latents", "text_embedding", "attention_mask"])

    # Sample and inspect results
    print("Sampling results:")
    for idx in range(min(10, len(dataset))):
        sample_idx = idx * (len(dataset) // 10) if len(dataset) >= 10 else idx
        print(f"\nSample {sample_idx}:")
        print(f"  Latent shape: {dataset[sample_idx]['latents'].shape}")
        print(f"  Text embedding shape: {dataset[sample_idx]['text_embedding'].shape}")
        print(f"  Attention mask shape: {dataset[sample_idx]['attention_mask'].shape}")
        print(f"  Attention mask: {dataset[sample_idx]['attention_mask']}")
        if pooling:
            print(f"  Pooled embedding shape: {dataset[sample_idx]['pooled_embedding'].shape}")

def _test_precompute_stateless():
    """Test the stateless precompute() convenience function"""
    from datasets import load_dataset

    print("Loading dataset...")
    dataset = load_dataset("kaupane/wikiart-captions-monet")["train"].select(range(256))
    print(f"Dataset size: {len(dataset)}\n")

    print("Running stateless precomputation...")
    dataset = precompute(
        dataset=dataset,
        stage=0.5,
        caption_fields=["mistral-caption"],
        text_encoder_path="Qwen/Qwen3-VL-2B-Instruct",
        pooling=True,
        vae_path="REPA-E/e2e-qwenimage-vae",
        preprocessing_batch_size=128,
        vae_batch_size=20,
        text_batch_size=32
    )

    print("\nPrecomputation complete!")
    print(f"Final dataset: {dataset}\n")

    # Set format and inspect
    dataset = dataset.with_format("torch", ["latents", "text_embedding", "attention_mask", "pooled_embedding"])

    print("Sample result:")
    sample = dataset[0]
    print(f"  Latent shape: {sample['latents'].shape}")
    print(f"  Text embedding shape: {sample['text_embedding'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Pooled embedding shape: {sample['pooled_embedding'].shape}")
    
if __name__ == "__main__":
    """Tests"""

    # Test full precomputation pipeline
    _test_precompute()

    # Test stateless convenience function
    #_test_precompute_stateless()

    # Test _sample_caption function
    _test_sample_caption(
        output_path="./test_sample_caption.png",
        captions=[
            "realism landscape by Vincent Van Gogh",
            "A desolate winter landscape, etched in delicate, intersecting lines, reveals bare trees standing starkly along a frozen waterway, their silhouettes mirrored faintly in the still surface, while two dark birds cut across the hazy, sepia-toned sky, evoking a quiet, melancholic solitude.",
            "Bare trees stand in the left and right foreground, framing a misty, shadowed expanse of water and distant land under a textured, hazy sky with two small birds flying above the left trees.",
            "detailed pen and ink landscape drawing, bare winter trees with intricate branches, dense forest on the left, solitary gnarled tree on the right, calm river or marsh reflecting the sky, distant silhouettes of buildings or farms, two birds flying in the upper left sky, textured paper surface with crosshatching and stippling, warm sepia tones with dark browns and light ochre, atmospheric perspective creating depth, soft diffused light suggesting late afternoon or overcast day, melancholic and serene mood, naturalistic composition with horizon line slightly above center, vertical framing emphasizing tall trees, influenced by Japanese woodblock prints and 19th-century landscape etchings, fine-line detail throughout, minimalist and contemplative aesthetic"
        ],
        stages_count=65,
        sample_count=1000
    )

    # Test _sample_resolution_bucket function
    _test_sample_resolution_bucket(
        resolution_buckets={
            1: (384, 384),
            2: (512, 288),
            3: (288, 512),
            4: (448, 336),
            5: (336, 448),
            6: (480, 320),
            7: (320, 480),
        },
        sample_count=1000
    )