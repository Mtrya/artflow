"""
Caption processing and curriculum sampling utilities.

Functions:
- clean_caption: Remove artifacts from caption text
- format_artist_name: Format artist names for display
- sample_caption: Curriculum-based caption sampling for training
"""

import random
from typing import List

import numpy as np


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

    prefixes = [
        "The image shows a painting of ",
        "The image shows a drawing of ",
        "The image shows ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    text = text.replace('"""', "").replace("'''", "")

    return text.strip().capitalize()


def format_artist_name(text: str) -> str:
    """Format artist name by replacing dashes with spaces and title-casing."""
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

