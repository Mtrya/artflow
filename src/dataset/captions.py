"""
Caption processing and curriculum sampling utilities.

Functions:
- clean_caption: Remove artifacts from caption text
- format_artist_name: Format artist names for display
- sample_caption: Curriculum-based caption sampling for training
"""

import re
import random
from typing import List

import numpy as np


def _estimate_token_counts(texts: List[str]) -> List[int]:
    """
    Approximate token counts based on first character for language determination.
    - English and similar: ~1.3 tokens per word
    - Chinese and similar: ~0.6 token per character
    This method should be extremely fast but has low accuracy for language detection or token approximation.
    """
    results = []
    for text in texts:
        if not text:
            results.append(0)
            continue

        first_char = text[0]
        if re.match(r'[a-zA-Z]', first_char):
            results.append(int(len(text.split()) * 1.3))
        else:
            results.append(int(len(text) * 0.6))
    return results


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
        "In the image, ",
        "In the picture, ",
        "The image depicts ",
        "The image features ",
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


def caption_probabilities_from_token_counts(
    token_counts: List[int],
    stage: float,
    min_prob: float = 0.15,
    max_prob: float = 0.80,
) -> List[float]:
    """Return the existing short-to-long curriculum probabilities.

    ``token_counts`` is deliberately accepted separately from the caption text so
    offline row metadata and the online sampler can use exactly the same
    distribution as :func:`sample_caption`.
    """
    if len(token_counts) == 0:
        raise ValueError("caption probabilities require at least one caption")

    token_counts = [int(count) for count in token_counts]
    total_tokens = sum(token_counts)
    if total_tokens == 0 or len(token_counts) == 1:
        return [1.0 / len(token_counts)] * len(token_counts)

    mean_tokens = total_tokens / len(token_counts)
    deviations = [(count - mean_tokens) / mean_tokens for count in token_counts]
    preference_strength = 2.0
    alpha = float(np.clip(stage, 0.0, 1.0))
    scores = [
        max(1.0 + preference_strength * (alpha - 0.5) * deviation, 1e-6)
        for deviation in deviations
    ]

    max_prob = (
        min(max_prob, 1.0 - min_prob * (len(token_counts) - 1))
        if len(token_counts) > 1
        else 1.0
    )
    if max_prob < min_prob:
        max_prob = min_prob

    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]
    clipped_probs = [np.clip(probability, min_prob, max_prob) for probability in probabilities]
    prob_sum = sum(clipped_probs)
    return [float(probability / prob_sum) for probability in clipped_probs]


def sample_caption_index_from_token_counts(
    token_counts: List[int],
    stage: float,
    min_prob: float = 0.15,
    max_prob: float = 0.80,
    rng=None,
) -> int:
    """Sample a caption index using the legacy curriculum distribution."""
    probabilities = caption_probabilities_from_token_counts(
        token_counts, stage=stage, min_prob=min_prob, max_prob=max_prob
    )
    chooser = random if rng is None else rng
    return chooser.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


def sample_caption(
    captions: List[str], stage: float, min_prob: float = 0.15, max_prob: float = 0.80
) -> str:
    """Sample one caption using the stage-controlled curriculum distribution."""
    if not captions:
        raise ValueError("sample_caption requires at least one caption")
    token_counts = _estimate_token_counts(captions)
    sampled_idx = sample_caption_index_from_token_counts(
        token_counts, stage=stage, min_prob=min_prob, max_prob=max_prob
    )
    return captions[sampled_idx]

