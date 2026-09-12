"""
Caption processing and curriculum sampling utilities.

Functions:
- clean_caption: Remove artifacts from caption text
- format_artist_name: Format artist names for display
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


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

    ``token_counts`` is deliberately accepted separately from the caption text
    so offline row metadata and the online sampler use exactly the same
    distribution: both pass the retained lengths the trainer encodes.
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


# ---------------------------------------------------------------------------
# Length-preference selector.
#
# The legacy curriculum above takes one retained length per caption and applies
# a hand-tuned preference strength with per-caption probability clipping.  The
# replacement below is defined directly on the same retained lengths, so its
# behaviour can be read off the formula:
#
#     q_beta(c | row) = L[c]^beta / sum_j L[j]^beta
#
# beta = 0 is uniform, beta < 0 prefers shorter captions, beta > 0 prefers
# longer ones.  It is evaluated as a softmax over beta * log(L) for numerical
# stability.  A reserve keeps short captions from being squeezed out entirely
# when a row gains a much longer caption:
#
#     p = epsilon * Uniform(captions with L < threshold) + (1 - epsilon) * q_beta
#
# The reserve is over the short-caption *group*, not each caption, so adding
# more long variants to a row does not dilute it.
# ---------------------------------------------------------------------------

DEFAULT_SHORT_THRESHOLD = 256
DEFAULT_SHORT_RESERVE = 0.20


@dataclass
class CaptionPolicy:
    """How one caption is chosen inside a selected row, over training."""

    kind: str = "legacy"
    beta_start: float = -1.0
    beta_end: float = 1.0
    schedule: str = "linear"
    early_at: float = 0.5
    short_reserve: float = DEFAULT_SHORT_RESERVE
    short_threshold: int = DEFAULT_SHORT_THRESHOLD

    def __post_init__(self) -> None:
        if self.kind not in ("legacy", "beta"):
            raise ValueError(f"unknown caption policy kind {self.kind!r}")
        if self.schedule not in ("linear", "stationary", "early"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if not 0.0 <= self.short_reserve <= 1.0:
            raise ValueError("short_reserve must be in [0, 1]")
        if self.short_threshold < 1:
            raise ValueError("short_threshold must be positive")

    def beta(self, progress: float) -> float:
        """Preference strength at normalised training progress ``progress``."""
        progress = float(np.clip(progress, 0.0, 1.0))
        if self.schedule == "early":
            span = max(self.early_at, 1e-6)
            progress = min(progress / span, 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * progress


def caption_probabilities_from_lengths(
    lengths: List[int],
    beta: float,
    reserve: float = DEFAULT_SHORT_RESERVE,
    threshold: int = DEFAULT_SHORT_THRESHOLD,
) -> List[float]:
    """Within-row caption probabilities from exact retained lengths."""
    if not lengths:
        raise ValueError("caption probabilities require at least one caption")
    values = np.asarray([max(int(length), 1) for length in lengths], dtype=np.float64)
    if values.size == 1:
        return [1.0]

    log_weights = beta * np.log(values)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    probabilities = weights / weights.sum()

    if reserve > 0.0:
        short = values < threshold
        count = int(short.sum())
        if count:
            probabilities = (1.0 - reserve) * probabilities
            probabilities[short] += reserve / count
    return [float(value) for value in probabilities]


def sample_caption_index_from_lengths(
    lengths: List[int],
    beta: float,
    reserve: float = DEFAULT_SHORT_RESERVE,
    threshold: int = DEFAULT_SHORT_THRESHOLD,
    rng=None,
) -> int:
    """Sample a caption index using the length-preference distribution."""
    probabilities = caption_probabilities_from_lengths(lengths, beta, reserve, threshold)
    chooser = random if rng is None else rng
    if hasattr(chooser, "choices"):
        return chooser.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
    draw = chooser.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def average_caption_probabilities(
    lengths,
    policy: CaptionPolicy,
    weights: Optional[np.ndarray] = None,
    grid: int = 64,
) -> np.ndarray:
    """Schedule-averaged caption probabilities.

    Used by the stationary comparison arm, which holds exposure fixed and
    removes only the ordering.  The average is over the *probabilities*, not
    over beta, because the mapping from beta to probabilities is nonlinear.
    ``weights`` optionally reweights progress points by expected sample-draw
    exposure; a uniform grid assumes progress and draws are proportional.

    Accepts either one row's lengths (1-D) or a matrix of equal-length rows
    (2-D, one row per sample) and returns probabilities with the same shape.
    """
    values = np.asarray(lengths, dtype=np.float64)
    if values.ndim == 1:
        if values.size == 0:
            raise ValueError("average probabilities require at least one caption")
        return average_caption_probabilities(values[None, :], policy, weights, grid)[0]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("lengths must be a 1-D row or a 2-D matrix")
    if values.shape[1] == 1:
        return np.ones_like(values)

    values = np.maximum(values, 1.0)
    points = np.linspace(0.0, 1.0, grid)
    if weights is None:
        weights = np.ones(grid, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    total = np.zeros_like(values)
    log_values = np.log(values)
    short = values < policy.short_threshold
    short_count = short.sum(axis=1, keepdims=True)
    for point, weight in zip(points, weights):
        beta = policy.beta(float(point))
        log_weights = beta * log_values
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights_ = np.exp(log_weights)
        probabilities = weights_ / weights_.sum(axis=1, keepdims=True)
        if policy.short_reserve > 0.0:
            # A row with no short caption has nothing for the reserve to cover.
            # Scaling such a row down would leave its probabilities summing to
            # less than one, and the sampler's fallback hands the remainder to
            # whichever caption happens to sit last in the list — so the row's
            # longest caption would quietly gain the missing share.
            has_short = short_count > 0
            share = np.divide(policy.short_reserve, short_count,
                              out=np.zeros(short_count.shape, dtype=np.float64),
                              where=has_short)
            probabilities = np.where(has_short,
                                     (1.0 - policy.short_reserve) * probabilities + short * share,
                                     probabilities)
        total += weight * probabilities
    return total

