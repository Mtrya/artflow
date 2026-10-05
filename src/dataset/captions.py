"""
Caption processing and curriculum sampling utilities.

Functions:
- clean_caption: Remove artifacts from caption text
- format_artist_name: Format artist names for display
"""

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


# Within-row probabilities are proportional to retained_length ** beta.
# A reserve distributes fixed mass over captions shorter than the threshold.
DEFAULT_SHORT_THRESHOLD = 256
DEFAULT_SHORT_RESERVE = 0.20


@dataclass
class CaptionPolicy:
    """How one caption is chosen inside a selected row, over training."""

    beta_start: float = -1.0
    beta_end: float = 1.0
    short_reserve: float = DEFAULT_SHORT_RESERVE
    short_threshold: int = DEFAULT_SHORT_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.short_reserve <= 1.0:
            raise ValueError("short_reserve must be in [0, 1]")
        if self.short_threshold < 1:
            raise ValueError("short_threshold must be positive")

    def beta(self, progress: float) -> float:
        """Preference strength at normalised training progress ``progress``."""
        progress = float(np.clip(progress, 0.0, 1.0))
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


def average_caption_probabilities(
    lengths,
    policy: CaptionPolicy,
    weights: Optional[np.ndarray] = None,
    grid: int = 64,
    progress_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Schedule-averaged caption probabilities.

    Used by bucket planners to estimate exposure. The average is over probabilities, not
    over beta, because the mapping from beta to probabilities is nonlinear.
    ``weights`` optionally reweights progress points by expected sample-draw
    exposure; a uniform grid assumes progress and draws are proportional.
    ``progress_points`` optionally supplies explicit positions, allowing offline
    planners to average a resolution stage rather than the whole run.

    Accepts either one row's lengths (1-D) or a matrix of equal-length rows
    (2-D, one row per sample) and returns probabilities with the same shape.
    """
    values = np.asarray(lengths, dtype=np.float64)
    if values.ndim == 1:
        if values.size == 0:
            raise ValueError("average probabilities require at least one caption")
        return average_caption_probabilities(
            values[None, :], policy, weights, grid, progress_points)[0]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("lengths must be a 1-D row or a 2-D matrix")
    if values.shape[1] == 1:
        return np.ones_like(values)

    values = np.maximum(values, 1.0)
    points = (np.linspace(0.0, 1.0, grid) if progress_points is None
              else np.asarray(progress_points, dtype=np.float64))
    if points.ndim != 1 or not points.size or not np.all(np.isfinite(points)) \
            or np.any((points < 0) | (points > 1)):
        raise ValueError("progress points must be a nonempty vector in [0, 1]")
    if weights is None:
        weights = np.ones(points.size, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != points.shape or not np.all(np.isfinite(weights)) \
            or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("progress weights must match points and have positive mass")
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
