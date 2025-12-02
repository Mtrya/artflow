"""
Resolution bucketing utilities for variable aspect ratio training.

Functions:
- get_resolution_bucket: Find closest resolution bucket for given dimensions
"""

from typing import Dict, Tuple


def get_resolution_bucket(
    width: int, height: int, resolution_buckets: Dict[int, Tuple[int, int]]
) -> Tuple[int, Tuple[int, int]]:
    """
    Return the closest resolution bucket to the original aspect ratio.

    Args:
        width: Original image width
        height: Original image height
        resolution_buckets: Dictionary mapping bucket IDs to (width, height)

    Returns:
        Tuple of (bucket_id, (bucket_width, bucket_height))

    Raises:
        ValueError: If resolution_buckets is empty or contains invalid entries
    """
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

