"""Dataset module for ArtFlow.

This module provides utilities for data handling including:
- Caption processing and curriculum sampling
- Resolution bucketing for variable aspect ratio training
- Custom samplers for bucket-based batch sampling
- Dataset precomputation with VAE encoding
- Multi-dataset mixing for training
"""

from .captions import clean_caption, format_artist_name, sample_caption
from .buckets import get_resolution_bucket
from .sampler import ResolutionBucketSampler, collate_fn
from .mix import parse_dataset_mix, load_mixed_dataset, get_dataset_weights, DatasetEntry

__all__ = [
    "clean_caption",
    "format_artist_name",
    "sample_caption",
    "get_resolution_bucket",
    "ResolutionBucketSampler",
    "collate_fn",
    "parse_dataset_mix",
    "load_mixed_dataset",
    "get_dataset_weights",
    "DatasetEntry",
]

