"""Dataset module for ArtFlow.

This module provides utilities for data handling including:
- Caption processing and curriculum sampling
- Resolution bucketing for variable aspect ratio training
- Custom samplers for bucket-based batch sampling
- Dataset precomputation with VAE encoding
- Multi-dataset mixing for training

Imports are lazy (PEP 562): fetchers/clean/label_vlm must work in data-only
environments without the torch stack.
"""

__all__ = [
    "clean_caption",
    "format_artist_name",
    "caption_probabilities_from_token_counts",
    "sample_caption_index_from_token_counts",
    "get_resolution_bucket",
    "ResolutionBucketSampler",
    "LenBucket",
    "BucketPlan",
    "RowRef",
    "RowDescriptorDataset",
    "RowLengthQueueBatchSampler",
    "row_length_collate_fn",
    "pad_text_to_hi",
    "RowLengthMetadata",
    "METADATA_VERSION",
    "collate_fn",
    "parse_dataset_mix",
    "load_mixed_dataset",
    "get_dataset_weights",
    "DatasetEntry",
]

_LAZY = {
    "clean_caption": (".captions", "clean_caption"),
    "format_artist_name": (".captions", "format_artist_name"),
    "caption_probabilities_from_token_counts": (
        ".captions", "caption_probabilities_from_token_counts"
    ),
    "sample_caption_index_from_token_counts": (
        ".captions", "sample_caption_index_from_token_counts"
    ),
    "get_resolution_bucket": (".buckets", "get_resolution_bucket"),
    "ResolutionBucketSampler": (".sampler", "ResolutionBucketSampler"),
    "LenBucket": (".sampler", "LenBucket"),
    "BucketPlan": (".sampler", "BucketPlan"),
    "RowRef": (".sampler", "RowRef"),
    "RowDescriptorDataset": (".sampler", "RowDescriptorDataset"),
    "RowLengthQueueBatchSampler": (".sampler", "RowLengthQueueBatchSampler"),
    "row_length_collate_fn": (".sampler", "row_length_collate_fn"),
    "pad_text_to_hi": (".sampler", "pad_text_to_hi"),
    "RowLengthMetadata": (".length_metadata", "RowLengthMetadata"),
    "METADATA_VERSION": (".length_metadata", "METADATA_VERSION"),
    "collate_fn": (".sampler", "collate_fn"),
    "parse_dataset_mix": (".mix", "parse_dataset_mix"),
    "load_mixed_dataset": (".mix", "load_mixed_dataset"),
    "get_dataset_weights": (".mix", "get_dataset_weights"),
    "DatasetEntry": (".mix", "DatasetEntry"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
