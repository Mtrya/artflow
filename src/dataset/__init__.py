"""Dataset module for Inko.

This module provides utilities for data handling including:
- Caption processing and curriculum sampling
- Resolution bucketing for variable aspect ratio training
- Custom samplers for bucket-based batch sampling
- Dataset precomputation with VAE encoding
- Multi-dataset mixing for training

Imports are lazy (PEP 562) so that caption-only utilities work in data-only
environments without the torch stack.
"""

__all__ = [
    "clean_caption",
    "format_artist_name",
    "get_resolution_bucket",
    "LenBucket",
    "BucketPlan",
    "RowRef",
    "RowDescriptorDataset",
    "RowLengthQueueBatchSampler",
    "row_length_collate_fn",
    "pad_text_to_hi",
    "RowLengthMetadata",
    "METADATA_VERSION",
    "parse_dataset_mix",
    "get_dataset_weights",
    "DatasetEntry",
]

_LAZY = {
    "clean_caption": (".captions", "clean_caption"),
    "format_artist_name": (".captions", "format_artist_name"),
    "get_resolution_bucket": (".buckets", "get_resolution_bucket"),
    "LenBucket": (".sampler", "LenBucket"),
    "BucketPlan": (".sampler", "BucketPlan"),
    "RowRef": (".sampler", "RowRef"),
    "RowDescriptorDataset": (".sampler", "RowDescriptorDataset"),
    "RowLengthQueueBatchSampler": (".sampler", "RowLengthQueueBatchSampler"),
    "row_length_collate_fn": (".sampler", "row_length_collate_fn"),
    "pad_text_to_hi": (".sampler", "pad_text_to_hi"),
    "RowLengthMetadata": (".length_metadata", "RowLengthMetadata"),
    "METADATA_VERSION": (".length_metadata", "METADATA_VERSION"),
    "parse_dataset_mix": (".mix", "parse_dataset_mix"),
    "get_dataset_weights": (".mix", "get_dataset_weights"),
    "DatasetEntry": (".mix", "DatasetEntry"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
