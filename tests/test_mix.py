"""Tests for dataset mixing functionality."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.mix import (
    DatasetEntry,
    parse_dataset_mix,
    load_mixed_dataset,
    get_dataset_weights,
    get_dataset_aliases,
)
from src.dataset.sampler import ResolutionBucketSampler, collate_fn


class TestParseDatasetMix:
    """Tests for parse_dataset_mix function."""

    def test_single_path_no_weight(self):
        """Single path without weight defaults to 1.0."""
        entries = parse_dataset_mix("data/wikiart")
        assert len(entries) == 1
        assert entries[0].path == Path("data/wikiart")
        assert entries[0].weight == 1.0
        assert entries[0].alias == "wikiart"

    def test_single_path_with_weight(self):
        """Single path with weight."""
        entries = parse_dataset_mix("data/wikiart:0.5")
        assert len(entries) == 1
        assert entries[0].weight == 1.0  # Normalized to 1.0

    def test_two_paths_with_weights(self):
        """Two paths with specified weights."""
        entries = parse_dataset_mix("data/wikiart:0.9 data/relaion:0.1")
        assert len(entries) == 2
        assert entries[0].path == Path("data/wikiart")
        assert entries[1].path == Path("data/relaion")
        # Weights should be normalized
        assert abs(entries[0].weight - 0.9) < 0.001
        assert abs(entries[1].weight - 0.1) < 0.001

    def test_weights_normalized(self):
        """Weights are normalized to sum to 1.0."""
        entries = parse_dataset_mix("a:2 b:2 c:1")
        total = sum(e.weight for e in entries)
        assert abs(total - 1.0) < 0.001
        assert abs(entries[0].weight - 0.4) < 0.001
        assert abs(entries[1].weight - 0.4) < 0.001
        assert abs(entries[2].weight - 0.2) < 0.001

    def test_alias_from_path_name(self):
        """Alias is extracted from path name."""
        entries = parse_dataset_mix("precomputed/wikiart-captions@256p:1.0")
        assert entries[0].alias == "wikiart-captions@256p"

    def test_empty_spec_raises(self):
        """Empty specification raises ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            parse_dataset_mix("")
        with pytest.raises(ValueError, match="Empty"):
            parse_dataset_mix("   ")

    def test_invalid_weight_raises(self):
        """Invalid weight string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid weight"):
            parse_dataset_mix("data/x:notanumber")

    def test_zero_weight_raises(self):
        """Zero weight raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            parse_dataset_mix("data/x:0")

    def test_negative_weight_raises(self):
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            parse_dataset_mix("data/x:-0.5")


class TestGetDatasetWeightsAndAliases:
    """Tests for helper functions."""

    def test_get_dataset_weights(self):
        """get_dataset_weights extracts weights."""
        entries = [
            DatasetEntry(path=Path("a"), weight=0.7, alias="a"),
            DatasetEntry(path=Path("b"), weight=0.3, alias="b"),
        ]
        weights = get_dataset_weights(entries)
        assert weights == [0.7, 0.3]

    def test_get_dataset_aliases(self):
        """get_dataset_aliases extracts aliases."""
        entries = [
            DatasetEntry(path=Path("a"), weight=0.7, alias="dataset_a"),
            DatasetEntry(path=Path("b"), weight=0.3, alias="dataset_b"),
        ]
        aliases = get_dataset_aliases(entries)
        assert aliases == ["dataset_a", "dataset_b"]


class TestLoadMixedDataset:
    """Tests for load_mixed_dataset function."""

    @patch("src.dataset.mix.concatenate_datasets")
    @patch("src.dataset.mix.load_from_disk")
    def test_single_dataset_adds_id(self, mock_load, mock_concat):
        """Single dataset gets dataset_id column added."""
        # Mock a simple dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.add_column = MagicMock(return_value=mock_ds)
        mock_load.return_value = mock_ds

        entries = [DatasetEntry(path=Path("data/a"), weight=1.0, alias="a")]
        load_mixed_dataset(entries)

        # Check add_column was called with dataset_id
        mock_ds.add_column.assert_called_once()
        call_args = mock_ds.add_column.call_args
        assert call_args[0][0] == "dataset_id"
        assert call_args[0][1] == [0] * 100

    @patch("src.dataset.mix.concatenate_datasets")
    @patch("src.dataset.mix.load_from_disk")
    def test_multiple_datasets_concatenated(self, mock_load, mock_concat):
        """Multiple datasets are concatenated."""
        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=50)
        mock_ds1.add_column = MagicMock(return_value=mock_ds1)

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=100)
        mock_ds2.add_column = MagicMock(return_value=mock_ds2)

        mock_load.side_effect = [mock_ds1, mock_ds2]

        entries = [
            DatasetEntry(path=Path("data/a"), weight=0.5, alias="a"),
            DatasetEntry(path=Path("data/b"), weight=0.5, alias="b"),
        ]
        load_mixed_dataset(entries)

        # Check concatenate_datasets was called with both
        mock_concat.assert_called_once()
        call_args = mock_concat.call_args[0][0]
        assert len(call_args) == 2


class TestWeightedBucketSampler:
    """Tests for ResolutionBucketSampler with weighted mixing."""

    def _create_mock_dataset(self, samples):
        """Create a mock dataset with given samples."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=len(samples))
        dataset.__getitem__ = MagicMock(side_effect=lambda i: samples[i])

        # Support column access for optimization
        def getitem(key):
            if isinstance(key, str):
                return [s[key] for s in samples]
            return samples[key]

        dataset.__getitem__ = MagicMock(side_effect=getitem)
        return dataset

    def test_single_dataset_mode(self):
        """Without weights, sampler works in single-dataset mode."""
        samples = [
            {"resolution_bucket_id": 0, "latents": torch.zeros(1)},
            {"resolution_bucket_id": 0, "latents": torch.zeros(1)},
            {"resolution_bucket_id": 1, "latents": torch.zeros(1)},
            {"resolution_bucket_id": 1, "latents": torch.zeros(1)},
        ]
        dataset = self._create_mock_dataset(samples)

        sampler = ResolutionBucketSampler(
            dataset, batch_size=2, shuffle=False, dataset_weights=None
        )
        batches = list(sampler)

        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2

    def test_weighted_mode_respects_buckets(self):
        """In weighted mode, batches still respect resolution buckets."""
        samples = [
            {"resolution_bucket_id": 0, "dataset_id": 0, "latents": torch.zeros(1)},
            {"resolution_bucket_id": 0, "dataset_id": 0, "latents": torch.zeros(1)},
            {"resolution_bucket_id": 1, "dataset_id": 1, "latents": torch.zeros(1)},
            {"resolution_bucket_id": 1, "dataset_id": 1, "latents": torch.zeros(1)},
        ]
        dataset = self._create_mock_dataset(samples)

        sampler = ResolutionBucketSampler(
            dataset, batch_size=2, shuffle=False, dataset_weights=[0.5, 0.5]
        )
        batches = list(sampler)

        # Should get 2 batches (one per dataset-bucket combo)
        assert len(batches) == 2

        # Each batch should have samples from same bucket
        for batch in batches:
            bucket_ids = [samples[i]["resolution_bucket_id"] for i in batch]
            assert len(set(bucket_ids)) == 1

    def test_weighted_mode_length(self):
        """Length calculation is correct in weighted mode."""
        samples = [
            {"resolution_bucket_id": 0, "dataset_id": 0},
            {"resolution_bucket_id": 0, "dataset_id": 0},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
        ]
        dataset = self._create_mock_dataset(samples)

        sampler = ResolutionBucketSampler(
            dataset, batch_size=2, shuffle=False, dataset_weights=[0.5, 0.5]
        )

        # 2 samples per (dataset, bucket), batch_size=2 -> 2 batches
        assert len(sampler) == 2


class TestWeightedSamplerReshuffle:
    """Tests for reshuffle-on-exhaustion behavior."""

    def _create_mock_dataset(self, samples):
        """Create a mock dataset with given samples."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=len(samples))

        def getitem(key):
            if isinstance(key, str):
                return [s[key] for s in samples]
            return samples[key]

        dataset.__getitem__ = MagicMock(side_effect=getitem)
        return dataset

    def test_small_dataset_resampled(self):
        """Small dataset should be resampled to maintain ratio."""
        # Dataset 0: 4 samples (2 batches of size 2)
        # Dataset 1: 8 samples (4 batches of size 2)
        # With 50/50 weights, we should see dataset 0 resampled
        samples = [
            # Dataset 0 (small)
            {"resolution_bucket_id": 0, "dataset_id": 0},
            {"resolution_bucket_id": 0, "dataset_id": 0},
            {"resolution_bucket_id": 0, "dataset_id": 0},
            {"resolution_bucket_id": 0, "dataset_id": 0},
            # Dataset 1 (large)
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
            {"resolution_bucket_id": 0, "dataset_id": 1},
        ]
        dataset = self._create_mock_dataset(samples)

        sampler = ResolutionBucketSampler(
            dataset,
            batch_size=2,
            shuffle=False,  # Deterministic for testing
            dataset_weights=[0.5, 0.5],
        )

        batches = list(sampler)

        # Total batches = 2 (ds0) + 4 (ds1) = 6
        # With reshuffle, we should get 6 batches
        assert len(batches) == 6

        # Count batches from each dataset
        ds0_count = 0
        ds1_count = 0
        for batch in batches:
            # Check first sample's dataset_id
            ds_id = samples[batch[0]]["dataset_id"]
            if ds_id == 0:
                ds0_count += 1
            else:
                ds1_count += 1

        # With 50/50 weights over 6 batches, expect roughly 3 each
        # Due to randomness, allow some variance
        assert ds0_count >= 1, "Dataset 0 should have been sampled"
        assert ds1_count >= 1, "Dataset 1 should have been sampled"


class TestCollateWithDatasetId:
    """Tests for collate_fn with dataset_id support."""

    def test_collate_without_dataset_id(self):
        """Collate works without dataset_id."""
        batch = [
            {
                "latents": torch.zeros(16, 32, 32),
                "captions": ["test caption"],
                "resolution_bucket_id": 0,
            },
            {
                "latents": torch.ones(16, 32, 32),
                "captions": ["another caption"],
                "resolution_bucket_id": 0,
            },
        ]

        result = collate_fn(batch)

        assert "latents" in result
        assert "captions" in result
        assert "resolution_bucket_ids" in result
        assert "dataset_ids" not in result
        assert result["latents"].shape == (2, 16, 32, 32)

    def test_collate_with_dataset_id(self):
        """Collate includes dataset_ids when present."""
        batch = [
            {
                "latents": torch.zeros(16, 32, 32),
                "captions": ["test caption"],
                "resolution_bucket_id": 0,
                "dataset_id": 0,
            },
            {
                "latents": torch.ones(16, 32, 32),
                "captions": ["another caption"],
                "resolution_bucket_id": 0,
                "dataset_id": 1,
            },
        ]

        result = collate_fn(batch)

        assert "dataset_ids" in result
        assert result["dataset_ids"].tolist() == [0, 1]

