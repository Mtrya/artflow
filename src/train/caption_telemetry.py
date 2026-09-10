"""Counters describing which captions the model is actually conditioned on.

The trainer logs ``txt_seq_len``, which is the last micro-batch's padded text
width on the logging rank.  That says nothing about the caption distribution a
run consumes.  This module keeps the missing measurements, accumulated on the
host from values the batch already carries, so nothing is re-tokenized and no
device synchronization is added per micro-batch.

Three distinct things are counted, because they differ:

* **selected** — the caption chosen inside the row, before classifier-free
  guidance dropout.  This is the caption-selection policy's output.
* **conditioned** — what the model actually saw, i.e. after dropout.  A dropped
  caption contributes to the dropout rate but not to exposure.
* **padded** — the text width the compute actually paid for, which the length
  bucket sets and which is always at least the retained length.

Percentiles come from a merged per-token histogram rather than from averaged
per-rank percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Exposure bands.  The first band is everything below the short-caption
# threshold used by the selection policy's reserve.
BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("lt256", 1, 255),
    ("256_511", 256, 511),
    ("512_895", 512, 895),
    ("896_1280", 896, 1280),
)
MAX_TRACKED_LENGTH = 1280


@dataclass
class _Accumulator:
    selected: int = 0
    dropped: int = 0
    retained_sum: int = 0
    padded_sum: int = 0
    micro_batches: int = 0
    samples: int = 0
    histogram: List[int] = field(default_factory=lambda: [0] * (MAX_TRACKED_LENGTH + 1))
    band_counts: Dict[str, int] = field(default_factory=lambda: {name: 0 for name, _, _ in BANDS})
    conditioned_band_counts: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name, _, _ in BANDS})
    per_bucket: Dict[Tuple[int, int], Tuple[int, int, int]] = field(default_factory=dict)
    """(resolution_id, bucket_idx) -> (micro_batches, samples, padded_token_sum)."""


class CaptionTelemetry:
    """Host-side caption and execution counters for the training loop."""

    def __init__(self, short_threshold: int = 256, log_every: int = 25):
        self.short_threshold = int(short_threshold)
        self.log_every = int(log_every)
        self._accumulator = _Accumulator()

    def record(self, retained_lengths: Sequence[int], dropped: Sequence[bool],
               bucket_hi: int, resolution_id: int, bucket_idx: int) -> None:
        """Add one micro-batch. ``dropped`` is the dropout mask actually used."""
        acc = self._accumulator
        if len(retained_lengths) != len(dropped):
            raise ValueError("retained lengths and dropout mask must have equal length")
        batch_size = len(retained_lengths)
        acc.micro_batches += 1
        acc.samples += batch_size
        acc.padded_sum += int(bucket_hi) * batch_size
        for length, was_dropped in zip(retained_lengths, dropped):
            length = int(length)
            acc.selected += 1
            acc.retained_sum += length
            if 0 < length <= MAX_TRACKED_LENGTH:
                acc.histogram[length] += 1
            band = _band_for(length)
            if band is not None:
                acc.band_counts[band] += 1
            if was_dropped:
                acc.dropped += 1
            elif band is not None:
                acc.conditioned_band_counts[band] += 1
        key = (int(resolution_id), int(bucket_idx))
        micro_batches, samples, padded = acc.per_bucket.get(key, (0, 0, 0))
        acc.per_bucket[key] = (micro_batches + 1, samples + batch_size,
                               padded + int(bucket_hi) * batch_size)

    def snapshot(self, reset: bool = True) -> Dict[str, float]:
        """Return the window's metrics and start a new window."""
        acc = self._accumulator
        if reset:
            self._accumulator = _Accumulator()
        if acc.selected == 0:
            return {}

        conditioned = acc.selected - acc.dropped
        metrics: Dict[str, float] = {
            "caption/selected": float(acc.selected),
            "caption/selected_mean_tokens": acc.retained_sum / acc.selected,
            "caption/dropout_rate": acc.dropped / acc.selected,
            "caption/conditioned": float(conditioned),
        }
        for name, low, high in BANDS:
            metrics[f"caption/selected_share/{name}"] = acc.band_counts[name] / acc.selected
            if conditioned:
                metrics[f"caption/conditioned_share/{name}"] = (
                    acc.conditioned_band_counts[name] / conditioned)
        for quantile in (50, 90, 99):
            metrics[f"caption/selected_p{quantile}"] = _histogram_quantile(
                acc.histogram, acc.selected, quantile / 100.0)

        metrics["exec/micro_batches"] = float(acc.micro_batches)
        metrics["exec/samples"] = float(acc.samples)
        metrics["exec/padded_tokens"] = float(acc.padded_sum)
        metrics["exec/retained_tokens"] = float(acc.retained_sum)
        metrics["exec/padding_fraction"] = (
            (acc.padded_sum - acc.retained_sum) / acc.padded_sum if acc.padded_sum else 0.0)
        metrics["exec/samples_per_micro_batch"] = acc.samples / acc.micro_batches
        for (resolution_id, bucket_idx), (micro_batches, samples, padded) in acc.per_bucket.items():
            key = f"exec/bucket/res{resolution_id}_len{bucket_idx}"
            metrics[f"{key}/micro_batches"] = float(micro_batches)
            metrics[f"{key}/samples"] = float(samples)
            metrics[f"{key}/mean_padded_tokens"] = padded / max(samples, 1)
        return metrics

    def reduce(self, device: Optional[torch.device] = None,
               world_size: int = 1) -> None:
        """Sum the window's counters across ranks.

        Counts and sums are all that is needed; percentiles stay derived from
        the merged histogram.  One collective per logging window, not per
        micro-batch.
        """
        if world_size <= 1 or not torch.distributed.is_available() \
                or not torch.distributed.is_initialized():
            return
        acc = self._accumulator
        flat = torch.tensor(
            [acc.selected, acc.dropped, acc.retained_sum, acc.padded_sum,
             acc.micro_batches, acc.samples]
            + acc.histogram
            + [acc.band_counts[name] for name, _, _ in BANDS]
            + [acc.conditioned_band_counts[name] for name, _, _ in BANDS],
            dtype=torch.float64, device=device or "cpu")
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        values = flat.tolist()
        cursor = 0
        acc.selected, acc.dropped, acc.retained_sum, acc.padded_sum, \
            acc.micro_batches, acc.samples = (int(round(v)) for v in values[cursor:cursor + 6])
        cursor += 6
        acc.histogram = [int(round(v)) for v in values[cursor:cursor + len(acc.histogram)]]
        cursor += len(acc.histogram)
        for index, (name, _, _) in enumerate(BANDS):
            acc.band_counts[name] = int(round(values[cursor + index]))
        cursor += len(BANDS)
        for index, (name, _, _) in enumerate(BANDS):
            acc.conditioned_band_counts[name] = int(round(values[cursor + index]))


def _band_for(length: int) -> Optional[str]:
    for name, low, high in BANDS:
        if low <= length <= high:
            return name
    return None


def _histogram_quantile(histogram: Sequence[int], total: int, quantile: float) -> float:
    if total <= 0:
        return 0.0
    target = quantile * total
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return float(value)
    return float(len(histogram) - 1)
