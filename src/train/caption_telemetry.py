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

When the loss weights a sample by its caption's length, the same recording
also carries the weight each sample received.  What is reported is where the
weight went, as moments rather than banded shares: the mean weight over the
conditioned samples, and the weight-weighted mean caption length - the average
caption length the step's gradient actually worked on.  Both are compared
against an offline prediction of the same numbers.  The unconditional samples
keep weight 1.0 and are left out of both, exactly as they are left out of the
exposure percentiles' conditioned view.

Each quantity is reported for the current logging window only.  Ratios are
never averaged — every reduction happens at the level of the counts and sums
they are derived from, so summing counters across ranks weights ranks by their
sample count by construction.  A window is folded into the run totals exactly
once, where it is closed, and only after that window has been summed across
ranks; the run totals feed the single cumulative metric (distinct rows drawn)
rather than a second copy of every series.

The policy scalars are recorded per micro-batch and reported as the mean over
the samples of the window (and of the run).  They are a property of the step, so
the same value is recorded on every rank and the cross-rank sums divide out.

Repetition is counted from the row position a batch carries (dataset index and
row index).  Membership is held as one bit per row position, so the structure is
bounded by the corpus rather than by the number of draws.  Ranks draw disjoint
row subsets — each walks a rank-strided partition of its dataset's rows — so the
summed first-time-draw counts are the run's distinct rows, not an over-count of
them.  The map lives for the life of the process: after a resume the rows drawn
before the checkpoint are not in it and will be counted as first-time draws
again.

Not counted here: original versus enriched captions, and per-row
domain/language/format breakdowns.  A caption is a caption: a row's caption list
is assembled from whatever rounds produced its dataset, so the first entry is not
"the original" and the last is not "the enrichment" — several of these corpora
already carry captions that a model wrote in an earlier round.  A precomputed row
also stores its captions as a plain list of strings with no image id, and the
tool that appends enriched captions rewrites that list in place without writing a
marker next to it, so the two are indistinguishable in any case.  No row-level
language or format label is stored either.  Corpus mix shares by dataset are
already logged as ``data/<alias>_ratio``, and caption length — the property that
actually affects training cost — is counted in full above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # torch is imported where it is used, so that reading the
    import torch      # counters needs no torch (the planner and the exposure
                      # simulation run on machines that only have numpy)

# Per-token histogram ceiling: the prompt contract caps a caption at 2048
# tokens, so every length training can serve is tracked exactly.
MAX_TRACKED_LENGTH = 2048

# Counter fields that are summed across ranks (and into the run totals).  The
# two tuples together define the flat vector's layout: names, then the length
# histogram.  ``short_reserve`` is deliberately absent — it is one value shared
# by every rank, so summing it would multiply it by the world size.
_COUNT_FIELDS: Tuple[str, ...] = (
    "selected", "dropped", "retained_sum", "padded_sum",
    "micro_batches", "samples",
    "rows", "first_time_rows", "repeat_rows",
    "policy_samples",
)
_SUM_FIELDS: Tuple[str, ...] = (
    "progress_sum", "position_sum", "strength_sum", "conditioned_weight_sum",
    "conditioned_weighted_length_sum",
)


@dataclass(frozen=True)
class PolicyState:
    """The caption-policy scalars a micro-batch was drawn under.

    ``progress`` is the training progress clock, 0 at the first step and 1 at
    the scheduled last one.  ``curriculum_position`` is the position the
    within-row caption selector itself used, which the trainer advances on its
    own schedule and which can therefore lag ``progress``.  ``strength`` is
    whatever scalar places the selector between short and long captions: beta
    for the length-preference selector, the curriculum position for the legacy
    token-count curriculum.
    """

    progress: float
    curriculum_position: float
    strength: float
    short_reserve: float


@dataclass
class _Accumulator:
    selected: int = 0
    dropped: int = 0
    retained_sum: int = 0
    padded_sum: int = 0
    micro_batches: int = 0
    samples: int = 0
    rows: int = 0
    """Drawn rows whose position was recorded in this window."""
    first_time_rows: int = 0
    repeat_rows: int = 0
    policy_samples: int = 0
    progress_sum: float = 0.0
    position_sum: float = 0.0
    strength_sum: float = 0.0
    conditioned_weight_sum: float = 0.0
    """Sum of the loss weights over the samples the model was conditioned on."""
    conditioned_weighted_length_sum: float = 0.0
    """Sum of weight x retained length over those samples."""
    short_reserve: Optional[float] = None
    histogram: List[int] = field(default_factory=lambda: [0] * (MAX_TRACKED_LENGTH + 1))


class CaptionTelemetry:
    """Host-side caption and execution counters for the training loop."""

    def __init__(self, short_threshold: int = 256, log_every: int = 25):
        self.short_threshold = int(short_threshold)
        self.log_every = int(log_every)
        self._accumulator = _Accumulator()
        self._cumulative = _Accumulator()
        # dataset index -> bitset of drawn row positions.  Grown towards the
        # corpus' own row range, which is what bounds it; the number of draws
        # does not.
        self._seen_rows: Dict[int, bytearray] = {}

    def record(self, retained_lengths: Sequence[int], dropped: Sequence[bool],
               bucket_hi: int, *,
               row_positions: Optional[Sequence[Tuple[int, int]]] = None,
               policy: Optional[PolicyState] = None,
               loss_weights: Optional[Sequence[float]] = None) -> None:
        """Add one micro-batch. ``dropped`` is the dropout mask actually used.

        ``row_positions`` holds one (dataset index, row index) pair per sample,
        which is what repetition accounting needs; without it that block is
        simply not reported.  ``policy`` carries the selector state that
        produced the batch, so policy values are never recomputed here.
        ``loss_weights`` holds the per-sample loss weight the step used, one
        per retained length, as host values; it is recorded so a run can report
        where its gradient weight went instead of predicting it.
        """
        acc = self._accumulator
        if len(retained_lengths) != len(dropped):
            raise ValueError("retained lengths and dropout mask must have equal length")
        if loss_weights is not None and len(loss_weights) != len(retained_lengths):
            raise ValueError("loss weights must have one entry per retained length")
        batch_size = len(retained_lengths)
        acc.micro_batches += 1
        acc.samples += batch_size
        acc.padded_sum += int(bucket_hi) * batch_size
        for index, (length, was_dropped) in enumerate(zip(retained_lengths, dropped)):
            length = int(length)
            acc.selected += 1
            acc.retained_sum += length
            if 0 < length <= MAX_TRACKED_LENGTH:
                acc.histogram[length] += 1
            if was_dropped:
                acc.dropped += 1
            elif loss_weights is not None:
                # The weight moments cover every conditioned sample, so their
                # denominator is the whole weight the step's gradient was
                # divided by.
                weight = float(loss_weights[index])
                acc.conditioned_weight_sum += weight
                acc.conditioned_weighted_length_sum += weight * length

        if row_positions is not None:
            if len(row_positions) != batch_size:
                raise ValueError("row positions must have one entry per retained length")
            for dataset_id, row_idx in row_positions:
                acc.rows += 1
                if self._mark_row(int(dataset_id), int(row_idx)):
                    acc.first_time_rows += 1
                else:
                    acc.repeat_rows += 1

        if policy is not None:
            acc.policy_samples += batch_size
            acc.progress_sum += float(policy.progress) * batch_size
            acc.position_sum += float(policy.curriculum_position) * batch_size
            acc.strength_sum += float(policy.strength) * batch_size
            acc.short_reserve = float(policy.short_reserve)

    def window_counts(self) -> List[float]:
        """The window's counters as one flat vector.

        This is what ``reduce`` sums across ranks and what
        ``merge_window_counts`` accepts, so the two share one layout.  Every
        entry is additive: counts, sums and the histogram, never a ratio.
        """
        return _flatten(self._accumulator)

    def merge_window_counts(self, values: Sequence[float]) -> None:
        """Add one rank's counters (see ``window_counts``) to this window."""
        _absorb(self._accumulator, values)

    def snapshot(self, reset: bool = True) -> Dict[str, float]:
        """Return the window's metrics, and close the window.

        ``reset`` is what closes a window: the counters — already summed across
        ranks by ``reduce`` when the run is distributed — are folded into the
        run totals, and the next ``record`` starts a new window.  Passing
        ``reset=False`` only reads, so it can be called repeatedly without
        disturbing either side.  A window that recorded nothing reports nothing,
        so a caller cannot read stale numbers out of an idle interval.

        Metrics describe the window, not the run: a windowed reading answers
        "what is the run doing now", which is the question a live review asks,
        and swanlab's own smoothing covers the trend.  The one exception is
        ``repetition/unique_rows_cumulative`` - the count of distinct rows the
        run has drawn at least once - which is only meaningful as a run total.
        """
        acc = self._accumulator
        if reset:
            self._close_window()
        if acc.selected == 0:
            return {}

        metrics = _summary_metrics(acc)
        metrics["repetition/unique_rows_cumulative"] = float(
            self._cumulative.first_time_rows)
        return metrics

    def reduce(self, device: Optional[torch.device] = None,
               world_size: int = 1) -> None:
        """Sum the window's counters across ranks.

        Counts and sums are all that is needed; percentiles stay derived from
        the merged histogram, and rates stay derived from the merged counts, so
        no rank's mean or percentile is ever averaged.  One collective per
        logging window, not per micro-batch.
        """
        import torch

        if world_size <= 1 or not torch.distributed.is_available() \
                or not torch.distributed.is_initialized():
            return
        flat = torch.tensor(self.window_counts(), dtype=torch.float64,
                            device=device or "cpu")
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        self.merge_window_counts(flat.tolist())

    def _close_window(self) -> None:
        """Fold what this window accumulated into the run totals.

        The window is closed exactly here, and the fold happens after the
        window was summed across ranks, so the run totals are global without a
        second collective.  The window's own reserve survives as a value rather
        than a sum, because every rank reports the same one.
        """
        _absorb(self._cumulative, _flatten(self._accumulator))
        if self._accumulator.short_reserve is not None:
            self._cumulative.short_reserve = self._accumulator.short_reserve
        self._accumulator = _Accumulator()

    def _mark_row(self, dataset_id: int, row_idx: int) -> bool:
        """Record one drawn row; return whether it had not been drawn before."""
        if row_idx < 0:
            raise ValueError("row indices must be non-negative")
        byte_index, bit = divmod(row_idx, 8)
        bits = self._seen_rows.get(dataset_id)
        if bits is None or byte_index >= len(bits):
            bits = _grow_bitset(bits, byte_index + 1)
            self._seen_rows[dataset_id] = bits
        mask = 1 << bit
        if bits[byte_index] & mask:
            return False
        bits[byte_index] |= mask
        return True


def _grow_bitset(bits: Optional[bytearray], needed: int) -> bytearray:
    """Return a bitset of at least ``needed`` bytes, copying ``bits``."""
    size = max(64, needed)
    if bits is not None:
        size = max(size, len(bits) * 2)
    grown = bytearray(size)
    if bits is not None:
        grown[:len(bits)] = bits
    return grown


def _flatten(acc: _Accumulator) -> List[float]:
    values = [float(getattr(acc, name)) for name in (*_COUNT_FIELDS, *_SUM_FIELDS)]
    values.extend(float(count) for count in acc.histogram)
    return values


def _absorb(acc: _Accumulator, values: Sequence[float]) -> None:
    """Add a flat counter vector produced by ``_flatten`` into ``acc``."""
    expected = len(_COUNT_FIELDS) + len(_SUM_FIELDS) + len(acc.histogram)
    if len(values) != expected:
        raise ValueError(f"expected {expected} counters, got {len(values)}")
    cursor = 0
    for name in _COUNT_FIELDS:
        setattr(acc, name, getattr(acc, name) + int(round(float(values[cursor]))))
        cursor += 1
    for name in _SUM_FIELDS:
        setattr(acc, name, getattr(acc, name) + float(values[cursor]))
        cursor += 1
    for index in range(len(acc.histogram)):
        acc.histogram[index] += int(round(float(values[cursor])))
        cursor += 1


def _summary_metrics(acc: _Accumulator) -> Dict[str, float]:
    """Turn one window's counters into metrics.

    The emitted set is deliberately short: every key answers a question a run
    review actually asks - how long are the captions the model is conditioned
    on (mean and percentiles), how much of the conditioning is dropped, where
    the loss weighting sent the gradient (mean weight and the weight-weighted
    mean caption length), how much padding the buckets paid for, and how often
    rows repeat.  Raw counts are left out; they describe the counters, not the
    run.
    """
    if acc.selected == 0:
        return {}
    conditioned = acc.selected - acc.dropped
    metrics: Dict[str, float] = {
        "caption/selected_mean_tokens": acc.retained_sum / acc.selected,
        "caption/dropout_rate": acc.dropped / acc.selected,
    }
    if acc.conditioned_weight_sum > 0.0 and conditioned:
        # What the weights did, as opposed to what the corpus holds: reported
        # only for a window that recorded weights, so an unweighted run has no
        # numbers that read like a measurement it did not make.
        metrics["caption/weight_mean"] = acc.conditioned_weight_sum / conditioned
        metrics["caption/grad_weighted_mean_tokens"] = (
            acc.conditioned_weighted_length_sum / acc.conditioned_weight_sum)
    for quantile in (50, 90, 99):
        metrics[f"caption/selected_p{quantile}"] = _histogram_quantile(
            acc.histogram, acc.selected, quantile / 100.0)

    metrics["exec/padding_fraction"] = (
        (acc.padded_sum - acc.retained_sum) / acc.padded_sum if acc.padded_sum else 0.0)
    metrics["exec/samples_per_micro_batch"] = acc.samples / acc.micro_batches

    if acc.rows:
        metrics["repetition/repeat_rate"] = acc.repeat_rows / acc.rows

    if acc.policy_samples:
        metrics["policy/strength"] = acc.strength_sum / acc.policy_samples
        if acc.short_reserve is not None:
            metrics["policy/short_reserve"] = float(acc.short_reserve)
    return metrics


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
