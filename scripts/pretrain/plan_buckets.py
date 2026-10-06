"""Size a bucket plan from caption lengths and measured DiT ceiling points.

What a bucket plan is
---------------------
The trainer reads a plan as ``{resolution id: [{max_length, batch_size}, ...]}``
(``src.dataset.sampler.load_bucket_plan``).  A resolution id is one (resolution,
aspect) bucket of the precompute step; inside it, the plan carries K length
buckets, each a caption-length upper bound plus the micro-batch size that runs
on it.  The bounds decide how much padding a step wastes, the sizes decide how
much memory a step needs and how long its micro-batches take.

The pipeline
------------
1. **Bounds** come from the caption lengths the datasets actually hold, read
   from the per-dataset ``length_metadata.npz`` sidecars the length-metadata
   pass writes (``src.dataset.length_metadata``).  Each resolution id gets the
   boundary set that minimises total padded compute
   (``optimal_boundaries``): a caption padded to its bucket bound costs
   ``t1*L + t2*L^2`` per sample under the fitted time model, so the partition
   spends buckets where padding is expensive — the long tail gets several
   bounds instead of one bucket padded to the cap. Each row contributes its
   dataset weight / row count, split by within-row caption probability.
2. **Batch sizes are solved, never scanned.**  Two models are fitted by least
   squares on a calibration sweep of DiT forward+backward points
   (calibration JSON), grouped by the image-token count
   of the measured latent shape:

       peak_mem(B, L) = m0 + B * m1 * L
       time    (B, L) = t0 + B * (t1 * L + t2 * L ** 2)

   with ``L = image tokens + caption bound`` (the padded sequence length).  The
   memory fit is linear because flash / memory-efficient attention keeps
   activation memory linear in the padded token count; when it fits badly
   (R^2 < 0.95) a quadratic term is added, the points are refitted, and both
   fits are reported.  The time fit carries its quadratic term unconditionally:
   attention is quadratic in the sequence length.  The largest batch whose
   predicted peak stays inside the VRAM budget is then a closed-form solve,
   clamped to the configured size range.
3. **Time alignment** (optional, on by default) spends a little size in the
   fast buckets to shorten the slowest micro-batch, which is what a DDP step
   waits on: it grid-searches a common target micro-batch time T, gives every
   bucket (across all resolutions) the largest size whose predicted time stays
   under T and under its memory ceiling, and keeps the choice with the highest
   predicted throughput.  The pass is kept only if it beats the plain memory
   solution by more than ``--align-gain-threshold`` (default 3%); a smaller gain
   is not worth the extra sizes to measure and validate.

Outputs are the plan JSON in the loader's own shape and a markdown report next
to it with every bound, size, prediction, fit diagnostic and decision.

Usage:
    python scripts/pretrain/plan_buckets.py \
        --config configs/pretrain.toml --stage 256p --storage-root /external/inko \
        --calibration ceiling-256p.json \
        --image-tokens '{"1": 256, "2": 252}' \
        --buckets 10 --vram-budget-gb 42.24 \
        --out plan-256p.json --report plan-256p.report.md
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset.length_buckets import (  # noqa: E402
    dump_plan,
    histogram_from_lengths,
    optimal_boundaries,
    padding_waste,
)
from src.dataset.length_metadata import RowLengthMetadata, sidecar_path  # noqa: E402
from src.dataset.captions import CaptionPolicy, average_caption_probabilities  # noqa: E402
from src.dataset.mix import DatasetEntry, parse_dataset_mix  # noqa: E402
from src.utils.prompt_contract import MAX_SEQUENCE_LENGTH  # noqa: E402

DEFAULT_BUCKETS = 10
# 0.88 of a 64 GB card: the fraction of the card the plan may use for peak
# allocated memory, leaving room for fragmentation and reserved overhead.
DEFAULT_VRAM_BUDGET_GB = 0.88 * 64.0
DEFAULT_MIN_BATCH = 2
DEFAULT_MAX_BATCH = 128
DEFAULT_ALIGN_GAIN = 0.03
# A linear memory fit below this R^2 is refitted with a quadratic term.
MEMORY_LINEAR_R2 = 0.95
# A shape group needs at least this many points to be fitted on its own; a
# thinner group falls back to the pooled fit.
MIN_GROUP_POINTS = 3

# ---------------------------------------------------------------------------
# Calibration points.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    """One measured (latent, text length, micro-batch) DiT forward+backward."""

    latent_hw: Tuple[int, int]
    img_tokens: int
    txt_len: int
    micro_batch: int
    peak_mem_gb: float
    ms_per_step: float

    @property
    def total_tokens(self) -> int:
        """The padded sequence length the point measures: image + text tokens."""
        return int(self.img_tokens) + int(self.txt_len)


def calibration_points(payload: Any, *, patch_size: int = 2) -> List[CalibrationPoint]:
    """Read a flat list of explicit DiT forward/backward measurements.

    peak_mem_gb is peak allocated memory in GiB (bytes / 2**30).
    ms_per_step measures the whole micro-batch, including forward and backward.
    OOM rows contain the three shape fields and error="oom", without measurements.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError("calibration must be a nonempty flat list")
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("patch_size must be positive")
    points = []
    excluded = 0
    shape_fields = {"latent_hw", "txt_len", "micro_batch"}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"calibration row {index} must be an object")
        oom = row.get("error") == "oom"
        required = shape_fields | ({"error"} if oom else {"peak_mem_gb", "ms_per_step"})
        if set(row) != required:
            raise ValueError(f"calibration row {index} requires exactly {sorted(required)}")
        shape = row["latent_hw"]
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(v) is not int or v < 1 or v % patch_size for v in shape)):
            raise ValueError(f"calibration row {index}: invalid latent_hw for patch size {patch_size}")
        if any(type(row[k]) is not int or row[k] < 1 for k in ("txt_len", "micro_batch")):
            raise ValueError(f"calibration row {index}: shape counts must be positive integers")
        if oom:
            excluded += 1
            continue
        if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
               for k in ("peak_mem_gb", "ms_per_step")):
            raise ValueError(f"calibration row {index}: measurements must be finite and positive")
        points.append(CalibrationPoint(
            latent_hw=tuple(shape), img_tokens=(shape[0] // patch_size) * (shape[1] // patch_size),
            txt_len=row["txt_len"], micro_batch=row["micro_batch"],
            peak_mem_gb=float(row["peak_mem_gb"]), ms_per_step=float(row["ms_per_step"]),
        ))
    if excluded:
        print(f"Calibration: excluded {excluded} explicitly recorded OOM measurements")
    if not points:
        raise ValueError("calibration has no successful measurements")
    return points


def load_calibration(path: str, *, patch_size: int = 2) -> List[CalibrationPoint]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return calibration_points(payload, patch_size=patch_size)


# ---------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryModel:
    """Predicted peak allocated memory: ``m0 + B*m1*L [+ B*m2*L^2]`` GB."""

    m0: float
    m1: float
    m2: float
    quadratic: bool
    r_squared: float
    max_residual_gb: float
    points: int
    rank: int
    columns: int
    tokens_min: float
    tokens_max: float

    @property
    def identifiable(self) -> bool:
        """Whether the sweep varied enough to pin every coefficient down."""
        return self.rank == self.columns

    def predict_gb(self, batch: int, total_tokens: float) -> float:
        x = float(batch) * float(total_tokens)
        return self.m0 + self.m1 * x + self.m2 * x * x

    def covers(self, total_tokens: float, *, slack: float = 1.2) -> bool:
        """Whether a sequence length is inside (or just past) the fitted range."""
        return (self.tokens_min / slack) <= total_tokens <= (self.tokens_max * slack)


@dataclass(frozen=True)
class TimeModel:
    """Predicted micro-batch time: ``t0 + B * (t1*L + t2*L^2)`` ms."""

    t0: float
    t1: float
    t2: float
    r_squared: float
    max_residual_ms: float
    points: int
    rank: int
    columns: int
    tokens_min: float
    tokens_max: float

    @property
    def identifiable(self) -> bool:
        return self.rank == self.columns

    def slope_ms(self, total_tokens: float) -> float:
        """Milliseconds one more sample adds at this sequence length."""
        return self.t1 * total_tokens + self.t2 * total_tokens * total_tokens

    def predict_ms(self, batch: int, total_tokens: float) -> float:
        return self.t0 + float(batch) * self.slope_ms(total_tokens)

    def covers(self, total_tokens: float, *, slack: float = 1.2) -> bool:
        return (self.tokens_min / slack) <= total_tokens <= (self.tokens_max * slack)


def _least_squares(design: np.ndarray, values: np.ndarray
                   ) -> Tuple[np.ndarray, float, float, int]:
    """Coefficients, R^2, largest residual and rank for ``design @ x = values``.

    R^2 alone is a weak check on a sweep with a wide ``batch * L`` range: the
    largest point dominates the total variance, so a model that is a few GB off
    in the middle can still score above 0.99.  The largest residual is reported
    next to it for exactly that reason.
    """
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    ss_res = float(residual @ residual)
    ss_tot = float(((values - values.mean()) ** 2).sum())
    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 1.0 if ss_res == 0.0 else 0.0
    return (coefficients, r_squared, float(np.max(np.abs(residual))),
            int(np.linalg.matrix_rank(design)))


def _memory_design(points: Sequence[CalibrationPoint], quadratic: bool) -> np.ndarray:
    columns = [
        np.ones(len(points)),
        np.array([point.micro_batch * point.total_tokens for point in points]),
    ]
    if quadratic:
        columns.append(np.array(
            [point.micro_batch * point.total_tokens ** 2 for point in points]))
    return np.stack(columns, axis=1)


def _time_design(points: Sequence[CalibrationPoint]) -> np.ndarray:
    return np.stack([
        np.ones(len(points)),
        np.array([point.micro_batch * point.total_tokens for point in points]),
        np.array([point.micro_batch * point.total_tokens ** 2 for point in points]),
    ], axis=1)


def fit_memory_model(points: Sequence[CalibrationPoint], *,
                     r2_threshold: float = MEMORY_LINEAR_R2) -> MemoryModel:
    """Fit the linear memory model, adding a quadratic term when it fits badly.

    The linear form is the one flash / memory-efficient attention implies; the
    quadratic term is a fallback for a sweep that says otherwise (a compiled
    graph re-materializing buffers, allocator behaviour at large batches).  When
    it is added, the returned model is the quadratic one and ``quadratic`` says
    so, which is what the report compares.
    """
    if len(points) < 2:
        raise ValueError("a memory fit needs at least two calibration points")
    values = np.array([point.peak_mem_gb for point in points])
    tokens = [point.total_tokens for point in points]
    coefficients, r_squared, residual, rank = _least_squares(
        _memory_design(points, False), values)
    if r_squared >= r2_threshold or len(points) < 3:
        return MemoryModel(m0=float(coefficients[0]), m1=float(coefficients[1]), m2=0.0,
                           quadratic=False, r_squared=r_squared,
                           max_residual_gb=residual, points=len(points), rank=rank,
                           columns=2, tokens_min=float(min(tokens)),
                           tokens_max=float(max(tokens)))
    coefficients, r_squared, residual, rank = _least_squares(
        _memory_design(points, True), values)
    return MemoryModel(m0=float(coefficients[0]), m1=float(coefficients[1]),
                       m2=float(coefficients[2]), quadratic=True, r_squared=r_squared,
                       max_residual_gb=residual, points=len(points), rank=rank,
                       columns=3, tokens_min=float(min(tokens)),
                       tokens_max=float(max(tokens)))


def fit_time_model(points: Sequence[CalibrationPoint]) -> TimeModel:
    """Fit ``t = t0 + B*(t1*L + t2*L^2)``; the quadratic term is mandatory."""
    if len(points) < 2:
        raise ValueError("a time fit needs at least two calibration points")
    values = np.array([point.ms_per_step for point in points])
    tokens = [point.total_tokens for point in points]
    coefficients, r_squared, residual, rank = _least_squares(_time_design(points), values)
    return TimeModel(t0=float(coefficients[0]), t1=float(coefficients[1]),
                     t2=float(coefficients[2]), r_squared=r_squared,
                     max_residual_ms=residual, points=len(points), rank=rank, columns=3,
                     tokens_min=float(min(tokens)), tokens_max=float(max(tokens)))


@dataclass(frozen=True)
class ModelTable:
    """A pooled fit plus one fit per measured image-token count.

    A resolution is solved with the fit of its own token count when the
    calibration measured that shape often enough, and with the pooled fit
    otherwise: a shape fit cannot see behaviour the pooled fit averages away,
    and the pooled fit can see sequence lengths a single shape never reached.
    """

    pooled: Any
    by_tokens: Dict[int, Any]

    def for_tokens(self, img_tokens: int) -> Any:
        return self.by_tokens.get(int(img_tokens), self.pooled)

    def source(self, img_tokens: int) -> str:
        return "shape fit" if int(img_tokens) in self.by_tokens else "pooled fit"


def _group_by_tokens(points: Sequence[CalibrationPoint]) -> Dict[int, List[CalibrationPoint]]:
    groups: Dict[int, List[CalibrationPoint]] = {}
    for point in points:
        groups.setdefault(int(point.img_tokens), []).append(point)
    return groups


def fit_models(points: Sequence[CalibrationPoint], *,
               min_group_points: int = MIN_GROUP_POINTS
               ) -> Tuple[ModelTable, ModelTable]:
    """Fit the memory and time models, pooled and per measured shape.

    A group is only kept when its design pins its coefficients down: a memory
    fit needs the intercept and the slope (rank 2), a time fit needs the linear
    and the quadratic term separately (rank 3).  A group that fails this falls
    back to the pooled fit rather than extrapolating a degenerate fit to another
    sequence length.
    """
    memory_groups: Dict[int, MemoryModel] = {}
    time_groups: Dict[int, TimeModel] = {}
    for tokens, group in _group_by_tokens(points).items():
        if len(group) < max(min_group_points, 3):
            continue
        memory = fit_memory_model(group)
        time_model = fit_time_model(group)
        if memory.rank < 2 or time_model.rank < 3:
            continue
        memory_groups[tokens] = memory
        time_groups[tokens] = time_model
    memory_table = ModelTable(pooled=fit_memory_model(points), by_tokens=memory_groups)
    time_table = ModelTable(pooled=fit_time_model(points), by_tokens=time_groups)
    return memory_table, time_table


# ---------------------------------------------------------------------------
# Solving: the largest batch each model allows.
# ---------------------------------------------------------------------------


def solve_batch_size(model: MemoryModel, *, budget_gb: float, total_tokens: float,
                     min_batch: int, max_batch: int) -> int:
    """Largest batch whose predicted peak memory stays inside the budget.

    Closed form for both model shapes: ``B = (budget - m0) / (m1 * L)`` for the
    linear model, the positive root of the quadratic for the refitted one.  The
    result is clamped to ``[min_batch, max_batch]``; a clamp at ``min_batch``
    means even the smallest allowed batch does not fit, which the caller checks
    against the predicted peak rather than assuming.
    """
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if min_batch < 1 or max_batch < min_batch:
        raise ValueError(f"invalid batch range [{min_batch}, {max_batch}]")
    # The swing the fit predicts across the whole allowed batch range has to be
    # real: a slope at numerical zero (a sweep whose points all report the same
    # peak) would otherwise be divided into the budget and silently return
    # max_batch for every bucket.
    swing_gb = model.m1 * total_tokens * max_batch
    if model.m1 <= 0.0 or swing_gb < 0.01:
        raise ValueError(
            "the memory fit predicts no meaningful change across the allowed batch "
            f"range (m1={model.m1:.6g}, swing {swing_gb:.6g} GB at {max_batch} "
            "samples); this calibration sweep cannot size a plan")
    available = budget_gb - model.m0
    if model.quadratic and model.m2 > 0:
        a = model.m2 * total_tokens ** 2
        b = model.m1 * total_tokens
        discriminant = b * b + 4 * a * available
        if discriminant <= 0:
            return min_batch
        largest = (-b + math.sqrt(discriminant)) / (2 * a)
    else:
        largest = available / (model.m1 * total_tokens)
    return int(max(min_batch, min(max_batch, math.floor(largest))))


# ---------------------------------------------------------------------------
# Time alignment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignTarget:
    """One bucket as the alignment sees it: what it costs and what it carries."""

    key: str
    total_tokens: int
    max_batch_size: int     # the size the memory model allows
    weight: float           # share of the draws
    model: TimeModel

    def ms(self, batch: int) -> float:
        return self.model.predict_ms(batch, self.total_tokens)


@dataclass(frozen=True)
class AlignmentOutcome:
    """The sizes the alignment chose, and what they were predicted to be worth."""

    sizes: Tuple[int, ...]
    aligned: bool
    target_ms: Optional[float]
    throughput_before: float
    throughput_after: float
    note: str

    @property
    def gain(self) -> float:
        if self.throughput_before <= 0:
            return 0.0
        return self.throughput_after / self.throughput_before - 1.0


def predict_throughput(targets: Sequence[AlignTarget], sizes: Sequence[int]) -> float:
    """Predicted samples per millisecond of the step's slowest micro-batch.

    A DDP step runs a micro-batch on every rank and waits for the slowest, so
    the step's cost is the largest micro-batch time in the bucket mix and the
    samples it carries are the sample-share-weighted harmonic mean of sizes.
    Returns 0 when the mix carries no draws.
    """
    weight_sum = sum(target.weight for target in targets)
    if not targets or weight_sum <= 0:
        return 0.0
    samples = mean_emitted_batch([target.weight for target in targets], sizes)
    slowest = max(target.ms(size) for target, size in zip(targets, sizes)
                  if target.weight > 0)
    return samples / slowest if slowest > 0 else 0.0


def mean_emitted_batch(sample_shares: Sequence[float], sizes: Sequence[int]) -> float:
    """Queue-emitted mean: sample shares become batch shares proportional to p/B."""
    if len(sample_shares) != len(sizes):
        raise ValueError("sample shares and sizes must have equal length")
    if any(size <= 0 for size in sizes) or any(p < 0 for p in sample_shares):
        raise ValueError("sizes must be positive and shares nonnegative")
    mass = sum(sample_shares)
    return mass / sum(p / b for p, b in zip(sample_shares, sizes)) if mass else 0.0


def _batch_curve(target: AlignTarget, min_batch: int,
                 max_batch: int) -> Tuple[List[float], List[int]]:
    """Ascending ``(predicted ms, batch)`` pairs over the sizes a bucket may take."""
    options = list(range(min_batch, min(max_batch, target.max_batch_size) + 1))
    return [target.ms(size) for size in options], options


def align_batches(targets: Sequence[AlignTarget], *, min_batch: int, max_batch: int,
                  gain_threshold: float = DEFAULT_ALIGN_GAIN,
                  min_mean_batch: float = 0.0,
                  sample_shares: Optional[Sequence[float]] = None) -> AlignmentOutcome:
    """Trade bucket size for step alignment, and keep it only when it pays.

    A candidate target time T gives every bucket the largest size it can run
    under T, and the candidate is scored by the predicted step throughput
    (:func:`predict_throughput`).  T only needs the values where some bucket's
    size would change, so the search is exact for the model rather than sampled.
    The plain memory solution is one of the candidates, so the returned sizes are
    never predicted slower than it; a gain below ``gain_threshold`` is dropped
    anyway, because it would cost sizes to measure and validate for nothing.
    """
    if not targets:
        raise ValueError("alignment needs at least one bucket")
    for target in targets:
        if target.max_batch_size < min_batch:
            raise ValueError(f"{target.key}: no batch in [{min_batch}, {max_batch}] fits")
    unaligned = tuple(min(max_batch, target.max_batch_size) for target in targets)
    shares = list(sample_shares) if sample_shares is not None else [t.weight for t in targets]
    if not math.isfinite(min_mean_batch) or min_mean_batch < 0:
        raise ValueError("minimum mean batch must be finite and nonnegative")
    if mean_emitted_batch(shares, unaligned) < min_mean_batch:
        raise ValueError("memory-limited plan cannot meet the minimum mean batch; "
                         "change accumulation or memory constraints explicitly")
    before = predict_throughput(targets, unaligned)

    curves = [_batch_curve(target, min_batch, max_batch) for target in targets]
    candidates: List[float] = []
    for times, _ in curves:
        candidates.extend(times)
    best_sizes = unaligned
    best_target: Optional[float] = None
    best_throughput = before
    for candidate in sorted(set(candidates)):
        sizes = []
        for times, options in curves:
            index = bisect.bisect_right(times, candidate) - 1
            sizes.append(options[index] if index >= 0 else options[0])
        if mean_emitted_batch(shares, sizes) < min_mean_batch:
            continue
        throughput = predict_throughput(targets, sizes)
        if throughput > best_throughput:
            best_throughput = throughput
            best_sizes = tuple(sizes)
            best_target = candidate
    gain = best_throughput / before - 1.0 if before > 0 else 0.0
    if best_target is None or gain < gain_threshold:
        note = (f"alignment would buy {100 * gain:+.2f}% predicted throughput, below "
                f"the {100 * gain_threshold:.0f}% threshold: the memory solution "
                "is kept")
        return AlignmentOutcome(sizes=unaligned, aligned=False, target_ms=None,
                                throughput_before=before, throughput_after=before,
                                note=note)
    note = (f"alignment at {best_target:.1f} ms buys {100 * gain:+.2f}% predicted "
            "throughput over the memory solution")
    return AlignmentOutcome(sizes=best_sizes, aligned=True, target_ms=best_target,
                            throughput_before=before, throughput_after=best_throughput,
                            note=note)


# ---------------------------------------------------------------------------
# Corpus lengths and boundaries.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetLengths:
    """One dataset's caption lengths per resolution id, with draw weights."""

    alias: str
    path: str
    weight: float
    rows: int
    lengths: Dict[int, np.ndarray]
    weights: Dict[int, np.ndarray]


@dataclass(frozen=True)
class ResolutionLengths:
    """One resolution id's caption lengths and draw weights, pooled over datasets."""

    resolution_id: int
    lengths: np.ndarray
    weights: np.ndarray

    @property
    def captions(self) -> int:
        return int(self.lengths.size)


def load_sidecar_lengths(entries: Sequence[DatasetEntry], *,
                         policy: CaptionPolicy,
                         progress_start: float = 0.0, progress_end: float = 1.0,
                         progress_grid: int = 8) -> List[DatasetLengths]:
    """Read every dataset's companion length sidecar, refusing anything unusable.

    Each row contributes dataset weight / row count, divided among its captions.
    The recipe's beta policy uses the shared
    trainer probability function, averaged at midpoints of the stage interval.
    Progress points have equal weight (an approximation to sample exposure).
    """
    if not (0 <= progress_start <= 1 and 0 <= progress_end <= 1) or progress_grid < 1:
        raise ValueError("progress interval must lie in [0, 1] and grid must be positive")
    points = progress_start + (np.arange(progress_grid) + 0.5) / progress_grid \
        * (progress_end - progress_start)
    records = []
    for entry in entries:
        path = sidecar_path(str(entry.path))
        if not path.is_file():
            raise FileNotFoundError(
                f"{entry.alias}: no caption-length sidecar at {path}. Build it with "
                "src.dataset.length_metadata.ensure_sidecar(dataset_dir, "
                "tokenizer_path) or as part of the precompute run.")
        metadata = RowLengthMetadata.load(path)
        if metadata.metadata_info is None:
            raise ValueError(
                f"{entry.alias}: sidecar {path} carries no prompt contract, so its "
                "lengths may have been produced under a different template or cap; "
                "rebuild it before planning")
        if metadata.num_captions == 0:
            raise ValueError(f"{entry.alias}: sidecar {path} holds no captions")
        values: Dict[int, List[np.ndarray]] = {}
        weights: Dict[int, List[np.ndarray]] = {}
        probabilities = np.zeros(metadata.num_captions, dtype=np.float64)
        counts = np.diff(metadata.caption_offsets)
        # Group equal-caption-count rows for vectorized policy evaluation;
        # bound temporary memory independently of the corpus size.
        for count in np.unique(counts):
            if count == 0:
                raise ValueError(f"{entry.alias}: row without a caption")
            rows = np.flatnonzero(counts == count)
            for start in range(0, rows.size, 4096):
                offsets = metadata.caption_offsets[rows[start:start + 4096]]
                indices = offsets[:, None] + np.arange(count)[None, :]
                probabilities[indices] = average_caption_probabilities(
                    metadata.prompt_lengths[indices], policy, progress_points=points)
        for row in range(metadata.num_rows):
            resolution = int(metadata.resolution_ids[row])
            lengths = metadata.prompt_lengths[metadata.row_slice(row)]
            if lengths.size == 0:
                continue
            values.setdefault(resolution, []).append(lengths)
            weights.setdefault(resolution, []).append(
                probabilities[metadata.row_slice(row)] * float(entry.weight) / metadata.num_rows)
        records.append(DatasetLengths(
            alias=entry.alias, path=str(entry.path), weight=float(entry.weight),
            rows=metadata.num_rows,
            lengths={key: np.concatenate(value) for key, value in values.items()},
            weights={key: np.concatenate(value) for key, value in weights.items()},
        ))
    if not any(record.lengths for record in records):
        raise ValueError("the sidecars contain no captions")
    return records


def resolution_lengths(records: Sequence[DatasetLengths]) -> Dict[int, ResolutionLengths]:
    """Pool the datasets into one length array per resolution id."""
    pooled: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for record in records:
        for resolution, values in record.lengths.items():
            bucket = pooled.setdefault(int(resolution), ([], []))
            bucket[0].append(values)
            bucket[1].append(record.weights[resolution])
    out: Dict[int, ResolutionLengths] = {}
    for resolution, (values, weights) in pooled.items():
        out[int(resolution)] = ResolutionLengths(
            resolution_id=int(resolution),
            lengths=np.concatenate(values), weights=np.concatenate(weights))
    if not out:
        raise ValueError("the sidecars contain no resolution ids")
    return out


def bucket_boundaries(probabilities: np.ndarray, buckets: int, img_tokens: int,
                      time_model: "TimeModel") -> List[int]:
    """Compute-optimal bounds: the K-boundary partition that minimises the
    padded-compute total under the fitted time model.

    A caption padded to its bucket bound costs ``t1*L + t2*L^2`` per sample
    (``L = img_tokens + bound``), so a bucket costs its draw mass times that
    expression at its bound.  Equal-mass partitions ignore that cost and
    leave the whole long tail in one bucket padded to the cap; on long-tailed
    caption distributions that multiplies the compute the tail pays.
    ``optimal_boundaries`` is an exact DP over the ordered partition, so the
    result is the minimum-padding plan for this K, not a heuristic cut.
    """
    lengths = np.arange(1, probabilities.size + 1, dtype=np.float64)
    sequence = float(img_tokens) + lengths
    costs = time_model.t1 * sequence + time_model.t2 * sequence * sequence
    return optimal_boundaries(probabilities, buckets, costs)


def length_histogram(values: np.ndarray, weights: np.ndarray, cap: int) -> np.ndarray:
    """Normalised p(l) over lengths ``1..cap``, weighted by the draw weights."""
    return histogram_from_lengths([int(value) for value in values], max_length=cap,
                                  weights=[float(value) for value in weights])


# ---------------------------------------------------------------------------
# Assembling the plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketDraft:
    """One bucket with its bound and its memory-limited size, before alignment."""

    key: str
    resolution_id: int
    index: int
    max_length: int
    total_tokens: int
    memory_batch_size: int
    draw_share: float
    align_weight: float
    memory_model: MemoryModel
    time_model: TimeModel
    memory_source: str
    time_source: str


@dataclass(frozen=True)
class PlanDraft:
    """Every resolution's bounds and memory-limited sizes."""

    boundaries: Dict[int, List[int]]
    drafts: List[BucketDraft]
    img_tokens: Dict[int, int]
    # Padded-compute overhead per resolution: the fraction of compute the
    # plan spends on padding, (padded - actual) / actual, under the time
    # model the boundaries were optimised against.
    padding_overhead: Dict[int, float]

    def align_targets(self) -> List[AlignTarget]:
        return [AlignTarget(key=draft.key, total_tokens=draft.total_tokens,
                            max_batch_size=draft.memory_batch_size,
                            weight=draft.align_weight, model=draft.time_model)
                for draft in self.drafts]


@dataclass(frozen=True)
class BucketRow:
    """One bucket of the finished plan, with everything the report says about it."""

    resolution_id: int
    index: int
    max_length: int
    batch_size: int
    memory_batch_size: int
    predicted_mem_gb: float
    predicted_ms: float
    draw_share: float
    over_budget: bool
    extrapolates: bool


@dataclass(frozen=True)
class ResolutionPlan:
    """One resolution id: its bounds, its rows, and the models behind them."""

    resolution_id: int
    img_tokens: int
    captions: int
    boundaries: List[int]
    rows: List[BucketRow]
    memory_source: str
    time_source: str
    memory_model: MemoryModel
    time_model: TimeModel
    padding_overhead: float = 0.0


def draft_plan(pooled: Mapping[int, ResolutionLengths],
               image_tokens: Mapping[int, int], *, buckets: int, cap: int,
               memory_table: ModelTable, time_table: ModelTable, budget_gb: float,
               min_batch: int, max_batch: int, weight_mode: str) -> PlanDraft:
    """Solve the bounds and the memory-limited size of every bucket."""
    boundaries: Dict[int, List[int]] = {}
    masses: Dict[int, List[float]] = {}
    overhead: Dict[int, float] = {}
    for resolution in sorted(pooled):
        values = pooled[resolution]
        probabilities = length_histogram(values.lengths, values.weights, cap)
        img = int(image_tokens[resolution])
        time_model = time_table.for_tokens(img)
        bounds = bucket_boundaries(probabilities, buckets, img, time_model)
        lengths_axis = np.arange(1, probabilities.size + 1, dtype=np.float64)
        costs = time_model.t1 * (img + lengths_axis) \
            + time_model.t2 * (img + lengths_axis) ** 2
        actual = float((probabilities * costs).sum())
        overhead[int(resolution)] = (
            padding_waste(probabilities, bounds, costs) / actual if actual > 0 else 0.0)
        boundaries[int(resolution)] = bounds
        cumulative = np.concatenate(([0.0], np.cumsum(probabilities)))
        lower = 0
        bucket_mass = []
        for bound in bounds:
            bucket_mass.append(float(cumulative[bound] - cumulative[lower]))
            lower = bound
        masses[int(resolution)] = bucket_mass
    # Histograms above are normalized within each aspect. Restore its actual
    # probability mass before pooling; aspect buckets are not equiprobable.
    for resolution in masses:
        masses[resolution] = [p * float(pooled[resolution].weights.sum())
                              for p in masses[resolution]]
    total_mass = sum(sum(mass) for mass in masses.values())
    if total_mass <= 0:
        raise ValueError("the caption-length histogram carries no mass")

    bucket_total = sum(len(mass) for mass in masses.values())
    uniform_weight = 1.0 / bucket_total
    drafts: List[BucketDraft] = []
    for resolution in sorted(pooled):
        img_tokens = int(image_tokens[resolution])
        memory = memory_table.for_tokens(img_tokens)
        time_model = time_table.for_tokens(img_tokens)
        for index, (bound, mass) in enumerate(
                zip(boundaries[int(resolution)], masses[int(resolution)])):
            total_tokens = img_tokens + int(bound)
            drafts.append(BucketDraft(
                key=f"res{resolution}/len<={bound}",
                resolution_id=int(resolution),
                index=index,
                max_length=int(bound),
                total_tokens=total_tokens,
                memory_batch_size=solve_batch_size(
                    memory, budget_gb=budget_gb, total_tokens=total_tokens,
                    min_batch=min_batch, max_batch=max_batch),
                draw_share=mass / total_mass,
                align_weight=(mass / total_mass) if weight_mode == "count"
                else uniform_weight,
                memory_model=memory,
                time_model=time_model,
                memory_source=memory_table.source(img_tokens),
                time_source=time_table.source(img_tokens),
            ))
    return PlanDraft(boundaries=boundaries, drafts=drafts,
                     img_tokens={int(key): int(value)
                                 for key, value in image_tokens.items()},
                     padding_overhead=overhead)


def solve_sizes(draft: PlanDraft, *, align: bool, min_batch: int, max_batch: int,
                gain_threshold: float, min_mean_batch: float = 0.0
                ) -> Tuple[List[int], Optional[AlignmentOutcome]]:
    """The size of every draft bucket, aligned or not."""
    if not align:
        sizes = [bucket.memory_batch_size for bucket in draft.drafts]
        if mean_emitted_batch([b.draw_share for b in draft.drafts], sizes) < min_mean_batch:
            raise ValueError("memory-limited plan cannot meet the minimum mean batch")
        return sizes, None
    outcome = align_batches(draft.align_targets(), min_batch=min_batch,
                            max_batch=max_batch, gain_threshold=gain_threshold,
                            min_mean_batch=min_mean_batch,
                            sample_shares=[b.draw_share for b in draft.drafts])
    return list(outcome.sizes), outcome


def finalize_plan(draft: PlanDraft, sizes: Sequence[int],
                  pooled: Mapping[int, ResolutionLengths],
                  *, budget_gb: float) -> List[ResolutionPlan]:
    """Turn the drafts and their sizes into the reported per-resolution plans."""
    if len(sizes) != len(draft.drafts):
        raise ValueError(f"{len(sizes)} sizes for {len(draft.drafts)} buckets")
    by_resolution: Dict[int, List[Tuple[BucketDraft, int]]] = {}
    for bucket, size in zip(draft.drafts, sizes):
        by_resolution.setdefault(bucket.resolution_id, []).append((bucket, int(size)))
    plans: List[ResolutionPlan] = []
    for resolution in sorted(by_resolution):
        entries = by_resolution[resolution]
        rows = []
        for bucket, size in entries:
            predicted = bucket.memory_model.predict_gb(size, bucket.total_tokens)
            rows.append(BucketRow(
                resolution_id=resolution,
                index=bucket.index,
                max_length=bucket.max_length,
                batch_size=size,
                memory_batch_size=bucket.memory_batch_size,
                predicted_mem_gb=predicted,
                predicted_ms=bucket.time_model.predict_ms(size, bucket.total_tokens),
                draw_share=bucket.draw_share,
                over_budget=predicted > budget_gb,
                extrapolates=not (bucket.memory_model.covers(bucket.total_tokens)
                                  and bucket.time_model.covers(bucket.total_tokens)),
            ))
        plans.append(ResolutionPlan(
            resolution_id=resolution,
            img_tokens=draft.img_tokens[resolution],
            captions=pooled[resolution].captions,
            boundaries=list(draft.boundaries[resolution]),
            rows=rows,
            memory_source=entries[0][0].memory_source,
            time_source=entries[0][0].time_source,
            memory_model=entries[0][0].memory_model,
            time_model=entries[0][0].time_model,
            padding_overhead=draft.padding_overhead.get(resolution, 0.0),
        ))
    return plans


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


@dataclass
class PlanContext:
    """Everything the report records about how the plan was produced."""

    out_path: str
    report_path: str
    command: str
    datasets: Sequence[DatasetLengths]
    calibration_path: str
    calibration_points: Sequence[CalibrationPoint]
    bucket_count: int
    length_cap: int
    image_tokens_spec: str
    budget_gb: float
    min_batch: int
    max_batch: int
    align: bool
    align_gain_threshold: float
    weight_mode: str
    memory: ModelTable
    time: ModelTable
    alignment: Optional[AlignmentOutcome]
    caption_distribution: str = "uniform within each row (approximation)"
    mean_micro_batch: float = 0.0
    min_mean_batch: float = 0.0
    warnings: List[str] = field(default_factory=list)


def _number_text(value: float, digits: int = 5) -> str:
    """Small numbers get plain digits, tiny and huge ones scientific notation."""
    if value == 0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _memory_fit_line(name: str, model: MemoryModel) -> str:
    quadratic = f" + {_number_text(model.m2)} * x^2" if model.quadratic else ""
    return (f"`{name}`: peak = {_number_text(model.m0)} + {_number_text(model.m1)} * x"
            f"{quadratic}   (R^2 {model.r_squared:.4f}, worst point off by "
            f"{model.max_residual_gb:.2f} GB, {model.points} points, x = batch * L)")


def _time_fit_line(name: str, model: TimeModel) -> str:
    return (f"`{name}`: t = {_number_text(model.t0)} + batch * ({_number_text(model.t1)}"
            f" * L + {_number_text(model.t2)} * L^2)   (R^2 {model.r_squared:.4f}, "
            f"worst point off by {model.max_residual_ms:.1f} ms, {model.points} points)")


def render_report(context: PlanContext, plans: Sequence[ResolutionPlan]) -> str:
    lines: List[str] = []
    add = lines.append
    add(f"# Bucket plan: {context.out_path}")
    add("")
    add("Produced by `scripts/pretrain/plan_buckets.py`.  Bounds minimise padded")
    add("compute under the fitted time model; batch sizes are solved from a")
    add("measured memory model, not scanned.  Validate the plan with a")
    add("mixed-stream training run before relying on it: an isolated measurement")
    add("bounds a shape, it does not certify a plan whose many shapes share one")
    add("allocator.")
    add("")
    add("```")
    add(context.command)
    add("```")
    add("")
    add("## Inputs")
    add("")
    add("| item | value |")
    add("| --- | --- |")
    add(f"| datasets | {len(context.datasets)} |")
    add(f"| buckets per resolution | {context.bucket_count} |")
    add(f"| caption length cap | {context.length_cap} "
        f"(the prompt contract allows {MAX_SEQUENCE_LENGTH}) |")
    add(f"| VRAM budget | {context.budget_gb:.2f} GB peak allocated, as the "
        f"calibration measures it |")
    add(f"| batch range | [{context.min_batch}, {context.max_batch}] |")
    add(f"| time alignment | {'on' if context.align else 'off'}, kept above "
        f"{100 * context.align_gain_threshold:.0f}% predicted gain |")
    add(f"| alignment weights | {context.weight_mode} |")
    add(f"| minimum mean emitted micro-batch | {context.min_mean_batch:g} |")
    add(f"| image tokens | `{context.image_tokens_spec}` |")
    add(f"| calibration | {context.calibration_path} "
        f"({len(context.calibration_points)} points) |")
    add("")
    for record in context.datasets:
        resolutions = ", ".join(f"res{key}" for key in sorted(record.lengths)) or "none"
        add(f"- `{record.path}` weight {record.weight:.4f}, {record.rows} rows, "
            f"resolutions: {resolutions}")
    add("")
    add("## Calibration")
    add("")
    points = context.calibration_points
    add(f"Image-token counts {sorted({point.img_tokens for point in points})}, "
        f"text lengths {sorted({point.txt_len for point in points})}, "
        f"micro-batches {sorted({point.micro_batch for point in points})}; "
        f"sequence lengths {min(p.total_tokens for p in points)}.."
        f"{max(p.total_tokens for p in points)}.")
    add("")
    add("## Memory model")
    add("")
    add("Peak allocated memory is fitted as `m0 + B * m1 * L`, `L` being the padded")
    add("sequence length (image tokens + caption bound) — the form flash /")
    add("memory-efficient attention implies.  A fit below R^2 0.95 is refitted with a")
    add("`+ B * m2 * L^2` term.")
    add("")
    add(_memory_fit_line("pooled", context.memory.pooled))
    for tokens in sorted(context.memory.by_tokens):
        add(_memory_fit_line(f"img tokens {tokens}", context.memory.by_tokens[tokens]))
    add("")
    add("R^2 alone is a weak check on a sweep with a wide `batch * L` range: the")
    add("largest point dominates the total variance, so a model that is a few GB off")
    add("in the middle still scores above 0.99.  Read the worst-point residual next to")
    add("it, against the budget the sizes have to fit in.")
    add("")
    if not context.memory.pooled.identifiable:
        add(f"- The pooled memory fit is rank deficient "
            f"({context.memory.pooled.rank}/{context.memory.pooled.columns} columns): "
            "the sweep does not vary batch and sequence length independently, so its")
        add("  coefficients are not all identified — measure more than one micro-batch")
        add("  per shape.")
    for tokens in sorted(context.memory.by_tokens):
        model = context.memory.by_tokens[tokens]
        if model.quadratic:
            add(f"- The img tokens {tokens} fit needed the quadratic term "
                f"(R^2 {model.r_squared:.4f}); the linear model is not enough for this")
            add("  shape.")
    add("")
    add("## Time model")
    add("")
    add("Micro-batch time is fitted as `t0 + B * (t1 * L + t2 * L^2)`.  The quadratic")
    add("term is mandatory: attention time grows with the square of the padded")
    add("sequence, and a plan sized without it mispredicts its long buckets.")
    add("")
    add(_time_fit_line("pooled", context.time.pooled))
    for tokens in sorted(context.time.by_tokens):
        add(_time_fit_line(f"img tokens {tokens}", context.time.by_tokens[tokens]))
    add("")
    if not context.time.pooled.identifiable:
        add(f"- The pooled time fit is rank deficient "
            f"({context.time.pooled.rank}/{context.time.pooled.columns} columns); its")
        add("  coefficients are a minimum-norm solution and extrapolate poorly —")
        add("  measure at least three distinct sequence lengths.")
    add("")
    add("## Buckets")
    add("")
    for plan in plans:
        add(f"### resolution {plan.resolution_id} "
            f"(image tokens {plan.img_tokens}, {plan.captions} captions)")
        add("")
        add(f"Bounds minimise padded compute under the {plan.time_source} "
            f"(padding overhead {plan.padding_overhead * 100:.1f}% of useful "
            f"compute); sizes come from the {plan.memory_source} for memory "
            f"and the {plan.time_source} for time.")
        add("")
        add("| bucket | max_length | draw share | batch (memory) | batch (plan) | "
            "predicted GB | predicted ms |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for row in plan.rows:
            flags = ""
            if row.over_budget:
                flags += " **over budget**"
            if row.extrapolates:
                flags += " **(extrapolated)**"
            add(f"| {row.index} | {row.max_length} | {100 * row.draw_share:.2f}% | "
                f"{row.memory_batch_size} | {row.batch_size} | "
                f"{row.predicted_mem_gb:.2f}{flags} | {row.predicted_ms:.1f} |")
        add("")
    add("## Time alignment")
    add("")
    if context.alignment is None:
        add("Switched off: every bucket runs the largest batch its memory ceiling")
        add("allows.  That leaves the buckets with short captions finishing their")
        add("micro-batch early while the step waits for the longest one.")
    else:
        add("Predicted throughput in samples per millisecond of the step's slowest")
        add(f"micro-batch: {context.alignment.throughput_before:.4f} for the memory "
            f"solution, {context.alignment.throughput_after:.4f} as planned "
            f"({100 * context.alignment.gain:+.2f}%).")
        add("")
        add(context.alignment.note)
        add("")
        add("The step model assumes one micro-batch per rank per optimizer step, so")
        add("every step pays the slowest runnable bucket, and each bucket is weighted")
        add("by its share of the draws.  With gradient accumulation that gain shrinks")
        add("towards the slowest-rank premium: a rank's micro-batches are a sum over")
        add("draws, and one rare bucket inside sixteen of them costs far less than the")
        add("whole step.  Read the number as an upper bound on what balancing buys,")
        add("and validate candidate plans on the full training workload.")
    add("")
    add("## Limits")
    add("")
    add("- The models are fitted on DiT forward+backward points.  A training step also")
    add("  carries the frozen text encoder, the optimizer and EMA state, DDP buffers,")
    add("  and the per-shape allocations a mixed stream retains, so the budget has to")
    add("  cover whatever the calibration did not measure.  The fitted `m0` is the")
    add("  fixed part the sweep saw: several GB for a training run's weights, optimizer")
    add("  state and kernels, much less for a DiT-only sweep, which spends the")
    add("  difference out of the same budget.")
    add("- Peak memory is a property of the whole plan, not of a single shape; the")
    add("  numbers above are per-shape predictions.  A bucket marked **over budget**")
    add("  does not fit even at the smallest allowed batch, so the plan is not usable")
    add("  at that shape until the batch floor or the budget changes.")
    add(f"- Caption distribution: {context.caption_distribution}.")
    add("  Stage-progress averaging assumes equal exposure per progress point;")
    add("  changing emitted batch sizes can make actual exposure differ.")
    add(f"- Predicted queue-emitted mean micro-batch: {context.mean_micro_batch:.6f}")
    add("  samples, using 1 / sum(p_i / B_i), not sum(p_i * B_i). This is a")
    add("  long-run mean, not a per-update floor; validate the actual sampler stream.")
    add("- The alignment model costs a step by its slowest micro-batch and ignores")
    add("  queue tails, compilation and synchronization, so treat its gain as a")
    add("  direction rather than a measurement.")
    if context.warnings:
        add("")
        add("## Warnings")
        add("")
        for warning in context.warnings:
            add(f"- {warning}")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def parse_image_tokens(spec: str, resolutions: Sequence[int]) -> Dict[int, int]:
    """Image tokens per resolution id: one integer for all, or a JSON mapping."""
    text = spec.strip()
    if text.lstrip("+-").isdigit():
        value = int(text)
        if value < 1:
            raise ValueError(f"--image-tokens must be positive, got {value}")
        return {int(resolution): value for resolution in resolutions}
    path = Path(text)
    payload = path.read_text(encoding="utf-8") if path.is_file() else text
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--image-tokens must be an integer or a JSON object mapping resolution "
            "id to token count") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("--image-tokens must be an integer or a JSON object")
    tokens: Dict[int, int] = {}
    for key, value in raw.items():
        try:
            resolution = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"--image-tokens key {key!r} is not a resolution id")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"--image-tokens for resolution {resolution} must be a positive "
                f"integer, got {value!r}")
        tokens[resolution] = int(value)
    missing = sorted(set(int(item) for item in resolutions) - set(tokens))
    if missing:
        raise ValueError(f"--image-tokens is missing resolution ids {missing}")
    return tokens


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--calibration", required=True, metavar="JSON",
                        help="calibration sweep written by "
                             "a measured DiT forward/backward sweep")
    parser.add_argument("--image-tokens", required=True, metavar="N|JSON",
                        help="image tokens per resolution id: one integer for all "
                             "resolutions, or a JSON object (or a path to one) "
                             "mapping resolution id to token count")
    parser.add_argument("--out", required=True, help="bucket plan JSON to write")
    parser.add_argument("--report", default=None,
                        help="markdown report path (default <out>.report.md)")
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS,
                        help=f"length buckets per resolution (default "
                             f"{DEFAULT_BUCKETS})")
    parser.add_argument("--vram-budget-gb", type=float, default=DEFAULT_VRAM_BUDGET_GB,
                        help="peak allocated memory the plan may use (default "
                             f"{DEFAULT_VRAM_BUDGET_GB:.2f} GB, 88%% of a 64 GB card)")
    parser.add_argument("--min-batch", type=int, default=DEFAULT_MIN_BATCH,
                        help=f"smallest micro-batch a bucket may take (default "
                             f"{DEFAULT_MIN_BATCH})")
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                        help=f"largest micro-batch a bucket may take (default "
                             f"{DEFAULT_MAX_BATCH})")
    parser.add_argument("--min-mean-batch", type=float, default=0.0,
                        help="minimum sample-share-weighted harmonic mean batch; "
                             "global target / (ranks * accumulation)")
    parser.add_argument("--length-cap", type=int, default=MAX_SEQUENCE_LENGTH,
                        help="highest caption length the plan buckets up to "
                             f"(default {MAX_SEQUENCE_LENGTH}, the prompt contract)")
    parser.add_argument("--patch-size", type=int, default=2,
                        help="DiT patch size, used when a calibration point names a "
                             "latent shape instead of an image-token count")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=True,
                        help="run the time-alignment pass (default on)")
    parser.add_argument("--align-gain-threshold", type=float, default=DEFAULT_ALIGN_GAIN,
                        help="predicted throughput gain the alignment must beat to be "
                             f"kept (default {DEFAULT_ALIGN_GAIN})")
    parser.add_argument("--align-weights", choices=("count", "uniform"), default="count",
                        help="weights in the alignment model: 'count' uses each "
                             "bucket's share of the draws, 'uniform' weights every "
                             "bucket equally (default count)")
    parser.add_argument("--progress-grid", type=int, default=8)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, argv: Sequence[str] = ()) -> int:
    if args.buckets < 1:
        raise ValueError("--buckets must be at least 1")
    if not 1 <= args.length_cap <= MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"--length-cap must be in [1, {MAX_SEQUENCE_LENGTH}], got {args.length_cap}")
    if args.buckets > args.length_cap:
        raise ValueError("--buckets cannot exceed --length-cap")
    if args.min_batch < 1 or args.max_batch < args.min_batch:
        raise ValueError(
            "--min-batch and --max-batch must satisfy 1 <= min <= max, got "
            f"[{args.min_batch}, {args.max_batch}]")
    if args.vram_budget_gb <= 0:
        raise ValueError("--vram-budget-gb must be positive")
    if args.align_gain_threshold < 0:
        raise ValueError("--align-gain-threshold must not be negative")
    if not math.isfinite(args.min_mean_batch) or args.min_mean_batch < 0:
        raise ValueError("--min-mean-batch must be finite and nonnegative")

    from src.pretrain.config import load_config, flatten, stage_caption_policy

    config = load_config(args.config, storage_root=args.storage_root)
    entries = parse_dataset_mix(flatten(config, args.stage)["dataset_mix"])
    policy, (progress_start, progress_end) = stage_caption_policy(config, args.stage)
    records = load_sidecar_lengths(
        entries, policy=policy, progress_start=progress_start,
        progress_end=progress_end, progress_grid=args.progress_grid)
    pooled = resolution_lengths(records)
    image_tokens = parse_image_tokens(args.image_tokens, sorted(pooled))
    points = load_calibration(args.calibration, patch_size=args.patch_size)
    memory_table, time_table = fit_models(points)

    # Aligning buckets needs a time model that separates the linear from the
    # quadratic term; a rank-deficient fit still predicts the measured lengths,
    # but it would rank the buckets of the plan — which sit at other lengths —
    # on noise.
    warnings: List[str] = []
    align = args.align
    if align and not time_table.pooled.identifiable:
        align = False
        warnings.append(
            f"the pooled time fit is rank deficient "
            f"({time_table.pooled.rank}/{time_table.pooled.columns} columns), so the "
            "time-alignment pass is skipped; measure at least three distinct "
            "sequence lengths to identify it")

    draft = draft_plan(pooled, image_tokens, buckets=args.buckets, cap=args.length_cap,
                       memory_table=memory_table, time_table=time_table,
                       budget_gb=args.vram_budget_gb, min_batch=args.min_batch,
                       max_batch=args.max_batch, weight_mode=args.align_weights)
    sizes, alignment = solve_sizes(draft, align=align, min_batch=args.min_batch,
                                   max_batch=args.max_batch,
                                   gain_threshold=args.align_gain_threshold,
                                   min_mean_batch=args.min_mean_batch)
    plans = finalize_plan(draft, sizes, pooled, budget_gb=args.vram_budget_gb)

    if not memory_table.pooled.identifiable:
        warnings.append(
            "the pooled memory fit is rank deficient; the calibration sweep does not "
            "vary batch and sequence length independently, so the sizes it derives "
            "are not trustworthy")
    for resolution, values in sorted(pooled.items()):
        if values.captions < args.buckets:
            warnings.append(
                f"resolution {resolution} holds only {values.captions} captions but "
                f"the plan asks for {args.buckets} buckets; the empty buckets split a "
                "zero-mass region")
    for plan in plans:
        for row in plan.rows:
            if row.over_budget:
                warnings.append(
                    f"res{plan.resolution_id} len<={row.max_length}: even batch "
                    f"{row.batch_size} is predicted at {row.predicted_mem_gb:.2f} GB, "
                    f"over the {args.vram_budget_gb:.2f} GB budget")
            if row.extrapolates:
                warnings.append(
                    f"res{plan.resolution_id} len<={row.max_length}: the padded "
                    "sequence is outside the calibrated range; measure this shape "
                    "before trusting its size")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump_plan(str(args.out), {plan.resolution_id: plan.boundaries for plan in plans},
              {plan.resolution_id: [row.batch_size for row in plan.rows]
               for plan in plans})

    report_path = args.report or f"{args.out}.report.md"
    context = PlanContext(
        out_path=str(args.out), report_path=str(report_path),
        command="python scripts/pretrain/plan_buckets.py " + shlex.join(argv),
        datasets=records, calibration_path=args.calibration, calibration_points=points,
        bucket_count=args.buckets, length_cap=args.length_cap,
        image_tokens_spec=args.image_tokens, budget_gb=args.vram_budget_gb,
        min_batch=args.min_batch, max_batch=args.max_batch, align=align,
        align_gain_threshold=args.align_gain_threshold, weight_mode=args.align_weights,
        memory=memory_table, time=time_table, alignment=alignment, warnings=warnings)
    context.caption_distribution = (
        f"{policy}; progress [{progress_start}, {progress_end}], "
        f"{args.progress_grid} midpoints")
    context.mean_micro_batch = mean_emitted_batch(
        [bucket.draw_share for bucket in draft.drafts], sizes)
    context.min_mean_batch = args.min_mean_batch
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(render_report(context, plans), encoding="utf-8")

    print(f"wrote plan: {args.out}")
    print(f"wrote report: {report_path}")
    print(f"mean emitted micro-batch: {context.mean_micro_batch:.6f}")
    for name, table in (("memory", memory_table), ("time", time_table)):
        kind = "quadratic" if getattr(table.pooled, "quadratic", False) else "linear"
        print(f"{name} model: {kind} pooled fit, R^2 {table.pooled.r_squared:.4f} "
              f"({table.pooled.points} points), {len(table.by_tokens)} shape fits")
    if alignment is not None:
        print(f"alignment: {'kept' if alignment.aligned else 'dropped'} "
              f"({100 * alignment.gain:+.2f}% predicted throughput)")
    for plan in plans:
        sizes_here = [row.batch_size for row in plan.rows]
        print(f"  resolution {plan.resolution_id}: {len(plan.rows)} buckets, bounds "
              f"{plan.boundaries[0]}..{plan.boundaries[-1]}, batch "
              f"{min(sizes_here)}..{max(sizes_here)}")
    for warning in warnings:
        print(f"plan_buckets: warning: {warning}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        return run(args, argv)
    except (OSError, ValueError) as exc:
        print(f"plan_buckets: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
