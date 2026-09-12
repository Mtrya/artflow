#!/usr/bin/env python3
"""Expected caption-length exposure for the candidate selection policies.

The question this answers
-------------------------
The caption selector decides which caption inside a drawn row is used for
training.  Several candidate policies differ in how strongly they prefer long
captions, and screening them costs GPU card-hours each.  Before spending them,
this tool computes how many draws each candidate sends into every
retained-length band over a run of a chosen size, from the real per-row caption
lengths (each dataset's ``length_metadata.npz`` sidecar) and the real data mix.
The GPU is never needed and never touched.

What one draw is
----------------
The trainer draws a dataset by its mix weight, then a row uniformly inside that
dataset, then exactly one caption inside that row with the caption policy (see
``RowLengthQueueBatchSampler``).  A row with several captions is one sample, not
several, so the expected share of a band is

    p(band) = sum_d w_d / rows_d * sum_rows sum_{c in row and band} p(c | row)

where ``p(c | row)`` is the policy's within-row probability.  The policies
differ only in that distribution:

* ``legacy`` -- ``caption_probabilities_from_token_counts`` on the retained
  lengths, driven by the curriculum position the trainer advances with
  progress;
* ``beta`` -- ``caption_probabilities_from_lengths`` on the retained lengths,
  driven by ``beta`` from the policy's schedule;
* ``stationary`` -- the per-caption probabilities of a reference beta arm
  averaged over training progress and held constant for the whole run.  This is
  the exposure-matched control: averaging preserves the reference arm's total
  exposure, so comparing against it isolates *when* long captions arrive rather
  than how many.  One caveat is measured here rather than assumed away: the
  averaging helper does not hand the reserve back on rows that have no caption
  below the threshold, so those rows do not sum to one and the sampler's
  weighted index gives the difference to the row's last caption.  The report
  prints that construction next to the renormalised, exposure-matched one.

Both selector formulas are reused rather than re-derived.  The functions here
are vectorised forms of them over a whole dataset's flat caption arrays;
``tests/test_exposure_simulation.py`` pins them to the originals caption by
caption, and pins the stationary average to the construction the training
sampler uses.

How the run is integrated
-------------------------
The schedules are continuous in progress, so a run's exposure is the integral
of ``p(band)`` over progress ``[0, 1]``.  It is evaluated on a small uniform
grid (11 points by default) with the trapezoid rule: *numerical integration,
not an exact answer*.  A second, denser grid (21 points by default) is computed
as well and the difference between the two is reported as the integration
error, per band and per arm.  The cumulative exposure at every grid point is
reported too, because two arms can differ in total exposure, in when the
exposure happens, or in both.

The stationary arm has its own averaging grid, separate from the integration
grid, defaulting to 64 points -- the grid ``average_caption_probabilities``
uses inside the training sampler.  That average is itself an approximation, so
the report separates it from the integration error of the other arms.

Cost
----
One vectorised pass over the captions per grid point per dataset; no Python
loop over rows and no per-step simulation.  A corpus of ~2.5M captions and a
handful of datasets finishes in well under a minute on a CPU, so the full-scale
run does not need sampling.  Per-row sums go through ``np.bincount`` over a
precomputed caption-to-row index: ``np.add.reduceat`` would be the shorter
route, but it returns a neighbouring element for a row with no captions and
refuses an index equal to the array length, which trailing empty rows produce.
``python -m scripts.simulate_exposure --help`` documents the arguments.  A run
on the machine that holds the data looks like:

    .venv/bin/python -m scripts.simulate_exposure \\
        --mix "path/to/painting:0.6 path/to/photo:0.4" \\
        --steps 50000 --samples-per-step 16 \\
        --config configs/base.toml \\
        --out exposure.json --report exposure.report.md

``--samples-per-step`` is the number of draws one optimizer step makes, i.e.
the sum of the plan's per-bucket micro-batch sizes; it has no safe default
here, because the bucket plan is what sets it.

The tool reads the same sidecars and the same mix syntax the trainer uses, and
takes every path from the command line: nothing about the machine that holds
the data is baked in.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.plan_buckets import (
    load_sidecars,
    mix_spec_text,
    parse_mix_spec,
    pick,
    read_config,
)
from src.dataset.captions import CaptionPolicy, average_caption_probabilities
from src.dataset.length_metadata import RowLengthMetadata
from src.dataset.mix import DatasetEntry
from src.train.config import DataConfig, TrainLoopConfig

# Retained-length bands the exposure is reported in.  The first band is
# everything below the short-caption threshold the reserve protects.  These
# bands are this tool's own reporting grid; training-side telemetry no longer
# reports banded shares, so compare against its percentiles instead.
BANDS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("<256", 1, 255),
    ("256-511", 256, 511),
    ("512-895", 512, 895),
    ("896-1279", 896, 1279),
    (">=1280", 1280, None),
)
# Upper edges: a length belongs to the first band whose edge is >= it.
BAND_EDGES: Tuple[int, ...] = tuple(low for _, low, _ in BANDS[1:])
# Long bands worth a headline number, as the band index the sum starts at.
LONG_BANDS: Tuple[Tuple[str, int], ...] = (("ge256", 1), ("ge512", 2))

# The legacy selector's clipping range, kept at the values its signature ships
# with so the arm is the selector training uses.
LEGACY_MIN_PROB = 0.15
LEGACY_MAX_PROB = 0.80

DEFAULT_PROGRESS_POINTS = 11
DEFAULT_CHECK_POINTS = 21
# ``average_caption_probabilities``' own default, which is what the training
# sampler's stationary construction uses.
DEFAULT_STATIONARY_GRID = 64
DEFAULT_BETA_SHIFT = 0.5
DEFAULT_STEPS = TrainLoopConfig().max_steps


def progress_grid(points: int) -> np.ndarray:
    """Uniform progress axis over the run, endpoints included."""
    if points < 2:
        raise ValueError("the progress grid needs at least 2 points")
    return np.linspace(0.0, 1.0, int(points))


def trapezoid_weights(points: Sequence[float]) -> np.ndarray:
    """Weights that integrate a function sampled at ``points`` over the run.

    Trapezoid rule on the (uniform) grid: the weights sum to the covered span,
    so for a grid that starts at 0 and ends at 1 they sum to 1 and the weighted
    sum is directly the run-average of the sampled quantity.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.size < 2:
        raise ValueError("integration needs at least 2 grid points")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("the progress grid must cover [0, 1]")
    steps = np.diff(values)
    weights = np.empty_like(values)
    weights[0] = steps[0] / 2.0
    weights[-1] = steps[-1] / 2.0
    if values.size > 2:
        weights[1:-1] = (steps[:-1] + steps[1:]) / 2.0
    return weights


def stage_at(progress: float, curriculum_start: float, curriculum_end: float) -> float:
    """The value the trainer passes to the policy at this progress.

    The training loop advances one scalar from ``curriculum_start`` to
    ``curriculum_end`` over the run and evaluates the policy on it; with the
    shipped endpoints (0 to 1) that scalar is the progress itself.
    """
    return float(curriculum_start + (curriculum_end - curriculum_start) * float(progress))


def band_index(lengths: np.ndarray) -> np.ndarray:
    """Band index per retained length, in ``BANDS`` order."""
    return np.searchsorted(np.asarray(BAND_EDGES), np.asarray(lengths), side="right").astype(
        np.int8
    )


@dataclass
class DatasetView:
    """One dataset's flat caption arrays, prepared once for every grid point.

    ``row_index`` maps each caption to its row, ``row_weights`` holds the
    1 / (rows with captions) share a row contributes to the dataset's exposure,
    and rows without captions get weight zero because they cannot be drawn.
    """

    alias: str
    weight: float
    lengths: np.ndarray
    log_lengths: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    row_index: np.ndarray
    short: np.ndarray
    band: np.ndarray
    row_weights: np.ndarray

    @classmethod
    def from_metadata(
        cls, entry: DatasetEntry, metadata: RowLengthMetadata, short_threshold: int
    ) -> "DatasetView":
        lengths = np.asarray(metadata.prompt_lengths, dtype=np.float64)
        offsets = np.asarray(metadata.caption_offsets, dtype=np.int64)
        counts = np.diff(offsets)
        rows = int(counts.size)
        nonempty = counts > 0
        row_weights = np.zeros(rows, dtype=np.float64)
        row_weights[nonempty] = 1.0 / max(int(nonempty.sum()), 1)
        return cls(
            alias=entry.alias,
            weight=float(entry.weight),
            lengths=lengths,
            # Logs are kept because the beta weights are powers of the lengths
            # and every grid point reuses them.
            log_lengths=np.log(lengths),
            offsets=offsets,
            counts=counts,
            row_index=np.repeat(np.arange(rows, dtype=np.int64), counts),
            short=lengths < float(short_threshold),
            band=band_index(lengths),
            row_weights=row_weights,
        )

    @property
    def rows(self) -> int:
        return int(self.counts.size)

    @property
    def rows_with_captions(self) -> int:
        return int(np.count_nonzero(self.counts))

    @property
    def captions(self) -> int:
        return int(self.lengths.size)


def beta_probabilities(
    view: DatasetView, beta: float, reserve: float, threshold: int
) -> np.ndarray:
    """Vectorised ``caption_probabilities_from_lengths`` for one dataset.

    ``p(c) = (1 - reserve) * L[c]**beta / sum_j L[j]**beta`` per row, plus
    ``reserve / n_short`` for every caption below ``threshold`` in a row that
    has one.  A row without a short caption is left at the bare beta
    distribution: the original only scales a row when that row has a short
    caption to give the reserve to, and scaling the rest would silently drop
    that much of the row's probability mass.  The original subtracts the row's
    largest log-weight before exponentiating; that stabilisation is unnecessary
    here because retained lengths never exceed the prompt contract's cap and
    the schedules keep ``|beta|`` small, so the largest weight that can appear
    is a few times 10**4 -- far from overflowing float64.
    """
    weights = np.power(view.lengths, float(beta))
    row_totals = np.bincount(view.row_index, weights=weights, minlength=view.rows)
    probabilities = weights / row_totals[view.row_index]

    if reserve > 0.0 and bool(view.short.any()):
        short_counts = np.bincount(view.row_index, weights=view.short, minlength=view.rows)
        has_short = short_counts > 0
        probabilities = probabilities * np.where(
            has_short[view.row_index], 1.0 - float(reserve), 1.0
        )
        share = np.divide(
            float(reserve), short_counts, out=np.zeros(view.rows), where=has_short
        )
        probabilities = probabilities + view.short * share[view.row_index]
    return probabilities


def legacy_probabilities(
    view: DatasetView, stage: float, min_prob: float = LEGACY_MIN_PROB,
    max_prob: float = LEGACY_MAX_PROB,
) -> np.ndarray:
    """Vectorised ``caption_probabilities_from_token_counts`` for one dataset.

    Same scores, same per-row normalisation, and the same per-caption clipping
    followed by a renormalisation.  The token counts are the retained lengths,
    which is what the trainer passes to the online selector.
    """
    row_totals = np.bincount(view.row_index, weights=view.lengths, minlength=view.rows)
    means = np.divide(
        row_totals, view.counts, out=np.ones(view.rows), where=view.counts > 0
    )
    deviation = view.lengths / means[view.row_index] - 1.0
    alpha = float(np.clip(stage, 0.0, 1.0))
    scores = np.maximum(1.0 + 2.0 * (alpha - 0.5) * deviation, 1e-6)
    totals = np.bincount(view.row_index, weights=scores, minlength=view.rows)
    probabilities = scores / totals[view.row_index]

    caps = np.where(
        view.counts > 1,
        np.minimum(max_prob, 1.0 - min_prob * (view.counts - 1)),
        1.0,
    )
    caps = np.maximum(caps, min_prob)
    clipped = np.clip(probabilities, min_prob, caps[view.row_index])
    clipped_totals = np.bincount(view.row_index, weights=clipped, minlength=view.rows)
    return clipped / clipped_totals[view.row_index]


def band_masses(view: DatasetView, probabilities: np.ndarray) -> np.ndarray:
    """Row-uniform mass this dataset's draw puts in each band.

    A drawn row contributes its caption probabilities divided by the number of
    rows that can be drawn, so a row with several captions is not over-weighted.
    """
    masses = np.zeros(len(BANDS), dtype=np.float64)
    for index in range(len(BANDS)):
        row_mass = np.bincount(
            view.row_index,
            weights=np.where(view.band == index, probabilities, 0.0),
            minlength=view.rows,
        )
        masses[index] = float(np.dot(view.row_weights, row_mass))
    return masses


def stationary_probabilities(
    view: DatasetView, policy: CaptionPolicy, grid: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-caption probabilities held constant for a whole run.

    The average is over the *probabilities* of the reference schedule, not over
    beta, because the mapping from beta to probabilities is nonlinear; the
    reserve is part of each point's probabilities and therefore part of the
    average.  ``policy`` carries the reference schedule's effective endpoints
    and the sampler's ``stationary`` schedule, whose ``beta()`` is the linear
    ramp those endpoints describe.

    Rows are grouped by caption count so the grid is evaluated once per group
    instead of once per row -- the construction ``RowLengthQueueBatchSampler``
    uses for its stationary arm, with the grid made explicit.  Rows without
    captions are skipped: they cannot be drawn, and skipping them does not move
    any caption's position in the flat length array.

    Returns the probabilities and their per-row sums.  A row that contains a
    short caption sums to one; a row that does not sums to ``1 - reserve``,
    because ``average_caption_probabilities`` scales the row by
    ``1 - reserve`` and only adds the reserved share back where a short
    caption exists.  Both numbers are needed: the sampler draws with the first
    and the report quantifies the second (see ``stationary_band_masses``).
    """
    values = view.lengths
    edges = view.offsets
    counts = view.counts
    probabilities = np.zeros(values.size, dtype=np.float64)
    for count in np.unique(counts):
        rows = np.nonzero(counts == count)[0]
        if count == 0:
            continue
        if count == 1:
            probabilities[edges[rows]] = 1.0
            continue
        matrix = np.stack([values[edges[row]:edges[row + 1]] for row in rows])
        averaged = average_caption_probabilities(matrix, policy, grid=int(grid))
        for position, row in enumerate(rows):
            probabilities[edges[row]:edges[row + 1]] = averaged[position]
    row_totals = np.bincount(view.row_index, weights=probabilities, minlength=view.rows)
    return probabilities, row_totals


def stationary_band_masses(
    view: DatasetView,
    probabilities: np.ndarray,
    row_totals: np.ndarray,
    *,
    complete_missing_mass: bool,
) -> np.ndarray:
    """Band masses for one dataset under the stationary construction.

    The probabilities are not a complete distribution for every row: rows
    without a short caption sum to ``1 - reserve``.  The sampler draws with
    ``_weighted_index``, whose cumulative sum never reaches a uniform draw in
    ``[0, 1)`` that falls past the row total, so it hands that missing mass to
    the row's last caption.  That is what ``complete_missing_mass=True``
    models -- the arm as it would actually run.

    ``complete_missing_mass=False`` renormalises each row first instead.  That
    is the arm as intended: the reserve applies to the short-caption group
    where the row has one, and the beta distribution alone where it does not,
    which is exactly the exposure-matched control the arm exists to be.
    """
    values = probabilities
    if complete_missing_mass:
        deficit = np.clip(1.0 - row_totals, 0.0, 1.0)
        deficit[view.counts == 0] = 0.0
        last_band = np.full(view.rows, -1, dtype=np.int64)
        nonempty = view.counts > 0
        last_band[nonempty] = view.band[view.offsets[1:][nonempty] - 1]
    else:
        safe = np.where(row_totals > 0, row_totals, 1.0)
        values = probabilities / safe[view.row_index]

    masses = np.zeros(len(BANDS), dtype=np.float64)
    for index in range(len(BANDS)):
        row_mass = np.bincount(
            view.row_index,
            weights=np.where(view.band == index, values, 0.0),
            minlength=view.rows,
        )
        if complete_missing_mass:
            row_mass += deficit * (last_band == index)
        masses[index] = float(np.dot(view.row_weights, row_mass))
    return masses


def stationary_exposures(
    views: Sequence[DatasetView], policy: CaptionPolicy, grid: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix-weighted band masses of the stationary construction, both readings.

    Returns ``(as_implemented, exposure_matched)``: the first completes the
    missing mass of rows without short captions on that row's last caption (the
    sampler's actual draw), the second renormalises those rows (the arm as
    intended).  Both come out of one pass because the 64-point averaging is the
    expensive part of this arm.
    """
    as_implemented = np.zeros(len(BANDS), dtype=np.float64)
    exposure_matched = np.zeros(len(BANDS), dtype=np.float64)
    for view in views:
        probabilities, row_totals = stationary_probabilities(view, policy, grid)
        as_implemented += view.weight * stationary_band_masses(
            view, probabilities, row_totals, complete_missing_mass=True
        )
        exposure_matched += view.weight * stationary_band_masses(
            view, probabilities, row_totals, complete_missing_mass=False
        )
    return as_implemented, exposure_matched


@dataclass(frozen=True)
class Arm:
    """One candidate caption-selection policy."""

    key: str
    label: str
    kind: str  # "legacy", "beta", or "stationary"
    policy: CaptionPolicy
    definition: str
    reference: Optional[str] = None


@dataclass(frozen=True)
class Settings:
    """Everything the simulation reads besides the data itself."""

    total_draws: int
    progress_points: int = DEFAULT_PROGRESS_POINTS
    check_points: int = DEFAULT_CHECK_POINTS
    stationary_grid: int = DEFAULT_STATIONARY_GRID
    curriculum_start: float = 0.0
    curriculum_end: float = 1.0
    short_reserve: float = 0.20
    short_threshold: int = 256
    beta_start: float = -1.0
    beta_end: float = 1.0
    early_at: float = 0.5
    beta_shift: float = DEFAULT_BETA_SHIFT


def build_arms(settings: Settings) -> List[Arm]:
    """The five candidate arms, defined the way the training config would them.

    Arm C is the reference ramp with both endpoints shifted up, so it starts
    closer to uniform and ends more long-preferring; arm D is the stationary
    control averaged from arm B; arm E reaches the reference ramp's end value
    at ``early_at`` and holds it there.
    """
    if settings.short_threshold < 1:
        raise ValueError("the short-caption threshold must be positive")
    if not 0.0 <= settings.short_reserve <= 1.0:
        raise ValueError("the short-caption reserve must be within [0, 1]")
    if not 0.0 < settings.early_at <= 1.0:
        raise ValueError("early_at must be within (0, 1]")

    common = dict(
        schedule="linear",
        early_at=float(settings.early_at),
        short_reserve=float(settings.short_reserve),
        short_threshold=int(settings.short_threshold),
    )
    reference = CaptionPolicy(
        kind="beta",
        beta_start=float(settings.beta_start),
        beta_end=float(settings.beta_end),
        **common,
    )
    shifted = CaptionPolicy(
        kind="beta",
        beta_start=float(settings.beta_start) + float(settings.beta_shift),
        beta_end=float(settings.beta_end) + float(settings.beta_shift),
        **common,
    )
    early = CaptionPolicy(
        kind="beta",
        beta_start=float(settings.beta_start),
        beta_end=float(settings.beta_end),
        schedule="early",
        early_at=float(settings.early_at),
        short_reserve=float(settings.short_reserve),
        short_threshold=int(settings.short_threshold),
    )
    # The stationary arm averages the reference ramp over progress.  Its own
    # schedule makes beta() linear in the grid point, so passing the reference
    # ramp evaluated at the curriculum endpoints reproduces the reference ramp
    # composed with the curriculum mapping exactly (both are affine).
    stationary = CaptionPolicy(
        kind="beta",
        beta_start=reference.beta(settings.curriculum_start),
        beta_end=reference.beta(settings.curriculum_end),
        schedule="stationary",
        early_at=float(settings.early_at),
        short_reserve=float(settings.short_reserve),
        short_threshold=int(settings.short_threshold),
    )
    return [
        Arm(
            key="A",
            label="legacy token-count curriculum",
            kind="legacy",
            policy=CaptionPolicy(kind="legacy", **common),
            definition=(
                "caption_probabilities_from_token_counts on the retained lengths, "
                "stage = curriculum mapping of progress"
            ),
        ),
        Arm(
            key="B",
            label="length preference, linear ramp",
            kind="beta",
            policy=reference,
            definition=(
                f"beta {reference.beta_start:g} -> {reference.beta_end:g}, linear over progress"
            ),
        ),
        Arm(
            key="C",
            label="length preference, shifted ramp",
            kind="beta",
            policy=shifted,
            definition=(
                f"beta {shifted.beta_start:g} -> {shifted.beta_end:g}, linear "
                f"(arm B shifted by {settings.beta_shift:g})"
            ),
        ),
        Arm(
            key="D",
            label="stationary control over arm B",
            kind="stationary",
            policy=stationary,
            definition=(
                "per-caption probabilities of arm B averaged over progress "
                f"({settings.stationary_grid} points) and held constant"
            ),
            reference="B",
        ),
        Arm(
            key="E",
            label="length preference, early transition",
            kind="beta",
            policy=early,
            definition=(
                f"beta {early.beta_start:g} -> {early.beta_end:g}, reaching the end "
                f"value at progress {early.early_at:g} and holding it"
            ),
        ),
    ]


def arm_curve(
    arm: Arm, views: Sequence[DatasetView], points: np.ndarray, settings: Settings,
    *, stationary_mass: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Band shares at every progress point, shape ``(points, bands)``."""
    if arm.kind == "stationary":
        # The arm's probabilities do not depend on progress, so one evaluation
        # covers every grid point.  The caller may pass the mass in, so that a
        # simulation can reuse it for the exposure-matched variant instead of
        # paying for the averaging twice.
        if stationary_mass is None:
            stationary_mass = stationary_exposures(
                views, arm.policy, settings.stationary_grid
            )[0]
        return np.tile(stationary_mass, (points.size, 1))

    curve = np.empty((points.size, len(BANDS)), dtype=np.float64)
    for index, progress in enumerate(points):
        mass = np.zeros(len(BANDS), dtype=np.float64)
        stage = stage_at(float(progress), settings.curriculum_start, settings.curriculum_end)
        for view in views:
            if arm.kind == "legacy":
                probabilities = legacy_probabilities(view, stage)
            else:
                probabilities = beta_probabilities(
                    view, arm.policy.beta(stage), settings.short_reserve, settings.short_threshold
                )
            mass += view.weight * band_masses(view, probabilities)
        curve[index] = mass
    return curve


def run_totals(
    curve: np.ndarray, weights: np.ndarray, total_draws: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Run-average band shares and the cumulative counts at every grid point.

    The cumulative counts are the trapezoid integral of the same samples up to
    (and including) each grid point, scaled to the run's draw count, which is
    what separates "different total exposure" from "same total, different
    order".
    """
    fractions = weights @ curve
    cumulative = np.cumsum(weights[:, None] * curve, axis=0) * float(total_draws)
    return fractions, cumulative


def _rounded(value: float, digits: int = 6) -> float:
    number = float(round(float(value), digits))
    return 0.0 if number == 0.0 else number


def simulate(
    entries: Sequence[DatasetEntry],
    metadatas: Sequence[RowLengthMetadata],
    settings: Settings,
    *,
    mix: str,
    command: str,
    size_source: str,
    steps: Optional[int] = None,
    samples_per_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the whole simulation and return a JSON-serialisable payload."""
    if settings.total_draws < 1:
        raise ValueError("the run size must be at least one draw")
    views = [
        DatasetView.from_metadata(entry, metadata, settings.short_threshold)
        for entry, metadata in zip(entries, metadatas)
    ]
    arms = build_arms(settings)
    points = progress_grid(settings.progress_points)
    weights = trapezoid_weights(points)
    if settings.check_points == settings.progress_points:
        check_points, check_weights = points, weights
    else:
        check_points = progress_grid(settings.check_points)
        check_weights = trapezoid_weights(check_points)

    payload_arms: List[Dict[str, Any]] = []
    # The stationary arm's averaging is the expensive part of that arm, and the
    # exposure-matched variant is read off the same numbers; evaluate it once.
    stationary_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
    stationary_arm = next((arm for arm in arms if arm.kind == "stationary"), None)
    if stationary_arm is not None:
        stationary_pair = stationary_exposures(
            views, stationary_arm.policy, settings.stationary_grid
        )
    for arm in arms:
        curve = arm_curve(
            arm, views, points, settings,
            stationary_mass=None if stationary_pair is None else stationary_pair[0],
        )
        fractions, cumulative = run_totals(curve, weights, settings.total_draws)
        if settings.check_points == settings.progress_points:
            check_fractions = fractions
        else:
            check_curve = arm_curve(
                arm, views, check_points, settings,
                stationary_mass=None if stationary_pair is None else stationary_pair[0],
            )
            check_fractions, _ = run_totals(check_curve, check_weights, settings.total_draws)

        bands: Dict[str, Any] = {}
        for index, (name, _, _) in enumerate(BANDS):
            count = float(fractions[index]) * settings.total_draws
            check_count = float(check_fractions[index]) * settings.total_draws
            share = float(fractions[index])
            bands[name] = {
                "fraction": _rounded(share, 9),
                "count": _rounded(count, 3),
                "check_fraction": _rounded(float(check_fractions[index]), 9),
                "check_count": _rounded(check_count, 3),
                "integration_delta": _rounded(check_count - count, 3),
                "noise_sd": _rounded(float(np.sqrt(max(settings.total_draws * share * (1.0 - share), 0.0))), 1),
            }

        long_bands: Dict[str, Any] = {}
        for name, first in LONG_BANDS:
            share = float(fractions[first:].sum())
            count = share * settings.total_draws
            check_count = float(check_fractions[first:].sum()) * settings.total_draws
            cumulative_long = cumulative[:, first:].sum(axis=1)
            half = np.interp(0.5, points, cumulative_long) / count if count > 0 else 1.0
            # Where half of this band's whole-run exposure has been delivered.
            p50 = float(np.interp(0.5 * count, cumulative_long, points)) if count > 1e-12 else None
            long_bands[name] = {
                "fraction": _rounded(share, 9),
                "count": _rounded(count, 3),
                "check_count": _rounded(check_count, 3),
                "integration_delta": _rounded(check_count - count, 3),
                "noise_sd": _rounded(float(np.sqrt(max(settings.total_draws * share * (1.0 - share), 0.0))), 1),
                "delivered_by_midpoint": _rounded(half, 6),
                "p50_progress": None if p50 is None else _rounded(p50, 6),
            }

        payload_arms.append(
            {
                "key": arm.key,
                "label": arm.label,
                "kind": arm.kind,
                "definition": arm.definition,
                "reference": arm.reference,
                "beta_start": None if arm.kind == "legacy" else _rounded(arm.policy.beta_start, 6),
                "beta_end": None if arm.kind == "legacy" else _rounded(arm.policy.beta_end, 6),
                "schedule": arm.policy.schedule,
                "bands": bands,
                "long_bands": long_bands,
                "cumulative_counts": {
                    name: [_rounded(float(value), 3) for value in cumulative[:, index]]
                    for index, (name, _, _) in enumerate(BANDS)
                },
            }
        )

    by_key = {arm["key"]: arm for arm in payload_arms}

    def count_of(key: str, name: str) -> float:
        return by_key[key]["long_bands"][name]["count"]

    pairwise = []
    for left in range(len(payload_arms)):
        for right in range(left + 1, len(payload_arms)):
            first, second = payload_arms[left]["key"], payload_arms[right]["key"]
            row: Dict[str, Any] = {"arms": [first, second]}
            for name, _ in LONG_BANDS:
                delta = count_of(second, name) - count_of(first, name)
                base = count_of(first, name)
                row[f"{name}_delta"] = _rounded(delta, 3)
                row[f"{name}_relative"] = None if base <= 0 else _rounded(delta / base, 6)
            pairwise.append(row)

    integration_deltas = [
        {
            "arm": arm["key"],
            "band": name,
            "delta": arm["bands"][name]["integration_delta"],
            "relative": (
                None
                if arm["bands"][name]["count"] <= 0
                else _rounded(arm["bands"][name]["integration_delta"] / arm["bands"][name]["count"], 6)
            ),
        }
        for arm in payload_arms
        for name, _, _ in BANDS
    ]
    largest = max(integration_deltas, key=lambda item: abs(item["delta"]))

    payload = {
        "tool": "scripts/simulate_exposure.py",
        "command": command,
        "method": {
            "draw": (
                "dataset by mix weight, then a row uniformly inside it, then one caption "
                "inside that row with the selection policy"
            ),
            "integration": (
                f"trapezoid rule on {settings.progress_points} uniform progress points; "
                f"a second {settings.check_points}-point grid gives the integration error"
            ),
            "stationary_grid": (
                f"arm D averages the reference arm's per-caption probabilities over "
                f"{settings.stationary_grid} uniform progress points "
                "(average_caption_probabilities' default, as the training sampler uses it)"
            ),
            "counts": "expectations: total_draws times the run-average band share",
        },
        "inputs": {
            "mix": mix,
            "datasets": [
                {
                    "alias": entry.alias,
                    "path": str(entry.path),
                    "weight": _rounded(entry.weight, 9),
                    "rows": metadata.num_rows,
                    "rows_with_captions": view.rows_with_captions,
                    "captions": metadata.num_captions,
                    "metadata_version": str(metadata.metadata_version),
                }
                for entry, metadata, view in zip(entries, metadatas, views)
            ],
            "total_draws": int(settings.total_draws),
            "steps": None if steps is None else int(steps),
            "samples_per_step": None if samples_per_step is None else int(samples_per_step),
            "size_source": size_source,
            "progress_points": int(settings.progress_points),
            "check_points": int(settings.check_points),
            "stationary_grid": int(settings.stationary_grid),
            "progress_axis": [_rounded(float(point), 6) for point in points],
            "curriculum_start": _rounded(settings.curriculum_start, 6),
            "curriculum_end": _rounded(settings.curriculum_end, 6),
            "short_reserve": _rounded(settings.short_reserve, 6),
            "short_threshold": int(settings.short_threshold),
            "beta_shift_for_arm_C": _rounded(settings.beta_shift, 6),
            "early_at": _rounded(settings.early_at, 6),
            "bands": [
                {"name": name, "low": low, "high": high} for name, low, high in BANDS
            ],
        },
        "arms": payload_arms,
        "pairwise": pairwise,
        "stationary_vs_reference": _stationary_comparison(
            payload_arms, stationary_pair, settings
        ),
        "integration_error": {
            "points": int(settings.progress_points),
            "check_points": int(settings.check_points),
            "largest": largest,
            "per_arm_band": integration_deltas,
        },
    }
    return payload


def _stationary_comparison(
    payload_arms: Sequence[Dict[str, Any]],
    stationary_pair: Optional[Tuple[np.ndarray, np.ndarray]],
    settings: Settings,
) -> Dict[str, Any]:
    """Arm D against arm B, in the two readings of the stationary arm.

    ``as_implemented`` is what the sampler draws today; ``exposure_matched``
    renormalises the rows whose probabilities do not sum to one.  The gap
    between them is an implementation effect, so it is reported next to the
    discretisation residual instead of being folded into it.
    """
    reference = next(arm for arm in payload_arms if arm["key"] == "B")
    implemented = next(arm for arm in payload_arms if arm["key"] == "D")
    reference_long = reference["long_bands"]

    # The reference here is arm B's reported (11-point) count, the same number
    # the tables above show, so every table in the report agrees.  Arm B's own
    # 11-versus-21 error is reported separately in the integration section.
    bands: Dict[str, Any] = {}
    if stationary_pair is not None:
        matched_counts = stationary_pair[1] * settings.total_draws
        for index, (name, _, _) in enumerate(BANDS):
            bands[name] = {
                "count": _rounded(float(matched_counts[index]), 3),
                "delta_vs_reference": _rounded(
                    float(matched_counts[index]) - reference["bands"][name]["count"], 3
                ),
                "implementation_gap": _rounded(
                    implemented["bands"][name]["count"] - float(matched_counts[index]), 3
                ),
            }

    def matched_long(first_band: int) -> float:
        if stationary_pair is None:
            return float("nan")
        return float(stationary_pair[1][first_band:].sum()) * settings.total_draws

    long_bands = {
        name: {
            "as_implemented_count": implemented["long_bands"][name]["count"],
            "exposure_matched_count": _rounded(matched_long(first), 3),
            "reference_count": reference_long[name]["count"],
            "as_implemented_delta_vs_reference": _rounded(
                implemented["long_bands"][name]["count"] - reference_long[name]["count"], 3
            ),
            "exposure_matched_delta_vs_reference": _rounded(
                matched_long(first) - reference_long[name]["count"], 3
            ),
            "implementation_gap": _rounded(
                implemented["long_bands"][name]["count"] - matched_long(first), 3
            ),
        }
        for name, first in LONG_BANDS
    }
    return {
        "arms": ["B", "D"],
        "reserve": _rounded(settings.short_reserve, 6),
        "matched_long_bands": {
            name: {
                "reference_count": reference_long[name]["count"],
                "stationary_count": implemented["long_bands"][name]["count"],
                "delta": _rounded(
                    implemented["long_bands"][name]["count"] - reference_long[name]["count"], 3
                ),
            }
            for name, _ in LONG_BANDS
        },
        "exposure_matched_variant": {"bands": bands, "long_bands": long_bands},
        "note": (
            "Arm D holds arm B's per-caption probabilities averaged over progress, so its "
            "total exposure is meant to equal arm B's and the two arms should differ only "
            "in when the exposure happens. The totals are close but not identical because "
            "both sides are evaluated on finite grids (arm B is a trapezoid integral on few "
            "points, arm D an average on its own grid), and the residual shrinks as both "
            "grids are refined. Two extra columns separate that from an implementation "
            "effect: average_caption_probabilities scales every row by (1 - reserve) but "
            "only adds the reserved share back where the row has a caption below the "
            "threshold, so rows without one sum to (1 - reserve) instead of 1; the "
            "sampler's weighted index then hands the missing mass to that row's last "
            "caption. 'exposure_matched' renormalises those rows instead, which is the "
            "arm's intent and what makes its total match arm B's. The missing mass "
            "always lands on a long caption, so the total above 256 is preserved by "
            "this; the split between bands is not, and neither is the total above 512 "
            "when such a row's last caption is one of its long ones."
        ),
    }


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def _percent(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _axis_text(axis: Sequence[float]) -> str:
    """The progress axis in a report row, spelled out only while that is short."""
    if len(axis) > 21:
        return "uniform"
    return ", ".join(f"{float(point):g}" for point in axis)


def render_report(payload: Dict[str, Any]) -> str:
    inputs = payload["inputs"]
    lines: List[str] = []
    add = lines.append

    add("# Caption-length exposure by selection policy")
    add("")
    add(
        f"Expected number of draws each candidate caption-selection policy puts into "
        f"each retained-length band over a run of {inputs['total_draws']:,} draws"
        + (
            f" ({inputs['steps']:,} steps x {inputs['samples_per_step']} samples per step)."
            if inputs["steps"] and inputs["samples_per_step"]
            else f" ({inputs['size_source']})."
        )
    )
    add("")
    add("Produced by `scripts/simulate_exposure.py`, from each dataset's caption-length "
        "sidecar and the real mix weights. Command:")
    add("")
    add(f"    {payload['command']}")
    add("")

    add("## Inputs")
    add("")
    add("| dataset | weight | rows | rows with captions | captions | sidecar version |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in inputs["datasets"]:
        add(f"| `{row['alias']}` | {row['weight']:.4f} | {row['rows']} | "
            f"{row['rows_with_captions']} | {row['captions']} | `{row['metadata_version']}` |")
    add("")
    settings = [
        ("mix as given", f"`{inputs['mix']}` (weights normalised to sum to 1)"),
        ("run size", f"{inputs['total_draws']:,} draws ({inputs['size_source']})"),
        ("progress grid", f"{inputs['progress_points']} points over [0, 1] "
                          f"({_axis_text(inputs['progress_axis'])}), trapezoid rule; a "
                          f"{inputs['check_points']}-point grid is computed as well for the "
                          "integration error"),
        ("stationary average grid", f"{inputs['stationary_grid']} points"
                                    + (" (`average_caption_probabilities`' default, which is "
                                       "what the training sampler uses)"
                                       if inputs["stationary_grid"] == DEFAULT_STATIONARY_GRID
                                       else " (the sampler's default is "
                                            f"{DEFAULT_STATIONARY_GRID})")),
        ("curriculum mapping", f"stage = {inputs['curriculum_start']} + "
                               f"({inputs['curriculum_end']} - {inputs['curriculum_start']}) "
                               "x progress"),
        ("short-caption reserve", f"{inputs['short_reserve']} of each row's probability "
                                  f"below {inputs['short_threshold']} retained tokens"),
        ("bands (retained tokens)", ", ".join(
            f"`{band['name']}`" if band["high"] is None else
            f"`{band['name']}` ({band['low']}-{band['high']})"
            for band in inputs["bands"])),
    ]
    add("| input | value |")
    add("| --- | --- |")
    for name, value in settings:
        add(f"| {name} | {value} |")
    add("")
    add("Arms:")
    add("")
    add("| arm | policy | selection | schedule |")
    add("| --- | --- | --- | --- |")
    for arm in payload["arms"]:
        add(f"| {arm['key']} | {arm['label']} | {arm['definition']} | {arm['schedule']} |")
    add("")
    add("The five arms are the candidates this tool compares whatever the config's "
        "own `caption_policy` says; the config supplies the shared settings (reserve, "
        "threshold, ramp endpoints and the curriculum mapping), not the arm list.")
    add("")

    add("## Method")
    add("")
    add("One draw is a dataset drawn by its mix weight, then a row drawn uniformly "
        "inside that dataset, then one caption drawn inside that row with the "
        "selection policy; a row with several captions is one sample, not several. "
        "The band shares below are therefore row-weighted, not caption-weighted.")
    add("")
    add("The schedules change over the run, so a share here is the *run-average* of "
        "the per-draw share, integrated over progress with the trapezoid rule on "
        "the grid above. That is numerical integration, not an exact answer; the "
        "next grid is used to estimate the error (see below). Counts are "
        "expectations, `total_draws x share`: an actual run will wobble around them "
        "by the sampling noise reported at the end, so differences much smaller "
        "than that noise are not evidence of anything.")
    add("")
    add("Every row of every listed dataset is read; the tool never samples rows, so "
        "it adds no sampling error of its own on top of the numbers above.")
    add("")

    add("## Expected exposure per band")
    add("")
    add("| arm | " + " | ".join(band["name"] for band in inputs["bands"]) +
        " | >=256 | >=512 |")
    add("| --- | " + " | ".join("---" for _ in inputs["bands"]) + " | --- | --- |")
    for arm in payload["arms"]:
        cells = " | ".join(f"{arm['bands'][band['name']]['count']:,.0f}"
                           for band in inputs["bands"])
        add(f"| {arm['key']} | {cells} | {arm['long_bands']['ge256']['count']:,.0f} | "
            f"{arm['long_bands']['ge512']['count']:,.0f} |")
    add("")
    add("As shares of the run's draws:")
    add("")
    add("| arm | " + " | ".join(band["name"] for band in inputs["bands"]) +
        " | >=256 | >=512 |")
    add("| --- | " + " | ".join("---" for _ in inputs["bands"]) + " | --- | --- |")
    for arm in payload["arms"]:
        cells = " | ".join(_percent(arm["bands"][band["name"]]["fraction"], 3)
                           for band in inputs["bands"])
        add(f"| {arm['key']} | {cells} | {_percent(arm['long_bands']['ge256']['fraction'], 3)} | "
            f"{_percent(arm['long_bands']['ge512']['fraction'], 3)} |")
    add("")

    add("## Pairwise differences in long-caption draws")
    add("")
    add("Every unordered pair of arms, at the two long-caption thresholds. A positive "
        "`delta` means the second arm draws more of that band than the first.")
    add("")
    add("| pair | >=256 delta | >=256 relative | >=512 delta | >=512 relative |")
    add("| --- | --- | --- | --- | --- |")
    for row in payload["pairwise"]:
        add(f"| {row['arms'][0]} - {row['arms'][1]} | {row['ge256_delta']:+,.0f} | "
            f"{_percent(row['ge256_relative'])} | {row['ge512_delta']:+,.0f} | "
            f"{_percent(row['ge512_relative'])} |")
    add("")

    add("## Arm D versus arm B")
    add("")
    add(payload["stationary_vs_reference"]["note"])
    add("")
    comparison = payload["stationary_vs_reference"]
    add("Long-caption totals, three readings of the stationary arm against the "
        "reference ramp it is built from:")
    add("")
    add("| band | reference (arm B) | arm D as implemented | delta | "
        "arm D exposure-matched | delta | implementation gap |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name, _ in LONG_BANDS:
        row = comparison["exposure_matched_variant"]["long_bands"][name]
        add(f"| {name} | {row['reference_count']:,.0f} | "
            f"{row['as_implemented_count']:,.0f} | "
            f"{row['as_implemented_delta_vs_reference']:+,.0f} | "
            f"{row['exposure_matched_count']:,.0f} | "
            f"{row['exposure_matched_delta_vs_reference']:+,.0f} | "
            f"{row['implementation_gap']:+,.0f} |")
    add("")
    add("Per band, including the short one:")
    add("")
    add("| band | arm B | arm D as implemented | arm D exposure-matched | "
        "implementation gap |")
    add("| --- | --- | --- | --- | --- |")
    for band in inputs["bands"]:
        name = band["name"]
        reference = next(arm for arm in payload["arms"] if arm["key"] == "B")
        implemented = next(arm for arm in payload["arms"] if arm["key"] == "D")
        variant = comparison["exposure_matched_variant"]["bands"].get(name, {})
        add(f"| {name} | {reference['bands'][name]['count']:,.0f} | "
            f"{implemented['bands'][name]['count']:,.0f} | "
            f"{variant.get('count', float('nan')):,.0f} | "
            f"{variant.get('implementation_gap', float('nan')):+,.0f} |")
    add("")
    add("`exposure_matched` is the column to read for the ordering comparison: it "
        "holds the design's promise that D's total equals B's, so a difference "
        "between B and this column is the finite-grid residual only. The gap "
        "between the two D columns is not a policy difference: it is how much the "
        "stationary construction moves within rows that have no caption below the "
        "short threshold.")
    add("")

    add("## Integration error")
    add("")
    add(f"Band shares computed on {payload['integration_error']['points']} progress "
        f"points versus {payload['integration_error']['check_points']} points. The "
        "difference estimates how much of any arm-to-arm gap is the integration "
        "rather than the policy:")
    add("")
    add("| arm | band | delta (denser grid - reported) | relative |")
    add("| --- | --- | --- | --- |")
    for item in payload["integration_error"]["per_arm_band"]:
        relative = "n/a" if item["relative"] is None else _percent(item["relative"])
        add(f"| {item['arm']} | {item['band']} | {item['delta']:+,.0f} | {relative} |")
    add("")
    largest = payload["integration_error"]["largest"]
    add(f"Largest absolute deviation: arm {largest['arm']}, band {largest['band']}, "
        f"{largest['delta']:+,.0f} draws. Treat smaller differences between arms as "
        "resolved by the integration alone.")
    add("")

    add("## Exposure over progress")
    add("")
    add("Cumulative counts at each progress point: how much of the run's long-caption "
        "exposure has already been delivered by then. Two arms with the same total "
        "can still differ here, which is what the stationary control arm is for. Each "
        "row is the integral of the exposure from progress 0 to that point, so the "
        "row at 0.0 is half of the first grid interval rather than exactly zero.")
    add("")
    for label, first_band in (
        (">=256 retained tokens", 1),
        (">=512 retained tokens", 2),
    ):
        band_names = [band["name"] for band in inputs["bands"][first_band:]]
        add(f"Cumulative draws with {label}:")
        add("")
        add("| progress | " + " | ".join(arm["key"] for arm in payload["arms"]) + " |")
        add("| --- | " + " | ".join("---" for _ in payload["arms"]) + " |")
        for index, progress in enumerate(inputs["progress_axis"]):
            cells = " | ".join(
                f"{sum(arm['cumulative_counts'][band][index] for band in band_names):,.0f}"
                for arm in payload["arms"])
            add(f"| {progress:.1f} | {cells} |")
        add("")
    add("| arm | >=256 delivered by halfway | >=512 delivered by halfway | "
        "half of the >=256 exposure arrived by | half of the >=512 exposure arrived by |")
    add("| --- | --- | --- | --- | --- |")
    for arm in payload["arms"]:
        ge256 = arm["long_bands"]["ge256"]
        ge512 = arm["long_bands"]["ge512"]
        add(f"| {arm['key']} | {_percent(ge256['delivered_by_midpoint'])} | "
            f"{_percent(ge512['delivered_by_midpoint'])} | "
            f"{'n/a' if ge256['p50_progress'] is None else format(ge256['p50_progress'], '.2f')} | "
            f"{'n/a' if ge512['p50_progress'] is None else format(ge512['p50_progress'], '.2f')} |")
    add("")
    add("The first two columns are how much of that band's whole-run exposure has "
        "already been delivered at progress 0.5; the last two are the progress at "
        "which half of it has arrived (interpolated), so a smaller number means the "
        "preference is delivered earlier. Two arms with the same total exposure and "
        "different numbers here differ in ordering, not in amount.")
    add("")

    add("## Run-to-run noise")
    add("")
    add("The counts above are expectations. If each draw were an independent "
        "Bernoulli trial, the realised count would have standard deviation "
        "`sqrt(N p (1 - p))`, listed per band in the JSON as `noise_sd` and "
        "approximated by:")
    add("")
    add("| arm | >=256 noise sd | >=512 noise sd |")
    add("| --- | --- | --- |")
    for arm in payload["arms"]:
        add(f"| {arm['key']} | ±{arm['long_bands']['ge256']['noise_sd']:,.0f} | "
            f"±{arm['long_bands']['ge512']['noise_sd']:,.0f} |")
    add("")
    add("Caption draws inside one row are correlated, so this is an upper bound, but "
        "it is the right order: differences between arms smaller than a few of these "
        "numbers are not distinguishable in a single run.")
    add("")

    add("## Limitations")
    add("")
    add("- **The counts are expectations, not a simulation of a specific run.** The "
        "sampler shuffles rows and draws them in cycles, so a real run's counts are "
        "these numbers plus noise; only the long-run average is modelled.")
    add("- **The integration is numerical.** 11 progress points resolve a schedule "
        "that changes smoothly, but an arm whose preference turns sharply (the "
        "early-transition arm) is approximated by straight segments between grid "
        "points. The 11-versus-21 comparison above is the size of that error; a "
        "denser `--progress-points` is the way to shrink it.")
    add("- **Arm D's average is itself a finite-grid approximation.** It uses "
        f"{inputs['stationary_grid']} points, the training sampler's own default. "
        "That is the remaining part of the B-versus-D total difference; refining "
        "`--stationary-grid` shrinks it (the test suite checks that both totals "
        "converge to the same value).")
    add("- **The stationary construction loses mass in rows with no short caption.** "
        "`average_caption_probabilities` leaves such a row at `1 - reserve` instead "
        "of 1, and the sampler's weighted index gives the difference to the row's "
        "last caption, so arm D as implemented is not exactly the exposure-matched "
        "control it is meant to be. The 'implementation gap' column in the arm D "
        "section is that effect, measured; it is not a property of the schedule. "
        "That missing mass always lands on a long caption, so the total above 256 "
        "is preserved; the split between bands is not, and neither is the total "
        "above 512 when such a row's last caption is one of its long ones.")
    add("- **Caption dropout is not modelled.** The trainer replaces a share of "
        "selected captions with the empty string for classifier-free guidance; that "
        "happens after selection and scales every arm's counts by the same factor, "
        "so it cannot reorder the arms, but the conditioned exposure is lower than "
        "the selected exposure reported here.")
    add("- **Only the selection policy is modelled, not the length buckets.** "
        "Padding and per-bucket batch sizes have no say in which caption is picked. "
        "The sampler's queues do delay draws -- a row sits in its length bucket's "
        "queue until that bucket has a full micro-batch -- and a queue that is still "
        "incomplete when the run stops is never trained on, so the very last draws of "
        "a run can fall off by up to one micro-batch per bucket shape; that cannot "
        "move a band's share measurably at a screening run's length.")
    add("- **Empty rows are excluded**, because the sampler skips rows without "
        "captions when it builds its draw cycles; the input table reports how many "
        "of each dataset's rows those were.")
    add("- **The last training step does not reach progress 1.0.** The trainer "
        "computes progress as `global_step / max_steps` for steps 0..max_steps-1, so "
        "the realised run is a left-Riemann view of the same integral instead of the "
        "trapezoid one; with the tens of thousands of steps a screening run uses, "
        "that difference is far below the integration error reported here.")
    add("- **Training telemetry bands differ from these.** The trainer logs an "
        "`896_1280` band rather than `896-1279` and `>=1280`; merge this tool's last "
        "two bands before comparing with logged shares.")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mix", default="",
                        help='dataset mix, same syntax as the training config\'s '
                             '[data] mix: "path:weight path:weight", space separated')
    parser.add_argument("--dataset", action="append", default=[], metavar="PATH",
                        help="dataset directory; repeat for several. Can replace --mix")
    parser.add_argument("--weight", action="append", default=[], type=float, metavar="W",
                        help="weight for the matching --dataset, in order; defaults to 1")
    parser.add_argument("--steps", type=int, default=None,
                        help="optimizer steps in the run (default: [train] max_steps of "
                             f"the config, else {DEFAULT_STEPS})")
    parser.add_argument("--samples-per-step", type=int, default=None,
                        help="draws per optimizer step: sum the per-bucket micro-batch "
                             "sizes of the plan the run will use")
    parser.add_argument("--total-draws", type=int, default=None,
                        help="run size in draws, as an alternative to "
                             "--steps x --samples-per-step")
    parser.add_argument("--progress-points", type=int, default=DEFAULT_PROGRESS_POINTS,
                        help=f"progress points for the numerical integration "
                             f"(default {DEFAULT_PROGRESS_POINTS})")
    parser.add_argument("--check-points", type=int, default=DEFAULT_CHECK_POINTS,
                        help="second progress grid whose difference from the first is "
                             f"reported as the integration error (default {DEFAULT_CHECK_POINTS})")
    parser.add_argument("--stationary-grid", type=int, default=DEFAULT_STATIONARY_GRID,
                        help="progress points averaged for the stationary arm; keep the "
                             "sampler's default unless checking that arm's own "
                             f"discretisation (default {DEFAULT_STATIONARY_GRID})")
    parser.add_argument("--out", required=True, help="JSON with the machine-readable numbers")
    parser.add_argument("--report", default=None,
                        help="report path (default: <out>.report.md, next to the JSON)")
    parser.add_argument("--config", action="append", default=[], metavar="TOML",
                        help="training config to read the caption policy and run length "
                             "from; repeatable, later files override earlier ones")
    parser.add_argument("--beta-start", type=float, default=None,
                        help="length preference at the start of arm B (default from the "
                             "config, else -1.0)")
    parser.add_argument("--beta-end", type=float, default=None,
                        help="length preference at the end of arm B (default from the "
                             "config, else 1.0)")
    parser.add_argument("--beta-shift", type=float, default=DEFAULT_BETA_SHIFT,
                        help="how far arm C's ramp is shifted above arm B's "
                             f"(default {DEFAULT_BETA_SHIFT}, i.e. -1..1 becomes -0.5..1.5)")
    parser.add_argument("--early-at", type=float, default=None,
                        help="progress at which arm E reaches its end value (default from "
                             "the config, else 0.5)")
    parser.add_argument("--short-reserve", type=float, default=None,
                        help="probability reserved for a row's short-caption group "
                             "(default from the config, else 0.20)")
    parser.add_argument("--short-threshold", type=int, default=None,
                        help="captions below this length belong to the short group "
                             "(default from the config, else 256)")
    parser.add_argument("--curriculum-start", type=float, default=None,
                        help="policy value at the start of the run (default from the "
                             "config, else 0.0)")
    parser.add_argument("--curriculum-end", type=float, default=None,
                        help="policy value at the end of the run (default from the "
                             "config, else 1.0)")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, argv: Sequence[str] = ()) -> int:
    config = read_config(args.config)
    data_section = config.get("data", {})
    defaults = DataConfig()

    if args.progress_points < 2:
        raise ValueError("--progress-points must be at least 2")
    if args.check_points < 2:
        raise ValueError("--check-points must be at least 2")
    if args.stationary_grid < 2:
        raise ValueError("--stationary-grid must be at least 2")

    if args.total_draws is not None:
        if args.steps is not None or args.samples_per_step is not None:
            raise ValueError(
                "give either --total-draws or --steps with --samples-per-step, not both"
            )
        if args.total_draws < 1:
            raise ValueError("--total-draws must be positive")
        total_draws = int(args.total_draws)
        size_source = "--total-draws"
        steps = samples_per_step = None
    else:
        if args.samples_per_step is None:
            raise ValueError(
                "give --total-draws, or --steps with --samples-per-step, so the run "
                "size is known"
            )
        if args.samples_per_step < 1:
            raise ValueError("--samples-per-step must be positive")
        train_section = config.get("train", {})
        steps = int(pick(args.steps, train_section, "max_steps", DEFAULT_STEPS))
        if steps < 1:
            raise ValueError("--steps must be positive")
        samples_per_step = int(args.samples_per_step)
        total_draws = steps * samples_per_step
        size_source = "--steps x --samples-per-step"

    settings = Settings(
        total_draws=total_draws,
        progress_points=args.progress_points,
        check_points=args.check_points,
        stationary_grid=args.stationary_grid,
        curriculum_start=float(pick(args.curriculum_start, data_section,
                                    "curriculum_start", defaults.curriculum_start)),
        curriculum_end=float(pick(args.curriculum_end, data_section,
                                  "curriculum_end", defaults.curriculum_end)),
        short_reserve=float(pick(args.short_reserve, data_section,
                                 "caption_short_reserve", defaults.caption_short_reserve)),
        short_threshold=int(pick(args.short_threshold, data_section,
                                 "caption_short_threshold", defaults.caption_short_threshold)),
        beta_start=float(pick(args.beta_start, data_section,
                              "caption_beta_start", defaults.caption_beta_start)),
        beta_end=float(pick(args.beta_end, data_section,
                            "caption_beta_end", defaults.caption_beta_end)),
        early_at=float(pick(args.early_at, data_section,
                            "caption_early_at", defaults.caption_early_at)),
        beta_shift=float(args.beta_shift),
    )

    mix_text = mix_spec_text(args.mix, args.dataset, args.weight)
    entries = parse_mix_spec(mix_text, [], [])
    metadatas = load_sidecars(entries)

    command = "python -m scripts.simulate_exposure " + shlex.join(argv)
    payload = simulate(
        entries, metadatas, settings,
        mix=mix_text, command=command, size_source=size_source,
        steps=steps, samples_per_step=samples_per_step,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = Path(args.report or f"{args.out}.report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload), encoding="utf-8")

    print(f"wrote exposure: {out_path}")
    print(f"wrote report: {report_path}")
    for arm in payload["arms"]:
        long_bands = arm["long_bands"]
        print(f"  arm {arm['key']}: >=256 {long_bands['ge256']['count']:,.0f} draws "
              f"({_percent(long_bands['ge256']['fraction'])}), >=512 "
              f"{long_bands['ge512']['count']:,.0f} draws "
              f"({_percent(long_bands['ge512']['fraction'])})")
    gap = payload["stationary_vs_reference"]["exposure_matched_variant"]["long_bands"]
    material = [
        name for name, row in gap.items()
        if row["reference_count"] > 0
        and abs(row["implementation_gap"]) > 0.01 * row["reference_count"]
    ]
    if material:
        details = ", ".join(
            f"{name} {gap[name]['implementation_gap']:+,.0f} draws "
            f"({_percent(gap[name]['implementation_gap'] / gap[name]['reference_count'])})"
            for name in material
        )
        print(
            f"simulate_exposure: warning: the stationary arm as currently constructed "
            f"does not have the reference arm's exposure in {details}; see the arm D "
            "versus arm B section of the report",
            file=sys.stderr,
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        return run(args, argv)
    except (OSError, ValueError) as exc:
        print(f"simulate_exposure: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
