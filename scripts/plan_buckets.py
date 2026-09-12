"""Build a training bucket plan from real caption lengths and the data mix.

Why the plan needs a sampled-length distribution
------------------------------------------------

The row-length sampler pads every caption in a micro-batch up to its bucket's
upper bound, so a bucket whose bound sits far above the lengths it holds burns
compute on padding.  Choosing those bounds well needs ``p(l)``, the probability
that a training draw ends up with a retained caption length of ``l``.  That is
the distribution of *selected* captions, not of stored ones: the trainer draws
a dataset by its mix weight, then a row uniformly inside that dataset, then
exactly one caption inside that row (see ``RowLengthQueueBatchSampler``).

Two things separate ``p(l)`` from a flat histogram over the corpus:

- Caption selection.  Inside a row, the caption policy gives every caption a
  probability that depends on its length
  (``src/dataset/captions.py:caption_probabilities_from_lengths``).  A row with
  one short and one long caption is not drawn evenly between them, and a
  histogram that counts captions cannot see that.
- The curriculum.  The preference changes over training as ``beta`` ramps from
  its start to its end value.  This planner splits the run into three phases
  (early, middle, late), evaluates the policy at each phase's midpoint, and
  merges the three length distributions with ``--phase-weights``, which default
  to equal thirds ("a third of the steps each").  Note that equal *steps* are
  not equal *sample exposure*: buckets carry different micro-batch sizes, so a
  phase that draws larger batches contributes more samples per step than its
  step share.  The default is an approximation, not a measured split; pass
  ``--phase-weights`` when a better estimate exists.

Boundaries are solved per resolution id: a longer sequence costs
quadratically more attention, and the image-token count sets where the text
length sits in that cost, so the same length distribution gives different
optima for different image shapes.  The exact minimiser of the padding waste
for the given cost model is the dynamic program in
``src/dataset/length_buckets.py``.

Batch sizes are *not* solved here.  Without ``--batch-sizes`` every bucket gets
``--default-batch-size``, which is a clearly-labelled placeholder: the real
numbers come from screening candidate batch sizes per bucket and resolution on
the GPU.

The plan file is written with ``dump_plan`` and carries only what the training
loader reads (resolution id -> [{max_length, batch_size}]).  Everything about
how it was produced goes into the report written next to it, including every
input, the per-resolution length distribution, the boundaries, the padding
waste against the uniform and equal-mass baselines, and the limitations.

Usage:
    .venv/bin/python -m scripts.plan_buckets \
        --mix "path/to/ds_a:0.7 path/to/ds_b:0.3" \
        --image-tokens '{"1": 256, "2": 640}' \
        --buckets 10 --out plan.json
    .venv/bin/python -m scripts.plan_buckets --config configs/base.toml \
        --dataset path/to/ds_a --weight 0.7 --dataset path/to/ds_b --weight 0.3 \
        --out plan.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.dataset.captions import CaptionPolicy, caption_probabilities_from_lengths
from src.dataset.length_buckets import (
    architecture_cost,
    dump_plan,
    equal_mass_boundaries,
    optimal_boundaries,
    padding_waste,
    per_bucket_waste,
    uniform_boundaries,
)
from src.dataset.length_metadata import RowLengthMetadata, sidecar_path
from src.dataset.mix import DatasetEntry, parse_dataset_mix
from src.train.config import DataConfig, ModelConfig
from src.utils.prompt_contract import MAX_SEQUENCE_LENGTH

PHASES = ("early", "middle", "late")
DEFAULT_BUCKETS = 10
# Micro-batch size put in every bucket when no per-bucket table is given. It is
# a placeholder, not a measurement: the caller replaces it with screened sizes
# (see the module docstring).
DEFAULT_BATCH_SIZE = 16
# Image tokens for the cost model when no per-resolution table is given:
# pixels / 256, i.e. the VAE's 8x spatial downscale followed by the DiT's
# patch size 2. A 256x256 image is 256 tokens, a 512x512 image 1024. Resolution
# buckets are chosen at roughly constant pixel area, so one resolution id
# normally has one token count.
DEFAULT_IMAGE_TOKENS = 256
# Lengths above this are reported as the "long tail" of p(l): half the cap.
LONG_CAPTION_TOKENS = MAX_SEQUENCE_LENGTH // 2


# ---------------------------------------------------------------------------
# Curriculum phases.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    """One third of the run, with the caption preference it is modelled at."""

    name: str
    progress: float
    stage: float
    beta: float


def phase_points(policy: CaptionPolicy, curriculum_start: float,
                 curriculum_end: float) -> List[Phase]:
    """The three curriculum phases, each with the beta the policy holds there.

    The run is cut into equal thirds of training progress and the policy is
    evaluated at each phase's midpoint: a phase is summarised by one beta
    instead of an integral over the schedule, because the boundary solver takes
    one distribution per phase.  ``curriculum_start``/``curriculum_end`` map
    training progress onto the value the policy is evaluated at, exactly as the
    training loop does when it advances the stage.
    """
    phases = []
    for index, name in enumerate(PHASES):
        lower = index / len(PHASES)
        upper = (index + 1) / len(PHASES)
        progress = 0.5 * (lower + upper)
        stage = curriculum_start + (curriculum_end - curriculum_start) * progress
        phases.append(Phase(name=name, progress=progress, stage=stage,
                            beta=float(policy.beta(stage))))
    return phases


# ---------------------------------------------------------------------------
# Length distributions.
# ---------------------------------------------------------------------------


@dataclass
class PhaseHistograms:
    """Row-derived length histograms, one per phase and resolution id."""

    resolutions: List[int]
    histograms: np.ndarray                      # (phase, resolution, length)
    row_counts: Dict[int, int] = field(default_factory=dict)
    caption_counts: Dict[int, int] = field(default_factory=dict)
    empty_rows: int = 0


@dataclass
class LengthDistribution:
    """Normalised p(l) for one resolution id, plus its share of all draws."""

    probabilities: np.ndarray
    draw_share: float


def phase_histograms(entries: Sequence[DatasetEntry],
                     metadatas: Sequence[RowLengthMetadata],
                     phases: Sequence[Phase],
                     policy: CaptionPolicy,
                     resolutions: Sequence[int]) -> PhaseHistograms:
    """Accumulate p(l) per phase from the real per-row caption probabilities.

    A draw selects a dataset with its mix weight, a row uniformly inside it,
    and then a caption inside that row, so each row contributes
    ``weight / rows_in_dataset`` times the within-row caption probabilities the
    policy gives it.  Rows without captions cannot produce a draw and are
    counted instead of contributing zero mass silently.
    """
    index_of = {int(resolution): index for index, resolution in enumerate(resolutions)}
    shape = (len(phases), len(resolutions), MAX_SEQUENCE_LENGTH)
    histograms = np.zeros(shape, dtype=np.float64)
    row_counts = {int(resolution): 0 for resolution in resolutions}
    caption_counts = {int(resolution): 0 for resolution in resolutions}
    empty_rows = 0

    for entry, metadata in zip(entries, metadatas):
        if metadata.num_rows == 0:
            raise ValueError(f"{entry.alias}: sidecar describes no rows")
        row_weight = float(entry.weight) / metadata.num_rows
        for row in range(metadata.num_rows):
            resolution = int(metadata.resolution_ids[row])
            resolution_index = index_of[resolution]
            lengths = metadata.prompt_lengths[metadata.row_slice(row)]
            row_counts[resolution] += 1
            caption_counts[resolution] += int(lengths.size)
            if lengths.size == 0:
                empty_rows += 1
                continue
            values = [int(length) for length in lengths.tolist()]
            indices = [value - 1 for value in values]
            for phase_index, phase in enumerate(phases):
                probabilities = caption_probabilities_from_lengths(
                    values, phase.beta,
                    reserve=policy.short_reserve,
                    threshold=policy.short_threshold,
                )
                counts = histograms[phase_index, resolution_index]
                for index, probability in zip(indices, probabilities):
                    counts[index] += row_weight * probability

    return PhaseHistograms(resolutions=[int(resolution) for resolution in resolutions],
                           histograms=histograms, row_counts=row_counts,
                           caption_counts=caption_counts, empty_rows=empty_rows)


def combine_phases(histograms: PhaseHistograms,
                   phase_weights: Sequence[float]) -> Dict[int, LengthDistribution]:
    """Merge the phase histograms into one p(l) per resolution id."""
    combined: Dict[int, LengthDistribution] = {}
    for resolution_index, resolution in enumerate(histograms.resolutions):
        total = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.float64)
        for phase_index, weight in enumerate(phase_weights):
            total += float(weight) * histograms.histograms[phase_index, resolution_index]
        mass = float(total.sum())
        if mass <= 0.0:
            raise ValueError(
                f"resolution {resolution}: no caption mass at all; its rows carry "
                "no captions, so no length distribution can be built for it"
            )
        combined[int(resolution)] = LengthDistribution(
            probabilities=total / mass, draw_share=mass
        )
    return combined


def phase_distributions(histograms: PhaseHistograms) -> Dict[int, Dict[str, np.ndarray]]:
    """Per-phase p(l), normalised per phase so the phases are comparable."""
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for resolution_index, resolution in enumerate(histograms.resolutions):
        per_phase: Dict[str, np.ndarray] = {}
        for phase_index, name in enumerate(PHASES):
            counts = histograms.histograms[phase_index, resolution_index]
            mass = float(counts.sum())
            per_phase[name] = counts / mass if mass > 0 else np.zeros_like(counts)
        out[int(resolution)] = per_phase
    return out


# ---------------------------------------------------------------------------
# Boundaries, batch sizes, and their inputs.
# ---------------------------------------------------------------------------


def solve_boundaries(probabilities: Sequence[float], num_buckets: int,
                     costs: Sequence[float]) -> List[int]:
    """Exact padding-waste minimiser, with the contract's cap as last bound."""
    if num_buckets < 1:
        raise ValueError("--buckets must be at least 1")
    if num_buckets > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"--buckets {num_buckets} exceeds the {MAX_SEQUENCE_LENGTH} retained "
            "lengths the prompt contract allows; a bucket partition cannot have "
            "more buckets than lengths"
        )
    boundaries = optimal_boundaries(probabilities, num_buckets, costs)
    # The dynamic program partitions 1..MAX_SEQUENCE_LENGTH, so the cap is the
    # last bound; assert it rather than trusting the call to keep doing that.
    assert boundaries[-1] == MAX_SEQUENCE_LENGTH
    return boundaries


def quantile_length(probabilities: np.ndarray, quantile: float) -> int:
    """Smallest length whose cumulative mass reaches ``quantile``."""
    cumulative = np.cumsum(probabilities)
    return int(np.searchsorted(cumulative, quantile, side="left")) + 1


def unpadded_cost(probabilities: np.ndarray, costs: np.ndarray) -> float:
    """Expected cost of the padding-free stream, the reference for overhead."""
    return float(np.sum(np.asarray(probabilities) * np.asarray(costs)))


def mix_spec_text(mix: str, datasets: Sequence[str], weights: Sequence[float]) -> str:
    """The mix string to plan for, from ``--mix`` and/or repeated ``--dataset``.

    The text uses the training configuration's syntax (``path:weight
    path:weight``); the repeated flags are a convenience for the same thing, and
    both may be given together. The returned text is what goes into the report,
    so it is the caller's own input rather than a rewrite of it.
    """
    parts: List[str] = []
    if mix.strip():
        parts.extend(mix.split())
    if weights and not datasets:
        raise ValueError("--weight needs a matching --dataset")
    if weights and len(weights) != len(datasets):
        raise ValueError(
            f"got {len(datasets)} --dataset and {len(weights)} --weight values; "
            "they must pair up"
        )
    for index, path in enumerate(datasets):
        if weights:
            parts.append(f"{path}:{weights[index]}")
        else:
            parts.append(path)
    if not parts:
        raise ValueError(
            'no data given: pass --mix "path:weight path:weight" or repeated '
            "--dataset"
        )
    return " ".join(parts)


def parse_mix_spec(mix: str, datasets: Sequence[str],
                   weights: Sequence[float]) -> List[DatasetEntry]:
    """Dataset entries with normalized weights, via the training mix parser."""
    return parse_dataset_mix(mix_spec_text(mix, datasets, weights))


def load_sidecars(entries: Sequence[DatasetEntry]) -> List[RowLengthMetadata]:
    """Load each entry's caption-length sidecar, refusing anything unusable."""
    metadatas = []
    for entry in entries:
        path = sidecar_path(str(entry.path))
        if not path.is_file():
            raise FileNotFoundError(
                f"{entry.alias}: no caption-length sidecar at {path}. The planner "
                "reads retained caption lengths from each dataset's companion "
                "file; build it with "
                "src.dataset.length_metadata.ensure_sidecar(dataset_dir, "
                "tokenizer_path) (training does this on first use) or as part of "
                "the precompute run."
            )
        metadata = RowLengthMetadata.load(path)
        if metadata.metadata_info is None:
            raise ValueError(
                f"{entry.alias}: sidecar {path} carries no prompt contract, so its "
                "lengths may have been produced under a different template or cap. "
                "Rebuild it (ensure_sidecar / build_from_dataset) before planning."
            )
        if metadata.num_captions == 0:
            raise ValueError(f"{entry.alias}: sidecar {path} holds no captions")
        metadatas.append(metadata)
    return metadatas


def read_config(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Merge the sections this planner reads from the given TOML configs.

    Later files override earlier ones, the same order the training
    configuration loader uses. Only flat sections are read; the planner needs
    the caption policy and the model shape, not the whole recipe.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, "rb") as handle:
            payload = tomllib.load(handle)
        for section, values in payload.items():
            if isinstance(values, Mapping):
                merged.setdefault(section, {}).update(values)
    return merged


def pick(cli_value: Any, section: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Command line first, then the config file, then the shipped default."""
    if cli_value is not None:
        return cli_value
    if key in section:
        return section[key]
    return fallback


def load_json_argument(spec: str, name: str) -> Any:
    """Read an inline JSON value or a path to a JSON file, like the trainer does."""
    path = Path(spec)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = spec
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON file path or inline JSON") from exc


def resolve_image_tokens(spec: Optional[str],
                         resolutions: Sequence[int]) -> Dict[int, int]:
    """Image-token count per resolution id, from one number or a JSON mapping."""
    if spec is None:
        value: Any = DEFAULT_IMAGE_TOKENS
    else:
        stripped = spec.strip()
        if stripped.lstrip("+-").isdigit():
            value = int(stripped)
        else:
            value = load_json_argument(spec, "--image-tokens")
    if isinstance(value, bool):
        raise ValueError("--image-tokens must be a positive integer or a JSON object")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"--image-tokens must be positive, got {value}")
        return {int(resolution): value for resolution in resolutions}
    if not isinstance(value, Mapping):
        raise ValueError(
            "--image-tokens must be one positive integer (used for every "
            "resolution) or a JSON object mapping resolution id to token count"
        )
    by_resolution: Dict[int, int] = {}
    for key, tokens in value.items():
        try:
            resolution = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"--image-tokens key {key!r} is not a resolution id")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError(
                f"--image-tokens for resolution {resolution} must be a positive "
                f"integer, got {tokens!r}"
            )
        by_resolution[resolution] = tokens
    missing = sorted(set(int(res) for res in resolutions) - set(by_resolution))
    extra = sorted(set(by_resolution) - set(int(res) for res in resolutions))
    if missing:
        raise ValueError(f"--image-tokens is missing resolution ids {missing}")
    if extra:
        raise ValueError(
            f"--image-tokens lists resolution ids {extra} that are not in the data"
        )
    return by_resolution


def resolve_batch_sizes(spec: Optional[str], resolutions: Sequence[int],
                        num_buckets: int, default_size: int) -> Tuple[Dict[int, List[int]], bool]:
    """Per-bucket micro-batch sizes; returns ``(by_resolution, measured)``.

    ``measured`` is False when the placeholder default filled every bucket, so
    the report and the caller can say so instead of presenting an untested
    table as if it had been screened.
    """
    if default_size < 1:
        raise ValueError(f"--default-batch-size must be positive, got {default_size}")
    if spec is None:
        return ({int(resolution): [default_size] * num_buckets for resolution in resolutions},
                False)

    raw = load_json_argument(spec, "--batch-sizes")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "--batch-sizes must be a JSON object mapping resolution id to a list "
            "of micro-batch sizes, one per bucket"
        )
    by_resolution: Dict[int, List[int]] = {}
    for key, value in raw.items():
        try:
            resolution = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"--batch-sizes key {key!r} is not a resolution id")
        if not isinstance(value, list):
            raise ValueError(
                f"--batch-sizes for resolution {resolution} must be a list of "
                f"micro-batch sizes, got {type(value).__name__}"
            )
        sizes = []
        for size in value:
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"--batch-sizes for resolution {resolution} must be positive "
                    f"integers, got {size!r}"
                )
            sizes.append(int(size))
        if len(sizes) != num_buckets:
            raise ValueError(
                f"--batch-sizes for resolution {resolution} has {len(sizes)} "
                f"entries but --buckets is {num_buckets}; one size per bucket is "
                "required"
            )
        by_resolution[resolution] = sizes

    expected = {int(resolution) for resolution in resolutions}
    missing = sorted(expected - set(by_resolution))
    extra = sorted(set(by_resolution) - expected)
    if missing:
        raise ValueError(
            f"--batch-sizes is missing resolution ids {missing}; the data has "
            f"{sorted(expected)}"
        )
    if extra:
        raise ValueError(
            f"--batch-sizes lists resolution ids {extra} that are not in the data "
            f"{sorted(expected)}"
        )
    return by_resolution, True


# ---------------------------------------------------------------------------
# Solving and reporting.
# ---------------------------------------------------------------------------


@dataclass
class ResolutionResult:
    """Everything the report says about one resolution id."""

    resolution_id: int
    image_tokens: int
    alpha: float
    gamma: float
    rows: int
    captions: int
    draw_share: float
    probabilities: np.ndarray
    boundaries: List[int]
    batch_sizes: List[int]
    optimal_waste: float
    uniform_waste: float
    equal_mass_waste: float
    unpadded: float
    phase_waste: Dict[str, float]
    bucket_mass: List[float]
    bucket_waste: List[float]


@dataclass
class PlanContext:
    """All inputs of one plan, as recorded in the report."""

    out_path: str
    report_path: str
    command: str
    mix_spec: str
    entries: List[DatasetEntry]
    metadatas: List[RowLengthMetadata]
    bucket_count: int
    phases: List[Phase]
    phase_weights: List[float]
    policy: CaptionPolicy
    curriculum_start: float
    curriculum_end: float
    width: int
    layers: int
    image_tokens: Dict[int, int]
    image_tokens_spec: Optional[str]
    batch_sizes_measured: bool
    batch_size_source: str
    config_paths: List[str]
    warnings: List[str]
    empty_rows: int


def solve_resolution(resolution_id: int, distribution: LengthDistribution,
                     num_buckets: int, width: int, layers: int, image_tokens: int,
                     batch_sizes: List[int],
                     phases: Sequence[Phase],
                     per_phase: Mapping[str, np.ndarray],
                     rows: int, captions: int) -> ResolutionResult:
    """Boundaries and waste for one resolution id."""
    model = architecture_cost(width, layers, image_tokens)
    costs = model.over_lengths(MAX_SEQUENCE_LENGTH)
    probabilities = distribution.probabilities
    boundaries = solve_boundaries(probabilities, num_buckets, costs)

    lower = 0
    bucket_mass = []
    for upper in boundaries:
        bucket_mass.append(float(probabilities[lower:upper].sum()))
        lower = upper

    return ResolutionResult(
        resolution_id=int(resolution_id),
        image_tokens=int(image_tokens),
        alpha=float(model.alpha),
        gamma=float(model.gamma),
        rows=int(rows),
        captions=int(captions),
        draw_share=float(distribution.draw_share),
        probabilities=probabilities,
        boundaries=boundaries,
        batch_sizes=list(batch_sizes),
        optimal_waste=padding_waste(probabilities, boundaries, costs),
        uniform_waste=padding_waste(probabilities,
                                    uniform_boundaries(MAX_SEQUENCE_LENGTH, num_buckets),
                                    costs),
        equal_mass_waste=padding_waste(probabilities,
                                       equal_mass_boundaries(probabilities, num_buckets),
                                       costs),
        unpadded=unpadded_cost(probabilities, costs),
        phase_waste={phase.name: padding_waste(per_phase[phase.name], boundaries, costs)
                     for phase in phases},
        bucket_mass=bucket_mass,
        bucket_waste=per_bucket_waste(probabilities, boundaries, costs),
    )


def overhead(waste: float, unpadded: float) -> float:
    """Padding waste as a fraction of the padding-free compute."""
    return waste / unpadded if unpadded > 0 else 0.0


def saved_fraction(waste: float, baseline: float) -> Optional[float]:
    """Share of the baseline's padding waste this plan avoids, if any.

    ``None`` when the baseline already wastes nothing (possible on a
    distribution concentrated on a few lengths), because there is no saving to
    express as a fraction of zero.
    """
    if baseline <= 0:
        return None
    return 1.0 - waste / baseline


def format_saved(saved: Optional[float]) -> str:
    return "n/a" if saved is None else f"{100 * saved:.1f}%"


def format_number(value: float) -> str:
    return f"{value:.4g}"


def render_report(context: PlanContext, results: Sequence[ResolutionResult]) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Bucket plan provenance")
    add("")
    add(f"Plan: `{context.out_path}`")
    add("")
    add("Produced by `scripts/plan_buckets.py`, which reads each dataset's "
        "retained-caption-length sidecar. Command:")
    add("")
    add(f"    {context.command}")
    add("")

    add("## Inputs")
    add("")
    add("Mix as given (weights are normalised to sum to 1 before use):")
    add("")
    add(f"    {context.mix_spec}")
    add("")
    add("| dataset | weight | rows | captions | sidecar | metadata version |")
    add("| --- | --- | --- | --- | --- | --- |")
    for entry, metadata in zip(context.entries, context.metadatas):
        add(f"| {entry.alias} | {entry.weight:.4f} | {metadata.num_rows} | "
            f"{metadata.num_captions} | `{sidecar_path(str(entry.path))}` | "
            f"`{metadata.metadata_version}` |")
    add("")
    inputs = [
        ("buckets K", str(context.bucket_count)),
        ("retained-length cap", f"{MAX_SEQUENCE_LENGTH} "
                                "(`src/utils/prompt_contract.py:MAX_SEQUENCE_LENGTH`)"),
        ("transformer width", str(context.width)),
        ("transformer layers", str(context.layers)),
        ("image tokens", context.image_tokens_spec or
         f"{DEFAULT_IMAGE_TOKENS} for every resolution (built-in default)"),
        ("phase weights (early, middle, late)",
         ", ".join(f"{weight:.4f}" for weight in context.phase_weights)),
        ("caption policy", f"kind={context.policy.kind}, schedule={context.policy.schedule}, "
                           f"beta {context.policy.beta_start} -> {context.policy.beta_end}, "
                           f"reserve={context.policy.short_reserve} below "
                           f"{context.policy.short_threshold} tokens, early_at={context.policy.early_at}"),
        ("curriculum mapping", f"stage = {context.curriculum_start} + "
                               f"({context.curriculum_end} - {context.curriculum_start}) * progress"),
        ("batch sizes", context.batch_size_source),
        ("config files", ", ".join(f"`{path}`" for path in context.config_paths) or "none"),
        ("rows without captions", str(context.empty_rows)),
    ]
    add("| input | value |")
    add("| --- | --- |")
    for name, value in inputs:
        add(f"| {name} | {value} |")
    add("")
    add("Curriculum phases, each evaluated at its midpoint:")
    add("")
    add("| phase | progress | stage | beta |")
    add("| --- | --- | --- | --- |")
    for phase in context.phases:
        add(f"| {phase.name} | {phase.progress:.4f} | {phase.stage:.4f} | {phase.beta:.4f} |")
    add("")

    add("## Length distribution p(l)")
    add("")
    add("`p(l)` is the probability that a training draw selects a caption whose "
        "retained length is `l`, under the mix weights, the per-row caption "
        "probabilities and the phase weights above. It is not a histogram of "
        "stored captions.")
    add("")
    add("| resolution | image tokens | rows | captions | share of draws | p50 | p90 | p99 | "
        f"max | share above {LONG_CAPTION_TOKENS} |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        probabilities = result.probabilities
        support = np.nonzero(probabilities)[0]
        add(f"| {result.resolution_id} | {result.image_tokens} | {result.rows} | "
            f"{result.captions} | {result.draw_share:.4f} | "
            f"{quantile_length(probabilities, 0.5)} | "
            f"{quantile_length(probabilities, 0.9)} | "
            f"{quantile_length(probabilities, 0.99)} | "
            f"{int(support[-1]) + 1 if support.size else 0} | "
            f"{probabilities[LONG_CAPTION_TOKENS:].sum():.4f} |")
    add("")

    add("## Boundaries and padding waste")
    add("")
    add("Waste is the expected padding compute of a plan under the cost proxy "
        "`C(l) = alpha * (I + l) + gamma * (I + l)^2` (alpha/gamma from width and "
        "layers, `I` the image tokens). Overhead is that waste divided by the "
        "padding-free compute of the same stream, so plans and resolutions can be "
        "compared without knowing the proxy's units.")
    add("")
    add("| resolution | boundaries | batch sizes | optimal waste | overhead | "
        "uniform waste | uniform overhead | equal-mass waste | equal-mass overhead | "
        "saved vs uniform | saved vs equal mass |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        saved_uniform = saved_fraction(result.optimal_waste, result.uniform_waste)
        saved_equal = saved_fraction(result.optimal_waste, result.equal_mass_waste)
        add(f"| {result.resolution_id} | "
            f"{', '.join(str(bound) for bound in result.boundaries)} | "
            f"{', '.join(str(size) for size in result.batch_sizes)} | "
            f"{format_number(result.optimal_waste)} | "
            f"{100 * overhead(result.optimal_waste, result.unpadded):.2f}% | "
            f"{format_number(result.uniform_waste)} | "
            f"{100 * overhead(result.uniform_waste, result.unpadded):.2f}% | "
            f"{format_number(result.equal_mass_waste)} | "
            f"{100 * overhead(result.equal_mass_waste, result.unpadded):.2f}% | "
            f"{format_saved(saved_uniform)} | {format_saved(saved_equal)} |")
    add("")

    for result in results:
        add(f"### Resolution {result.resolution_id}")
        add("")
        add(f"Cost model: alpha={format_number(result.alpha)}, "
            f"gamma={format_number(result.gamma)}, image tokens={result.image_tokens}.")
        add("")
        add("| bucket | lengths | mass | batch size | expected waste | waste share |")
        add("| --- | --- | --- | --- | --- | --- |")
        lower = 0
        for index, upper in enumerate(result.boundaries):
            waste = result.bucket_waste[index]
            share = waste / result.optimal_waste if result.optimal_waste > 0 else 0.0
            add(f"| {index + 1} | {lower + 1}-{upper} | {result.bucket_mass[index]:.4f} | "
                f"{result.batch_sizes[index]} | {format_number(waste)} | {100 * share:.2f}% |")
            lower = upper
        add("")
        add("Padding overhead of these boundaries under each phase's own "
            "distribution (what the waste would be if the run stayed in that phase):")
        add("")
        add("| phase | beta | overhead | waste |")
        add("| --- | --- | --- | --- |")
        for phase in context.phases:
            waste = result.phase_waste[phase.name]
            add(f"| {phase.name} | {phase.beta:.4f} | "
                f"{100 * overhead(waste, result.unpadded):.2f}% | {format_number(waste)} |")
        add("")

    add("## Limitations")
    add("")
    add("- **Phase weights are an exposure approximation.** The default equal "
        "thirds means equal steps per phase. Buckets carry different micro-batch "
        "sizes, so a phase whose buckets have larger batches draws more samples "
        "per step than its step share; sample exposure and step share agree only "
        "for a uniform batch size. Pass `--phase-weights` to model a different "
        "split.")
    add("- **Batch sizes are not solved here.** "
        + ("They came from `--batch-sizes` and are whatever that run measured."
           if context.batch_sizes_measured else
           f"Every bucket uses the placeholder `--default-batch-size`, which has "
           f"*not* been measured on a GPU. These sizes must be replaced by a "
           f"per-bucket, per-resolution batch-size screen before training."))
    add("- **The cost model is a proxy.** It counts attention and projection "
        "work, not kernels, the frozen text encoder, the optimizer or the data "
        "path; only the ratio of its linear and quadratic terms affects the "
        "optimum.")
    add("- **Image tokens are an input, not a measurement.** The sidecar records "
        "resolution ids, not image shapes, so each id's token count comes from "
        "`--image-tokens` (see the input table) and should match the precompute "
        "bucket for that id.")
    add("- **Boundaries minimise padding waste, not wall-clock throughput.** A "
        "different boundary can pair with a cheaper batch size or kernel; that "
        "trade is a GPU-screening question, and re-solving with measured "
        "interval times is the follow-up.")
    for warning in context.warnings:
        add(f"- {warning[0].upper()}{warning[1:]}")
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
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS,
                        help=f"number of length buckets per resolution (default {DEFAULT_BUCKETS})")
    parser.add_argument("--out", required=True, help="bucket plan JSON to write")
    parser.add_argument("--report", default=None,
                        help="report path (default: <out>.report.md, next to the plan)")
    parser.add_argument("--batch-sizes", default=None, metavar="JSON",
                        help="JSON file path or inline JSON mapping resolution id to a "
                             "list of per-bucket micro-batch sizes")
    parser.add_argument("--default-batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="placeholder micro-batch size for every bucket when "
                             f"--batch-sizes is absent (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--phase-weights", type=float, nargs=3, default=None,
                        metavar=("EARLY", "MIDDLE", "LATE"),
                        help="phase weights of the early/middle/late length "
                             "distributions (default: equal thirds)")
    parser.add_argument("--image-tokens", default=None, metavar="N|JSON",
                        help="image tokens per resolution id for the cost model: one "
                             "integer for all resolutions, or a JSON object (or path to "
                             f"one) mapping resolution id to token count "
                             f"(default {DEFAULT_IMAGE_TOKENS}, a 256x256 image at 16 "
                             "pixels per token)")
    parser.add_argument("--width", type=int, default=None,
                        help="transformer width for the cost model (default: [model] "
                             "hidden_size of the config, else the shipped recipe)")
    parser.add_argument("--layers", type=int, default=None,
                        help="transformer depth for the cost model (default: [model] "
                             "depths of the config, else the shipped recipe)")
    parser.add_argument("--config", action="append", default=[], metavar="TOML",
                        help="training config to read the caption policy and model "
                             "shape from; repeatable, later files override earlier ones")
    parser.add_argument("--beta-start", type=float, default=None,
                        help="caption length preference at the start of the run")
    parser.add_argument("--beta-end", type=float, default=None,
                        help="caption length preference at the end of the run")
    parser.add_argument("--schedule", choices=("linear", "stationary", "early"),
                        default=None, help="caption beta schedule")
    parser.add_argument("--early-at", type=float, default=None,
                        help="progress at which the 'early' schedule reaches beta_end")
    parser.add_argument("--short-reserve", type=float, default=None,
                        help="probability reserved for a row's short-caption group")
    parser.add_argument("--short-threshold", type=int, default=None,
                        help="captions below this length belong to the short group")
    parser.add_argument("--curriculum-start", type=float, default=None,
                        help="policy value at the start of the run")
    parser.add_argument("--curriculum-end", type=float, default=None,
                        help="policy value at the end of the run")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, argv: Sequence[str] = ()) -> int:
    config = read_config(args.config)
    data_section = config.get("data", {})
    model_section = config.get("model", {})
    warnings: List[str] = []

    policy_kind = pick(None, data_section, "caption_policy", "beta")
    if policy_kind != "beta":
        warnings.append(
            f"the config names caption_policy = {policy_kind!r}, but this planner "
            "always models length-preference selection; p(l) describes the "
            "'beta' policy, not that one")

    mix_text = mix_spec_text(args.mix, args.dataset, args.weight)
    entries = parse_dataset_mix(mix_text)
    metadatas = load_sidecars(entries)
    resolutions = sorted({int(resolution)
                          for metadata in metadatas
                          for resolution in metadata.resolution_ids})
    if not resolutions:
        raise ValueError("the sidecars contain no resolution ids")

    defaults = CaptionPolicy()
    policy = CaptionPolicy(
        kind="beta",
        beta_start=pick(args.beta_start, data_section, "caption_beta_start",
                        defaults.beta_start),
        beta_end=pick(args.beta_end, data_section, "caption_beta_end", defaults.beta_end),
        schedule=pick(args.schedule, data_section, "caption_schedule", defaults.schedule),
        early_at=pick(args.early_at, data_section, "caption_early_at", defaults.early_at),
        short_reserve=pick(args.short_reserve, data_section, "caption_short_reserve",
                           defaults.short_reserve),
        short_threshold=pick(args.short_threshold, data_section,
                             "caption_short_threshold", defaults.short_threshold),
    )
    curriculum_start = float(pick(args.curriculum_start, data_section, "curriculum_start",
                                  DataConfig().curriculum_start))
    curriculum_end = float(pick(args.curriculum_end, data_section, "curriculum_end",
                                DataConfig().curriculum_end))
    phases = phase_points(policy, curriculum_start, curriculum_end)

    if args.phase_weights is None:
        phase_weights = [1.0 / len(PHASES)] * len(PHASES)
    else:
        phase_weights = [float(weight) for weight in args.phase_weights]
        if any(weight < 0 for weight in phase_weights) or sum(phase_weights) <= 0:
            raise ValueError("--phase-weights must be non-negative and not all zero")
        total = sum(phase_weights)
        phase_weights = [weight / total for weight in phase_weights]

    model_defaults = ModelConfig()
    width = int(pick(args.width, model_section, "hidden_size", model_defaults.hidden_size))
    if args.layers is not None:
        layers = int(args.layers)
    elif "single_stream_depth" in model_section or "double_stream_depth" in model_section:
        # The cost model wants total depth, while the config splits it into the
        # single- and double-stream stacks.
        layers = int(model_section.get("single_stream_depth",
                                       model_defaults.single_stream_depth)) + int(
            model_section.get("double_stream_depth", model_defaults.double_stream_depth))
    else:
        layers = model_defaults.single_stream_depth + model_defaults.double_stream_depth
    image_tokens = resolve_image_tokens(args.image_tokens, resolutions)

    if args.image_tokens is None and len(resolutions) > 1:
        warnings.append(
            f"all {len(resolutions)} resolutions are modelled with "
            f"{DEFAULT_IMAGE_TOKENS} image tokens (the built-in default). If they "
            "differ in shape, pass --image-tokens as a resolution id -> token count "
            "map; the text/attention balance, and so the optimal boundaries, "
            "depends on it")

    histograms = phase_histograms(entries, metadatas, phases, policy, resolutions)
    distributions = combine_phases(histograms, phase_weights)
    per_phase = phase_distributions(histograms)

    batch_sizes_by_resolution, measured = resolve_batch_sizes(
        args.batch_sizes, resolutions, args.buckets, args.default_batch_size)
    placeholder_warning = ""
    if measured:
        batch_size_source = ("`--batch-sizes`: measured per-bucket sizes supplied by "
                             "the caller")
    else:
        batch_size_source = (f"`--default-batch-size {args.default_batch_size}` for "
                             "every bucket: placeholder, not measured")
        placeholder_warning = (
            f"batch sizes are placeholders ({args.default_batch_size} for every "
            "bucket), not measurements; replace them with a per-bucket, "
            "per-resolution batch-size screen before training")

    results = []
    for resolution in resolutions:
        distribution = distributions[resolution]
        support = int(np.count_nonzero(distribution.probabilities))
        if support < args.buckets:
            warnings.append(
                f"resolution {resolution}: {args.buckets} buckets requested but only "
                f"{support} distinct retained lengths carry any probability mass; the "
                "remaining buckets split a zero-mass region and add nothing")
        result = solve_resolution(
            resolution, distribution, args.buckets, width, layers,
            image_tokens[resolution], batch_sizes_by_resolution[resolution],
            phases, per_phase[resolution],
            rows=histograms.row_counts[resolution],
            captions=histograms.caption_counts[resolution],
        )
        # The report claims the solved partition beats both baselines; the
        # minimiser cannot lose to a valid partition, so a loss here means the
        # plan must not be trusted.
        for baseline_name, baseline in (("uniform", result.uniform_waste),
                                        ("equal-mass", result.equal_mass_waste)):
            if result.optimal_waste > baseline * (1.0 + 1e-9):
                warnings.append(
                    f"resolution {resolution}: the solved boundaries waste more "
                    f"than the {baseline_name} baseline, which should be "
                    "impossible; do not trust this plan")
        results.append(result)

    boundaries_by_resolution = {result.resolution_id: result.boundaries for result in results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump_plan(str(args.out), boundaries_by_resolution, batch_sizes_by_resolution)

    report_path = args.report or f"{args.out}.report.md"
    command = "python -m scripts.plan_buckets " + shlex.join(argv)
    context = PlanContext(
        out_path=str(args.out),
        report_path=str(report_path),
        command=command,
        mix_spec=mix_text,
        entries=list(entries),
        metadatas=list(metadatas),
        bucket_count=args.buckets,
        phases=phases,
        phase_weights=phase_weights,
        policy=policy,
        curriculum_start=curriculum_start,
        curriculum_end=curriculum_end,
        width=width,
        layers=layers,
        image_tokens=image_tokens,
        image_tokens_spec=args.image_tokens,
        batch_sizes_measured=measured,
        batch_size_source=batch_size_source,
        config_paths=list(args.config),
        warnings=warnings,
        empty_rows=histograms.empty_rows,
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(render_report(context, results), encoding="utf-8")

    print(f"wrote plan: {args.out}")
    print(f"wrote report: {report_path}")
    for result in results:
        saved = saved_fraction(result.optimal_waste, result.uniform_waste)
        print(f"  resolution {result.resolution_id}: boundaries "
              f"{', '.join(str(bound) for bound in result.boundaries)} | batch sizes "
              f"{', '.join(str(size) for size in result.batch_sizes)} | padding "
              f"overhead {100 * overhead(result.optimal_waste, result.unpadded):.2f}% "
              f"({format_saved(saved)} less waste than uniform)")
    if placeholder_warning:
        print(f"plan_buckets: warning: {placeholder_warning}", file=sys.stderr)
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
