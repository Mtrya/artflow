"""Screen micro-batch sizes for every (resolution, caption-length) bucket.

Why the sizes have to be measured
---------------------------------

A bucket plan fixes, per bucket, the padded text width of a micro-batch and the
number of samples in it.  The same number cannot serve every bucket: the padded
width of a long-caption bucket makes a large micro-batch overflow memory, while
a short bucket run at a small micro-batch leaves the GPU idle.  The boundaries
come from ``scripts/plan_buckets.py``; the batch sizes it writes are explicitly
labelled placeholders.  This script replaces them with measured ones, per
bucket: for each candidate size it measures the cost of one sample and keeps the
size whose per-sample cost is lowest (not the largest size that fits).

How one run isolates one bucket
-------------------------------

The training sampler draws a dataset by its mix weight, a row inside it, then
one caption inside that row; it assigns the draw to the first bucket whose bound
covers the retained caption length, and it emits a micro-batch only once that
bucket's own queue has reached the bucket's own batch size
(``src/dataset/sampler.py:RowLengthQueueBatchSampler``).  A run of the real plan
therefore mixes every bucket whose queue happens to fill, and the trainer's
throughput summary is a weighted average over that mix.  Nothing in the log
attributes a timing to one bucket, and no training flag restricts the sampler to
a single bucket.

To attribute a timing to one bucket, a screening run uses a plan in which only
the bucket under test has a real batch size: every other bucket - the rest of
that resolution's buckets and every bucket of the other resolutions - carries a
*sink* size, chosen far above the number of draws the run can possibly make.
Those queues never fill, so they never emit, and the micro-batches that reach
the GPU all belong to the bucket under test, with its padded width and its own
caption lengths.  The plan is a measurement artefact: it is never written to the
screened plan.

The sampler still draws the other buckets' rows and drops them, so the measured
step time includes sampling work a real run would not do.  That overhead is a
per-sample constant for a given bucket (all candidates of that bucket draw with
the same sink plan, so the same amount of discarded work is in every candidate),
which is why it does not change the ranking; the report prints a measured
estimate of its size so the absolute per-sample time can be read with it in
mind.  The ``combinations`` that would need a wasteful number of discards are
not screened at all - see ``--min-caption-share``.

Every run is checked against the sink: the summary's ``samples_per_step`` must
equal ``gradient_accumulation_steps * batch_size``, which can only hold when no
other bucket emitted.  A run that fails the check is recorded as a failure and
excluded from the selection.

What is measured, and how
-------------------------

Runs are launched exactly like the throughput comparison in
``scripts/bench/gate_ab.py``: a temporary TOML overrides ``[data]`` and
``[train]`` on top of the base config, and ``src.train.train`` runs for a short
number of optimizer steps with tracker logging disabled.  The trainer's own
timer covers the whole step - latent preparation, caption dropout, text
encoding, DiT forward and backward, the optimizer, EMA and logging - and
excludes only evaluation and checkpoint work, so text encoding and input
preparation are inside the measured interval by construction.

Two figures are recorded per run, as the trainer reports them: the all-steps
average (which carries the one-time ``torch.compile`` stall and any other
start-up cost of the shape) and the steady average over the steps after
``steady_state_skip_steps``.  The selection uses the steady figure; the ratio
between the two shows what compile warm-up cost.

Memory is read after the optimizer state exists, because that state is what
makes the last few gigabytes of a training step unavailable.  The trainer's
``[throughput-summary]`` reports peak *allocated* and peak *reserved*, and the
screening reads both from it.  ``--mem-probe`` can sample reserved memory from
inside the training process instead, for a trainer that does not report it, but
it is off by default: injecting a module into someone else's process is one more
thing that can break the run.  A candidate that runs out of memory is recorded
as an OOM and skipped; the scan continues.

Each candidate is run ``--repeats`` times so the report can state how much the
measurement moves on its own, and candidates whose per-sample times overlap
within that spread are reported as indistinguishable (the smaller size wins
then, because it costs less memory).

Caching
-------

Measurements are cached in a JSON file given by ``--cache``, keyed by a digest
of everything the numbers depend on: the model recipe, the GPU, the software and
execution settings, and the length distribution the run draws from.  A key that
differs in any of those does not reuse the other key's runs.

Outputs
-------

``--out`` (the plan with measured sizes), ``--measurements`` (every run of every
candidate, machine readable), ``--report`` (what was screened, what was
extrapolated or not screened, and the memory margins), and one directory per run
under ``--run-dir`` holding the temporary TOML and the full training log.

A first pass usually sweeps a coarse grid over every bucket; a follow-up pass
then probes the sizes between one that passed and one that failed with
``--bucket-batch-sizes``, which names candidates per bucket and skips every
bucket it does not list.

Usage:
    python -m scripts.bench.batch_size_screen \
        --plan plan.json --mix "data/a@256p:0.7 data/b@256p:0.3" \
        --batch-sizes 4 8 12 16 24 32 --steps 30 --repeats 2 \
        --vae /path/to/vae --text-encoder /path/to/tokenizer \
        --out plan.screened.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.dataset.length_buckets import dump_plan
from src.dataset.length_metadata import RowLengthMetadata, ensure_sidecar
from src.dataset.mix import DatasetEntry, parse_dataset_mix
from src.dataset.sampler import BucketPlan, LenBucket, RowLengthQueueBatchSampler
from src.utils.prompt_contract import MAX_SEQUENCE_LENGTH

DEFAULT_CANDIDATES = (4, 8, 16, 32)
DEFAULT_STEPS = 30
DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 2
DEFAULT_MIN_CAPTION_SHARE = 0.05
DEFAULT_MIN_MARGIN_FRACTION = 0.10
DEFAULT_TIMEOUT_S = 3600.0
# The sink is the batch size given to every bucket that must not emit.  A run
# draws at most (emitted samples / share) rows, so the sink is set a safety
# factor above the largest number of samples any run can emit.  Only one sink
# entry per resolution is written, and a sink that did fill would emit an
# enormous batch, so the value is deliberately far above what a short screen
# run can reach; a run that still emits one is detected by the samples-per-step
# check.
SINK_SAFETY_FACTOR = 64
SINK_FLOOR = 4096
# Environment the screen sets for the training runs unless the caller already
# set it.  Tracker logging is network I/O inside the training loop, and these
# runs are measurements rather than experiments, so it is switched off; the
# parallelism knob is the one gate_ab.py sets, kept here so a screening run and
# a throughput run behave alike.
DEFAULT_RUN_ENV = {
    "SWANLAB_MODE": "disabled",
    "TOKENIZERS_PARALLELISM": "false",
    # A run's stdout is a pipe, so Python block-buffers it and nothing reaches
    # the log until the process exits.  A run that stalls mid-step then looks
    # exactly like a run that never started, and there is no way to tell where
    # it stopped.  Line buffering costs nothing measurable at the log interval.
    "PYTHONUNBUFFERED": "1",
}
# Environment variables that change the measured time or memory, so they belong
# in the cache key whichever side set them.  The reference throughput harness
# runs with PYTORCH_ALLOC_CONF=expandable_segments:True; this screen does not
# set it, because the plain allocator is the conservative choice for a memory
# margin, so pass it with --env when that is what training runs with.
ENV_KEYS_IN_KEY = (
    "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
    "SWANLAB_MODE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
)
# Keys of the pinned [train]/[eval] settings that define the measured interval.
CACHE_VERSION = 1

# Memory probe: planted as ``sitecustomize`` on the training process' path.
# It exists for a trainer that does not report reserved memory itself: the
# allocator's ceiling cannot be reconstructed from outside the process.  This
# samples the two sticky peaks - which the trainer resets once per optimizer
# step - from a daemon thread, and prints them so the screen can parse them out
# of the run's log.  It never imports torch itself: it waits for the training
# process to import and initialize CUDA, so it cannot change device setup.
# Off by default; the trainer reports both peaks, so this is the fallback.
MEM_PROBE_SOURCE = '''\
"""Sample the CUDA allocator's peaks from inside a training process.

Planted by scripts/bench/batch_size_screen.py on PYTHONPATH.  Writes one
[mem-probe] line every few seconds with the sticky peaks since the last
torch.cuda.reset_peak_memory_stats() call, which the training loop makes once
per optimizer step.  Doing nothing is a valid outcome: if torch never
initializes CUDA there is nothing to sample.
"""

import sys
import threading
import time

POLL_S = 0.01
PRINT_S = 2.0


def _gb(value):
    return value / 1024 ** 3


def _probe():
    torch = None
    while True:
        torch = sys.modules.get("torch")
        if torch is not None:
            break
        time.sleep(0.05)
    while not torch.cuda.is_initialized():
        time.sleep(0.05)
    last = 0.0
    while True:
        try:
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
        except Exception:
            return
        now = time.monotonic()
        if now - last >= PRINT_S:
            last = now
            print(
                f"[mem-probe] peak_allocated_gb={_gb(allocated):.3f} "
                f"peak_reserved_gb={_gb(reserved):.3f}",
                flush=True,
            )
        time.sleep(POLL_S)


if "torch" not in sys.modules:
    threading.Thread(target=_probe, daemon=True).start()
'''


# ---------------------------------------------------------------------------
# Command line helpers.
# ---------------------------------------------------------------------------


def parse_int_list(values: Sequence[str], name: str) -> List[int]:
    """Parse one or more ints, accepting "4 8 12" and "4,8,12" alike."""
    out: List[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            try:
                number = int(part)
            except ValueError:
                raise ValueError(f"{name}: {part!r} is not an integer") from None
            if number < 1:
                raise ValueError(f"{name}: {number} must be positive")
            out.append(number)
    if not out:
        raise ValueError(f"{name} must list at least one value")
    if len(set(out)) != len(out):
        raise ValueError(f"{name}: duplicate values in {out}")
    return out


def parse_bucket_batch_sizes(values: Sequence[str],
                             name: str) -> Dict[Tuple[Optional[int], int], List[int]]:
    """Parse per-bucket candidate lists for a follow-up pass.

    Each entry is ``bucket:sizes`` (applies to that bucket in every screened
    resolution) or ``resolution:bucket:sizes`` (one resolution only), with the
    sizes comma-separated: ``0:80,96`` or ``4:0:80,96``.  Buckets an entry does
    not name are skipped, so a dense pass can probe only the sizes between one
    that passed and one that failed.
    """
    out: Dict[Tuple[Optional[int], int], List[int]] = {}
    for value in values:
        head, sep, rest = str(value).partition(":")
        if not sep:
            raise ValueError(
                f"{name}: {value!r} needs the form bucket:sizes or "
                "resolution:bucket:sizes")
        resolution: Optional[int] = None
        bucket_text = head
        sizes_text = rest
        if ":" in rest:
            bucket_text, _, sizes_text = rest.partition(":")
            try:
                resolution = int(head)
            except ValueError:
                raise ValueError(f"{name}: {head!r} is not a resolution id") from None
        try:
            bucket = int(bucket_text)
        except ValueError:
            raise ValueError(f"{name}: {bucket_text!r} is not a bucket index") from None
        key = (resolution, bucket)
        if key in out:
            raise ValueError(f"{name}: duplicate entry {value!r}")
        out[key] = parse_int_list([sizes_text], name)
    return out


def parse_key_values(values: Sequence[str], name: str) -> Dict[str, str]:
    """Parse repeated KEY=VALUE arguments."""
    out: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{name}: {value!r} must be KEY=VALUE")
        key, _, text = value.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"{name}: {value!r} has an empty key")
        out[key] = text
    return out


# ---------------------------------------------------------------------------
# The plan under test.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bucket:
    """One bucket of the plan: its padded width and the size used for it."""

    max_length: int
    batch_size: int


def read_plan(path: str) -> Dict[int, List[Bucket]]:
    """Read the plan JSON, keeping its resolution ids and bucket order.

    The shape is the one ``src.train.train.load_bucket_plan`` reads
    (resolution id -> [{max_length, batch_size}, ...]); only the batch sizes are
    rewritten by this script, so the file is read as-is rather than through the
    trainer's loader, which would drop the resolutions the caller is not
    screening and would not survive a round trip unchanged.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("the plan must be a JSON object keyed by resolution id")
    plan: Dict[int, List[Bucket]] = {}
    for key, buckets in raw.items():
        try:
            resolution_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"plan key {key!r} is not a resolution id") from None
        if not isinstance(buckets, list) or not buckets:
            raise ValueError(f"plan resolution {resolution_id} has no buckets")
        entries = []
        for bucket in buckets:
            if isinstance(bucket, dict):
                try:
                    entries.append(Bucket(int(bucket["max_length"]),
                                          int(bucket["batch_size"])))
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        f"plan resolution {resolution_id}: every bucket needs "
                        "max_length and batch_size"
                    ) from None
            elif isinstance(bucket, (list, tuple)) and len(bucket) == 2:
                entries.append(Bucket(int(bucket[0]), int(bucket[1])))
            else:
                raise ValueError(
                    f"plan resolution {resolution_id}: a bucket must be "
                    "{max_length, batch_size} or [max_length, batch_size]"
                )
        plan[resolution_id] = entries
    return plan


def plan_bounds(plan: Mapping[int, Sequence[Bucket]]) -> Dict[int, List[int]]:
    return {int(r): [bucket.max_length for bucket in buckets]
            for r, buckets in plan.items()}


def plan_sizes(plan: Mapping[int, Sequence[Bucket]]) -> Dict[int, List[int]]:
    return {int(r): [bucket.batch_size for bucket in buckets]
            for r, buckets in plan.items()}


def apply_sizes(plan: Mapping[int, Sequence[Bucket]],
                sizes: Mapping[Tuple[int, int], int]) -> Dict[int, List[Bucket]]:
    """Return a copy of the plan with some (resolution, bucket) sizes replaced."""
    out: Dict[int, List[Bucket]] = {}
    for resolution_id, buckets in plan.items():
        out[int(resolution_id)] = [
            Bucket(bucket.max_length,
                   int(sizes[(int(resolution_id), index)])
                   if (int(resolution_id), index) in sizes else bucket.batch_size)
            for index, bucket in enumerate(buckets)
        ]
    return out


def write_plan(path: str, plan: Mapping[int, Sequence[Bucket]]) -> None:
    """Write the plan in the loader's shape, with no extra top-level keys."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump_plan(path, plan_bounds(plan), plan_sizes(plan))


def plan_spec(plan: Mapping[int, Sequence[Bucket]]) -> str:
    """The plan as inline JSON, the form ``[data] bucket_plan`` also accepts."""
    return json.dumps(
        {str(resolution_id): [
            {"max_length": bucket.max_length, "batch_size": bucket.batch_size}
            for bucket in buckets
        ] for resolution_id, buckets in plan.items()},
        sort_keys=True,
    )


def load_plan_for_trainer(spec: str, resolution_ids: Sequence[int]) -> BucketPlan:
    """Validate a plan with the trainer's own loader and return its buckets.

    Reading through ``load_bucket_plan`` is what makes "a plan this script
    accepts" and "a plan training accepts" the same statement: it applies the
    same shape checks and the same resolution coverage check the training run
    will apply.  Imported here rather than at module level so the pure helpers
    above stay importable without the training stack.
    """
    from src.train.train import load_bucket_plan

    return load_bucket_plan(spec, resolution_ids)


def screening_plan(plan: Mapping[int, Sequence[Bucket]], resolution_id: int,
                   bucket_index: int, candidate: int, sink: int) -> Dict[int, List[Bucket]]:
    """The plan one screening run uses: the target bucket, and sinks elsewhere.

    Every bucket keeps its own bound, so the sampler assigns every drawn length
    exactly where the real plan assigns it: the target bucket sees its own
    caption lengths, and the sink buckets only absorb the rows that the run is
    not measuring.  A sink size never emits within a run (see
    ``sink_batch_size``), which is what makes the run's micro-batches all belong
    to the target bucket.
    """
    if not 1 <= candidate <= sink:
        raise ValueError(f"candidate {candidate} must be within [1, {sink}]")
    screened: Dict[int, List[Bucket]] = {}
    for other_id, buckets in plan.items():
        screened[int(other_id)] = [
            Bucket(bucket.max_length,
                   int(candidate) if (int(other_id) == resolution_id and index == bucket_index)
                   else int(sink))
            for index, bucket in enumerate(buckets)
        ]
    return screened


def sink_batch_size(steps: int, accumulation: int, candidates: Sequence[int]) -> int:
    """A batch size no screening run can reach, for every bucket but the target.

    A run emits at most ``steps * accumulation * max(candidate)`` samples, and
    it draws at most one row per emitted sample divided by the target bucket's
    share of the draws - so the number of rows that can land in any one sink
    queue is bounded by that product times the inverse share.  The safety factor
    covers shares down to a few percent, which is also the point below which
    ``--min-caption-share`` refuses to screen a bucket.
    """
    emitted = max(1, int(steps)) * max(1, int(accumulation)) * max(int(c) for c in candidates)
    return max(SINK_FLOOR, SINK_SAFETY_FACTOR * emitted)


# ---------------------------------------------------------------------------
# The distribution the runs draw from.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetLengths:
    """One mix entry together with the caption lengths its sidecar holds."""

    alias: str
    weight: float
    rows: int
    metadata: RowLengthMetadata


def load_dataset_lengths(entries: Sequence[DatasetEntry],
                         tokenizer: str) -> List[DatasetLengths]:
    """Load each dataset's caption-length sidecar, building it if needed.

    Building a missing sidecar tokenizes the dataset, which is why the run
    would otherwise pay for it inside the first screening run: ``gate_ab.py``
    prepares them the same way, with the same tokenizer.
    """
    out = []
    for entry in entries:
        metadata = ensure_sidecar(str(entry.path), tokenizer)
        if metadata.num_rows == 0:
            raise ValueError(f"{entry.alias}: sidecar describes no rows")
        out.append(DatasetLengths(alias=entry.alias, weight=float(entry.weight),
                                  rows=int(metadata.num_rows), metadata=metadata))
    return out


def lengths_for_resolution(metadata: RowLengthMetadata, resolution_id: int) -> np.ndarray:
    """Caption lengths of the rows that carry this resolution id."""
    captions_per_row = np.diff(metadata.caption_offsets)
    caption_resolution = np.repeat(metadata.resolution_ids, captions_per_row)
    return metadata.prompt_lengths[caption_resolution == int(resolution_id)]


def bucket_index_of(lengths: Sequence[int], bounds: Sequence[int]) -> np.ndarray:
    """Bucket index of each length, using the sampler's own assignment rule.

    ``BucketPlan.bucket_for`` gives a length to the first bucket whose bound is
    at least the length, so the plan must cover the largest caption or training
    cannot place it at all; ``side="left"`` is what reproduces that rule for a
    bound that a caption hits exactly.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    bounds = np.asarray(bounds, dtype=np.int64)
    index = np.searchsorted(bounds, lengths, side="left")
    if index.size and int(index.max()) >= bounds.size:
        raise ValueError(
            f"the plan's last bound {int(bounds[-1])} is below a caption of "
            f"{int(lengths[index.argmax()])} tokens, so those rows have no bucket"
        )
    return index


def bucket_draw_mass(records: Sequence[DatasetLengths], resolution_id: int,
                     bounds: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Draw mass and caption count of each bucket, before the caption policy.

    A draw picks a dataset with its mix weight, then a row uniformly inside it,
    then one of that row's captions, so a caption's own draw probability is
    ``weight / rows`` times the policy's probability for that caption.  Summing
    that weight over the captions of a bucket gives its share of draws up to the
    policy factor, which is the honest thing this screen knows about a bucket
    before running anything.  A bucket far below ``--min-caption-share`` is not
    screened: filling even one of its micro-batches would need a wasteful number
    of draws.
    """
    mass = np.zeros(len(bounds), dtype=np.float64)
    counts = np.zeros(len(bounds), dtype=np.int64)
    for record in records:
        lengths = lengths_for_resolution(record.metadata, resolution_id)
        if lengths.size == 0:
            continue
        index = bucket_index_of(lengths, bounds)
        bucket_counts = np.bincount(index, minlength=len(bounds))
        counts += bucket_counts[:len(bounds)]
        mass += bucket_counts[:len(bounds)] * (record.weight / record.rows)
    return mass, counts


def bucket_draw_mass_by_resolution(plan: Mapping[int, Sequence[Bucket]],
                                   resolutions: Sequence[int],
                                   records: Sequence[DatasetLengths]
                                   ) -> Dict[int, List[float]]:
    """The share of all draws every bucket takes, for aligning buckets in time.

    ``bucket_draw_mass`` gives the raw mass of each bucket of one resolution;
    dividing by the total over every requested resolution turns the per-bucket
    numbers into one distribution over (resolution, bucket) pairs, which is
    what a training rank draws its micro-batches from.  The result is the
    ``--bucket-mass`` file ``merge_screen_results.py`` reads: one value per
    bucket of the plan, resolution by resolution, summing to 1 over the table.
    """
    raw: Dict[int, np.ndarray] = {}
    for resolution_id in resolutions:
        bounds = [bucket.max_length for bucket in plan[int(resolution_id)]]
        bucket_mass, _ = bucket_draw_mass(records, int(resolution_id), bounds)
        raw[int(resolution_id)] = bucket_mass
    total = float(sum(float(bucket_mass.sum()) for bucket_mass in raw.values()))
    if total <= 0:
        raise ValueError("the mix draws no captions the plan can bucket")
    return {resolution_id: (bucket_mass / total).tolist()
            for resolution_id, bucket_mass in raw.items()}


def write_bucket_mass(path: str, mass: Mapping[int, Sequence[float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({str(resolution_id): [float(value) for value in values]
                   for resolution_id, values in mass.items()},
                  handle, indent=2, sort_keys=True)
        handle.write("\n")


def distribution_digest(records: Sequence[DatasetLengths]) -> str:
    """A digest of the caption lengths the measurements are based on.

    Two runs with the same mix text can still draw from different lengths, so
    the cache key carries the lengths themselves rather than the file names: one
    histogram per (dataset, resolution) plus the row and caption counts.  A
    rebuilt or re-captioned dataset changes the digest, and nothing is reused.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.alias}|{record.weight!r}|{record.rows}|".encode())
        resolutions = np.unique(record.metadata.resolution_ids)
        for resolution_id in resolutions.tolist():
            lengths = lengths_for_resolution(record.metadata, int(resolution_id))
            counts = np.bincount(lengths, minlength=MAX_SEQUENCE_LENGTH + 1)[1:]
            digest.update(f"{int(resolution_id)}:".encode())
            digest.update(counts.astype(np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Combination:
    """One (resolution, bucket) pair to screen."""

    resolution_id: int
    bucket_index: int
    max_length: int
    lower_bound: int
    draw_share: float
    captions: int

    @property
    def label(self) -> str:
        return f"res{self.resolution_id}/bucket{self.bucket_index}"

    @property
    def interval(self) -> str:
        return f"({self.lower_bound}, {self.max_length}]"

    def to_json(self) -> Dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "bucket_index": self.bucket_index,
            "max_length": self.max_length,
            "lower_bound": self.lower_bound,
            "draw_share": self.draw_share,
            "captions": self.captions,
        }


def combinations(plan: Mapping[int, Sequence[Bucket]],
                 resolution_ids: Sequence[int],
                 records: Sequence[DatasetLengths]) -> List[Combination]:
    """Every bucket of every requested resolution, with its share of draws."""
    out: List[Combination] = []
    for resolution_id in resolution_ids:
        buckets = plan[int(resolution_id)]
        bounds = [bucket.max_length for bucket in buckets]
        mass, counts = bucket_draw_mass(records, int(resolution_id), bounds)
        total = float(mass.sum())
        lower = 0
        for index, bucket in enumerate(buckets):
            share = float(mass[index]) / total if total > 0 else 0.0
            out.append(Combination(
                resolution_id=int(resolution_id), bucket_index=index,
                max_length=int(bucket.max_length), lower_bound=lower,
                draw_share=share, captions=int(counts[index]),
            ))
            lower = int(bucket.max_length)
    return out
# ---------------------------------------------------------------------------
# What the sampler does for one bucket, measured on the CPU.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingProbe:
    """One pass of the sampler over a bucket, without a GPU.

    ``draws`` counts every row the sampler took from the dataset (``weight /
    rows``-weighted) while the run emitted ``emitted_samples`` samples of the
    bucket, so ``draws_per_sample`` is how much sampling work one measured
    sample costs, and ``ms_per_sample`` is that work in the units the screen
    reports.  The probe runs on the CPU only: it says nothing about GPU time,
    and everything about how much work the sink isolation adds.
    """

    emitted_batches: int
    emitted_samples: int
    draws: int
    seconds: float
    unexpected_batches: int
    truncated: bool

    @property
    def draws_per_sample(self) -> float:
        return self.draws / self.emitted_samples if self.emitted_samples else 0.0

    @property
    def ms_per_sample(self) -> float:
        return 1000.0 * self.seconds / self.emitted_samples if self.emitted_samples else 0.0


def probe_sampling(records: Sequence[DatasetLengths], plan: Mapping[int, Sequence[Bucket]],
                   combination: Combination, candidate: int, sink: int, *,
                   seed: int, policy: Any, initial_stage: float,
                   batches: int = 16, max_seconds: float = 30.0) -> SamplingProbe:
    """Run the sampler alone against the screening plan for one bucket.

    Two things the GPU runs cannot show cheaply are measured here: that the
    screening plan really emits only the bucket under test, and how many rows
    the sampler has to draw per emitted sample - the latter is the cost the
    isolation adds, which is otherwise buried in the reported per-sample time.
    The loop stops at ``batches`` emitted micro-batches or after
    ``max_seconds``; a truncated probe is the signal that the bucket is too rare
    to screen at this run length.
    """
    screened = screening_plan(plan, combination.resolution_id, combination.bucket_index,
                              candidate, sink)
    bucket_plan = BucketPlan({int(r): [LenBucket(bucket.max_length, bucket.batch_size)
                                       for bucket in buckets]
                              for r, buckets in screened.items()})
    sampler = RowLengthQueueBatchSampler(
        metadata=[record.metadata for record in records],
        bucket_plan=bucket_plan,
        dataset_weights=[record.weight for record in records],
        num_replicas=1, rank=0, shuffle=True, seed=seed,
        initial_stage=float(initial_stage), caption_policy=policy,
    )
    iterator = iter(sampler)
    target = (int(combination.resolution_id), int(combination.bucket_index))
    emitted_batches = emitted_samples = unexpected = 0
    start = time.monotonic()
    while emitted_batches < batches and time.monotonic() - start < max_seconds:
        batch = next(iterator)
        key = (int(batch[0].resolution_id), int(batch[0].len_bucket_idx))
        if key != target:
            unexpected += 1
        emitted_batches += 1
        emitted_samples += len(batch)
    seconds = time.monotonic() - start
    state = sampler.state_dict()
    queued = sum(len(queue) for queue in state["queues"].values())
    ready = sum(len(block) for block in state["ready_batches"])
    return SamplingProbe(
        emitted_batches=emitted_batches, emitted_samples=emitted_samples,
        draws=emitted_samples + ready + queued, seconds=seconds,
        unexpected_batches=unexpected, truncated=emitted_batches < batches,
    )


# ---------------------------------------------------------------------------
# Running training.
# ---------------------------------------------------------------------------

# Lines worth echoing from a run's log while it is in flight.  The full log is
# kept next to the run either way; this only keeps the screen's own output
# readable at a few dozen runs.
ECHO_PATTERNS = (
    "[throughput-summary]", "[mem-probe]", "Length-bucketed sampler plan",
    "Error", "Traceback", "OutOfMemoryError", "error:",
)


@dataclass
class RunOutcome:
    """Everything one training run produced."""

    status: str                     # "ok" | "oom" | "failed" | "timeout"
    returncode: int
    log_path: str
    command: List[str]
    summary: Dict[str, float] = field(default_factory=dict)
    mem_probe: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "log_path": self.log_path,
            "command": self.command,
            "summary": self.summary,
            "mem_probe": self.mem_probe,
            "note": self.note,
        }


def build_run_env(repo: str, probe_dir: Optional[str],
                  overrides: Mapping[str, str]) -> Dict[str, str]:
    """Environment for a training run: this repo first on the path, and the probe.

    ``PYTHONPATH`` points at the repository so ``python -m src.train.train``
    resolves from any working directory, exactly as ``gate_ab.py`` does.  The
    probe directory comes before it so the planted ``sitecustomize`` is the one
    Python imports at start-up.
    """
    env = dict(os.environ)
    parts = [part for part in (probe_dir, repo) if part]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    for key, value in DEFAULT_RUN_ENV.items():
        env.setdefault(key, value)
    env.update({str(k): str(v) for k, v in overrides.items()})
    return env


def screen_config_text(*, mix: str, plan: Mapping[int, Sequence[Bucket]],
                       tokenizer: str, vae: str, output_dir: str,
                       steps: int, warmup: int, accumulation: int) -> str:
    """The temporary TOML that turns the base recipe into one measurement.

    Only the knobs that define the measured interval are written; everything
    else - model shape, optimizer, caption policy, data loading - stays exactly
    as the base config defines it, because the screen is meant to measure the
    configuration training will use.

    ``checkpoint_interval`` and ``eval_interval`` are pushed past ``max_steps``
    so no checkpoint or image grid interrupts a short run; the fixed-sample
    loss probe is switched off for the same reason.  The throughput summary
    already excludes evaluation and checkpoint time, but not paying for them at
    all keeps a 30-step run short.
    """
    return (
        "[data]\n"
        f"mix = {json.dumps(mix)}\n"
        f"bucket_plan = {json.dumps(plan_spec(plan))}\n"
        "[text_encoder]\n"
        f"path = {json.dumps(tokenizer)}\n"
        "[paths]\n"
        f"vae = {json.dumps(vae)}\n"
        f"output_dir = {json.dumps(output_dir)}\n"
        "[train]\n"
        f"max_steps = {int(steps)}\n"
        f"steady_state_skip_steps = {int(warmup)}\n"
        f"gradient_accumulation_steps = {int(accumulation)}\n"
        f"checkpoint_interval = {int(steps) + 1}\n"
        f"eval_interval = {int(steps) + 1}\n"
        "[eval]\n"
        "loss_interval = 0\n"
        "kid_at_end = false\n"
        "compute_metrics = false\n"
    )


def build_run_command(configs: Sequence[str], run_name: str,
                      trainer_args: Sequence[str]) -> List[str]:
    """The training command, in the form ``gate_ab.py`` uses.

    Later ``--config`` files override earlier ones, and this script's temporary
    file is always last, so the measurement settings win over the recipe while
    the recipe supplies everything the measurement does not mention.
    """
    command = [sys.executable, "-m", "src.train.train"]
    for path in configs:
        command += ["--config", str(path)]
    command += ["--run_name", run_name]
    command += [str(arg) for arg in trainer_args]
    return command


def execute(command: Sequence[str], log_path: str, env: Mapping[str, str],
            timeout: float, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run one training process, streaming its log to a file.

    The process is killed at ``timeout``: a bucket that cannot fill a
    micro-batch leaves the sampler drawing forever, and a screen must record
    that instead of hanging on it.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(list(command), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               env=dict(env), cwd=cwd)
    killed = {"flag": False}

    def _kill():
        killed["flag"] = True
        process.kill()

    timer = threading.Timer(float(timeout), _kill)
    timer.start()
    lines: List[str] = []
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                lines.append(line)
                if any(pattern in line for pattern in ECHO_PATTERNS):
                    print("    " + line.rstrip(), flush=True)
    finally:
        timer.cancel()
        process.wait()
    return {
        "returncode": int(process.returncode),
        "timed_out": bool(killed["flag"]),
        "output": "".join(lines),
        "log_path": str(log_path),
    }


def parse_throughput_summary(output: str) -> Dict[str, float]:
    """The last ``[throughput-summary]`` line as a number dictionary."""
    fields: Dict[str, float] = {}
    for line in output.splitlines():
        if "[throughput-summary]" in line:
            fields = {name: float(value)
                      for name, value in re.findall(r"(\w+)=([\d.]+)", line)}
    return fields


def parse_memory_probe(output: str) -> Dict[str, float]:
    """The largest peaks the in-process probe reported, in GB.

    The trainer resets its peaks once per optimizer step, so every probe line
    describes a window inside one step and the run's figure is the largest of
    them.
    """
    peaks: Dict[str, float] = {}
    for line in output.splitlines():
        if "[mem-probe]" not in line:
            continue
        for name, value in re.findall(r"(\w+)=([\d.]+)", line):
            number = float(value)
            if name not in peaks or number > peaks[name]:
                peaks[name] = number
    return peaks


def detect_out_of_memory(output: str) -> bool:
    return bool(re.search(r"OutOfMemoryError|CUDA out of memory", output))
# ---------------------------------------------------------------------------
# One run's record, and what the candidates of a bucket say together.
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """The measurement one training run contributed, or why it contributed none."""

    resolution_id: int
    bucket_index: int
    batch_size: int
    repeat: int
    status: str                       # "ok" | "oom" | "failed" | "timeout"
    isolated: bool
    steps: int = 0
    steady_steps: int = 0
    samples: int = 0
    samples_per_step: Optional[float] = None
    ms_per_sample: Optional[float] = None       # steady: the selection metric
    ms_per_sample_all: Optional[float] = None   # includes compile and start-up
    peak_allocated_gb: Optional[float] = None
    peak_reserved_gb: Optional[float] = None
    log_path: str = ""
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RunRecord":
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def record_from_outcome(combination: Combination, batch_size: int, repeat: int,
                        outcome: RunOutcome, accumulation: int) -> RunRecord:
    """Turn one finished process into a record, checking the isolation.

    ``samples_per_step`` is the run's own report of how many samples each
    optimizer step carried.  Every micro-batch of a screening run belongs to the
    bucket under test and has the candidate size, so the number must equal
    ``accumulation * batch_size``; anything else means a sink bucket emitted, and
    the run measured a mix rather than the bucket.
    """
    record = RunRecord(
        resolution_id=int(combination.resolution_id),
        bucket_index=int(combination.bucket_index),
        batch_size=int(batch_size), repeat=int(repeat),
        status=outcome.status, isolated=False,
        log_path=outcome.log_path, note=outcome.note,
    )
    summary = outcome.summary
    if summary:
        record.steps = int(summary.get("steps", 0))
        record.steady_steps = int(summary.get("steady_steps", 0))
        record.samples = int(summary.get("samples", 0))
        record.samples_per_step = summary.get("samples_per_step")
        if summary.get("samples_per_sec", 0.0) > 0:
            record.ms_per_sample_all = 1000.0 / summary["samples_per_sec"]
        if summary.get("samples_per_sec_steady", 0.0) > 0:
            record.ms_per_sample = 1000.0 / summary["samples_per_sec_steady"]
        record.peak_allocated_gb = summary.get("peak_mem_gb")
        # The trainer reports the allocator's own ceiling (reserved) as well as
        # what it handed out (allocated); a batch-size decision turns on the
        # first, so it is preferred over any outside probe.
        record.peak_reserved_gb = summary.get("peak_mem_reserved_gb")
    if outcome.mem_probe:
        if record.peak_reserved_gb is None:
            record.peak_reserved_gb = outcome.mem_probe.get("peak_reserved_gb")
        if record.peak_allocated_gb is None:
            record.peak_allocated_gb = outcome.mem_probe.get("peak_allocated_gb")
    expected = float(accumulation) * float(batch_size)
    if record.samples_per_step is not None:
        record.isolated = abs(record.samples_per_step - expected) <= max(0.55, 1e-3 * expected)
    if record.status == "ok" and not record.isolated and outcome.summary:
        record.status = "missed"
        record.note = (record.note + " " + (
            f"samples_per_step={record.samples_per_step} but the bucket's micro-batches "
            f"carry {expected:.1f}: another bucket emitted, so this run measured a mix"
        )).strip()
    return record


@dataclass
class CandidateStats:
    """What the repeats of one candidate size say."""

    batch_size: int
    runs: int
    ok_runs: int
    status: str         # "ok" only when every repeat was ok, else the first status seen
    ms_per_sample: Optional[float] = None
    spread_ms: Optional[float] = None
    ms_per_sample_all: Optional[float] = None
    peak_allocated_gb: Optional[float] = None
    peak_reserved_gb: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}


def candidate_stats(records: Sequence[RunRecord],
                    candidates: Sequence[int]) -> List[CandidateStats]:
    """Per-candidate medians, spread, and peak memory over the repeats.

    The median is the reported per-sample time and the spread is the distance
    between the repeats, which is what says whether two candidates really
    differ.  Memory is the largest peak any repeat reached, since the choice has
    to be safe for the worst repeat rather than the average one.
    """
    out: List[CandidateStats] = []
    for candidate in candidates:
        runs = [record for record in records if record.batch_size == int(candidate)]
        usable = [record for record in runs
                  if record.status == "ok" and record.ms_per_sample is not None]
        statuses = [record.status for record in runs]
        stats = CandidateStats(
            batch_size=int(candidate), runs=len(runs), ok_runs=len(usable),
            status=("ok" if usable and all(s == "ok" for s in statuses)
                    else (statuses[0] if statuses else "not run")),
        )
        if usable:
            times = [float(record.ms_per_sample) for record in usable]
            stats.ms_per_sample = float(statistics.median(times))
            stats.spread_ms = float(max(times) - min(times)) if len(times) > 1 else 0.0
            all_times = [record.ms_per_sample_all for record in usable
                         if record.ms_per_sample_all is not None]
            if all_times:
                stats.ms_per_sample_all = float(statistics.median(all_times))
            allocated = [record.peak_allocated_gb for record in runs
                         if record.peak_allocated_gb is not None]
            reserved = [record.peak_reserved_gb for record in runs
                        if record.peak_reserved_gb is not None]
            stats.peak_allocated_gb = float(max(allocated)) if allocated else None
            stats.peak_reserved_gb = float(max(reserved)) if reserved else None
        out.append(stats)
    return out


@dataclass
class Selection:
    """The size written for one bucket, and the evidence behind it."""

    combination: Combination
    batch_size: int
    source: str                     # "measured" | "fallback"
    reason: str
    candidate: Optional[CandidateStats] = None
    runner_up: Optional[CandidateStats] = None
    at_edge: bool = False
    margin_gb: Optional[float] = None
    margin_fraction: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.combination.to_json(),
            "batch_size": self.batch_size,
            "source": self.source,
            "reason": self.reason,
            "at_edge": self.at_edge,
            "ms_per_sample": self.candidate.ms_per_sample if self.candidate else None,
            "spread_ms": self.candidate.spread_ms if self.candidate else None,
            "ms_per_sample_all": self.candidate.ms_per_sample_all if self.candidate else None,
            "runner_up_batch_size": self.runner_up.batch_size if self.runner_up else None,
            "runner_up_ms_per_sample": self.runner_up.ms_per_sample if self.runner_up else None,
            "peak_allocated_gb": self.candidate.peak_allocated_gb if self.candidate else None,
            "peak_reserved_gb": self.candidate.peak_reserved_gb if self.candidate else None,
            "margin_gb": self.margin_gb,
            "margin_fraction": self.margin_fraction,
        }


def choose_batch_size(stats: Sequence[CandidateStats]
                      ) -> Tuple[Optional[int], str, Optional[CandidateStats],
                                 Optional[CandidateStats], bool]:
    """Pick the candidate with the lowest per-sample time, and explain it.

    The rule is the measured minimum, not the largest size that fit.  Two
    refinements follow from the spread: a candidate whose time is within the
    noise of a smaller candidate is not distinguishable from it, and the
    smaller one wins because it needs less memory; and a minimum that sits at
    the edge of the candidate list is reported as such, because the true
    optimum may lie outside the range that was screened.
    """
    feasible = [item for item in stats
                if item.ok_runs > 0 and item.ms_per_sample is not None]
    if not feasible:
        return None, "no candidate completed a run", None, None, False
    best = min(feasible, key=lambda item: (item.ms_per_sample, item.batch_size))
    noise = max(item.spread_ms or 0.0 for item in feasible)
    chosen = min((item for item in feasible
                  if item.ms_per_sample <= best.ms_per_sample + noise),
                 key=lambda item: item.batch_size)
    # A minimum at either end of what could be measured is an extrapolation: the
    # optimum may lie outside the candidates.  A larger candidate that was tried
    # and failed is not that case - its failure is reported as the reason.
    usable = sorted(item.batch_size for item in feasible)
    ran = [item for item in stats if item.runs > 0]
    blocked_above = any(item.batch_size > best.batch_size and item.ok_runs == 0
                        for item in ran)
    at_edge = best.batch_size == usable[0] or (
        best.batch_size == usable[-1] and not blocked_above)
    if chosen.batch_size == best.batch_size:
        reason = (f"lowest per-sample time ({best.ms_per_sample:.3f} ms) of the "
                  "screened candidates")
    else:
        reason = (f"within the measurement spread of the smallest time "
                  f"({best.ms_per_sample:.3f} ms at batch {best.batch_size}); "
                  "the smaller size needs less memory")
    if at_edge:
        reason += (f"; the minimum sits at the edge of the candidates that produced a "
                   f"measurement ({best.batch_size}), so the optimum may lie outside "
                   "the screened range")
    return chosen.batch_size, reason, chosen, _runner_up(feasible, best), at_edge


def _runner_up(feasible: Sequence[CandidateStats],
               best: CandidateStats) -> Optional[CandidateStats]:
    rest = [item for item in feasible if item.batch_size != best.batch_size]
    if not rest:
        return None
    return min(rest, key=lambda item: (item.ms_per_sample, item.batch_size))


# ---------------------------------------------------------------------------
# Cache.
# ---------------------------------------------------------------------------


def settings_digest(settings: Mapping[str, Any]) -> str:
    """A short, stable digest of everything a measurement depends on."""
    payload = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_cache(path: str) -> Dict[str, Any]:
    if not Path(path).is_file():
        return {"version": CACHE_VERSION, "entries": {}}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("version", -1)) != CACHE_VERSION:
        return {"version": CACHE_VERSION, "entries": {}}
    payload.setdefault("entries", {})
    return payload


def store_cache(path: str, cache: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")


def cache_records(cache: Mapping[str, Any], key: str) -> List[RunRecord]:
    entry = cache.get("entries", {}).get(key, {})
    return [RunRecord.from_json(payload) for payload in entry.get("records", [])]


def cache_append(cache: Dict[str, Any], key: str, settings: Mapping[str, Any],
                 records: Sequence[RunRecord]) -> None:
    entry = cache.setdefault("entries", {}).setdefault(key, {})
    entry["settings"] = json.loads(json.dumps(settings, default=str))
    stored = entry.setdefault("records", [])
    for record in records:
        stored.append(record.to_json())
# ---------------------------------------------------------------------------
# Settings, device identity, and the cache key.
# ---------------------------------------------------------------------------


def read_config_sections(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Merge the TOML sections this screen reads; later files override earlier.

    The same order the training configuration loader applies, so the values
    that end up in the cache key are the values the runs will use.
    """
    import tomllib

    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, "rb") as handle:
            payload = tomllib.load(handle)
        for section, values in payload.items():
            if isinstance(values, Mapping):
                merged.setdefault(section, {}).update(values)
    return merged


def caption_policy_from(sections: Mapping[str, Mapping[str, Any]]) -> Any:
    """The caption policy the runs will use, from the data section."""
    from src.dataset.captions import CaptionPolicy

    data = sections.get("data", {})
    defaults = CaptionPolicy()
    return CaptionPolicy(
        kind=str(data.get("caption_policy", "legacy")),
        beta_start=float(data.get("caption_beta_start", defaults.beta_start)),
        beta_end=float(data.get("caption_beta_end", defaults.beta_end)),
        schedule=str(data.get("caption_schedule", defaults.schedule)),
        early_at=float(data.get("caption_early_at", defaults.early_at)),
        short_reserve=float(data.get("caption_short_reserve", defaults.short_reserve)),
        short_threshold=int(data.get("caption_short_threshold", defaults.short_threshold)),
    )


def detect_device() -> Tuple[str, float]:
    """The GPU this screen is running on: its name and its total memory."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "no CUDA device is visible; the screen has to run on the machine "
            "that will train, or the device has to be named with --gpu and "
            "--device-memory-gb"
        )
    properties = torch.cuda.get_device_properties(0)
    return str(torch.cuda.get_device_name(0)), float(properties.total_memory / 1024 ** 3)


def resolve_device(gpu: Optional[str], memory_gb: Optional[float]) -> Tuple[str, float]:
    """Device identity, from the command line when given and from CUDA otherwise."""
    if gpu is not None and memory_gb is not None:
        return str(gpu), float(memory_gb)
    detected_name, detected_memory = detect_device()
    return str(gpu or detected_name), float(memory_gb or detected_memory)


def software_versions() -> Dict[str, Any]:
    import torch

    versions: Dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": str(torch.__version__),
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version(),
    }
    try:
        import accelerate

        versions["accelerate"] = str(accelerate.__version__)
    except ImportError:  # pragma: no cover - accelerate ships with the trainer
        versions["accelerate"] = None
    return versions


def build_settings(*, sections: Mapping[str, Mapping[str, Any]], gpu: str,
                   device_memory_gb: float, versions: Mapping[str, Any],
                   steps: int, warmup: int, accumulation: int, candidates: Sequence[int],
                   trainer_args: Sequence[str], run_env: Mapping[str, str], seed: int,
                   mix: str, data_digest: str, bounds: Mapping[int, Sequence[int]],
                   device_count: int) -> Dict[str, Any]:
    """Everything a measured per-sample time depends on, for the cache key.

    The key has to separate measurements that are not interchangeable: a
    different model or GPU obviously is not the same number, but neither is a
    different compile flag, a different allocator setting, a different number of
    optimizer steps, or a different caption-length distribution behind the same
    dataset paths.  All of them are in here.
    """
    return {
        "profile": "batch-size-screen",
        "version": CACHE_VERSION,
        "model": {
            **{key: sections.get("model", {}).get(key) for key in (
                "hidden_size", "num_heads", "double_stream_depth",
                "single_stream_depth", "mlp_ratio", "conditioning_scheme",
                "double_stream_modulation", "single_stream_modulation",
                "ffn_type", "qkv_bias", "rope_centered_grid")},
            "text_encoder": sections.get("text_encoder", {}).get("path"),
            "text_encoder_exit_layer": sections.get("text_encoder", {}).get("exit_layer"),
            "vae": sections.get("paths", {}).get("vae"),
        },
        "gpu": {"name": gpu, "total_memory_gb": round(float(device_memory_gb), 3),
                "count": int(device_count)},
        "software": dict(versions),
        "execution": {
            "steps": int(steps),
            "warmup_steps": int(warmup),
            "gradient_accumulation_steps": int(accumulation),
            "candidates": [int(value) for value in candidates],
            "trainer_args": [str(arg) for arg in trainer_args],
            "seed": int(seed),
            "env": {str(key): str(run_env[key])
                    for key in ENV_KEYS_IN_KEY if key in run_env},
        },
        "distribution": {
            "mix": mix,
            "data_digest": data_digest,
            "bucket_bounds": {str(r): [int(bound) for bound in bounds[r]]
                              for r in sorted(bounds)},
        },
    }


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


@dataclass
class ReportContext:
    """Everything the report states, gathered while the screen runs."""

    argv: Sequence[str]
    plan_path: str
    out_path: str
    measurements_path: str
    cache_path: str
    run_dir: str
    mix: str
    entries: Sequence[DatasetEntry]
    base_config: str
    extra_configs: Sequence[str]
    trainer_args: Sequence[str]
    vae: str
    text_encoder: str
    gpu: str
    device_memory_gb: float
    candidates: Sequence[int]
    steps: int
    warmup: int
    repeats: int
    accumulation: int
    sink: int
    min_caption_share: float
    min_margin_fraction: float
    cache_key: str
    settings: Mapping[str, Any]
    selections: Sequence[Selection]
    unscreened: Sequence[Tuple[Combination, str]]
    stats: Mapping[Tuple[int, int], Sequence[CandidateStats]]
    probes: Mapping[Tuple[int, int], SamplingProbe]
    records: Sequence[RunRecord]
    resolutions: Sequence[int]
    kept_resolutions: Mapping[int, Sequence[int]]
    fallback_size: int
    warnings: Sequence[str]
    draw_probe: bool


def _table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    lines = ["| " + " | ".join(str(cell) for cell in header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _number(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(context: ReportContext) -> str:
    """The human-readable record of what was screened and what was not."""
    lines: List[str] = []
    lines.append("# Micro-batch size screen")
    lines.append("")
    lines.append("Per-(resolution, caption-length bucket) micro-batch sizes, measured on one "
                 "GPU with the training configuration, for a bucket plan whose boundaries "
                 "were fixed beforehand.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- command: `python -m scripts.bench.batch_size_screen "
                 f"{shlex.join([str(arg) for arg in context.argv])}`")
    lines.append(f"- plan, boundaries fixed: `{context.plan_path}`; sizes written to "
                 f"`{context.out_path}`, measurements to `{context.measurements_path}`")
    lines.append(f"- data mix: `{context.mix}`")
    for entry in context.entries:
        lines.append(f"  - `{entry.path}` weight {entry.weight:.4f}")
    lines.append(f"- resolutions screened: {', '.join(str(r) for r in context.resolutions)}")
    lines.append(f"- candidate batch sizes: {', '.join(str(c) for c in context.candidates)}")
    lines.append(f"- steps per run: {context.steps} optimizer steps; the first "
                 f"{context.warmup} are dropped from the steady figure; "
                 f"gradient_accumulation_steps {context.accumulation}; "
                 f"{context.repeats} run(s) per candidate")
    lines.append(f"- base config `{context.base_config}`"
                 + (f", then {', '.join(context.extra_configs)}" if context.extra_configs else "")
                 + (f", then `--trainer-arg {' '.join(context.trainer_args)}`"
                    if context.trainer_args else ""))
    lines.append(f"- vae `{context.vae}`, text encoder `{context.text_encoder}`")
    lines.append(f"- GPU: {context.gpu}, {context.device_memory_gb:.1f} GB total")
    environment = context.settings.get("execution", {}).get("env", {})
    if environment:
        lines.append("- run environment: "
                     + ", ".join(f"`{key}={value}`" for key, value in environment.items()))
    lines.append(f"- cache: `{context.cache_path}` keyed by `{context.cache_key}`")
    lines.append(f"- run logs and their temporary configs: `{context.run_dir}`")
    lines.append("")
    lines.append("## How one run measures one bucket")
    lines.append("")
    lines.append("The sampler assigns a drawn caption to the first bucket whose bound covers "
                 "its retained length and emits a micro-batch only once that bucket's queue "
                 "holds the bucket's own batch size, so a run of the real plan mixes every "
                 "bucket and the trainer's throughput summary averages over that mix. Each "
                 "screening run therefore uses a plan in which only the bucket under test has "
                 "a real size: every other bucket gets a sink size of " + f"{context.sink}"
                 + ", far above the number of draws the run can make, so those queues never "
                 "fill and never emit. Every micro-batch that reaches the GPU is then a "
                 "micro-batch of the bucket under test, with its padded width and its own "
                 "caption lengths, and the run's per-sample time is that bucket's.")
    lines.append("")
    lines.append("The timer is the trainer's own: it covers input preparation, text encoding, "
                 "DiT forward and backward, the optimizer, EMA and logging, and excludes only "
                 "evaluation and checkpoint work. Two figures are reported per run - the "
                 "average over all steps, which carries the one-time `torch.compile` stall, "
                 "and the steady average after the first "
                 f"{context.warmup} steps, which is what the selection uses. Peak memory is "
                 "read after the optimizer state exists. Peak allocated comes from the "
                 "trainer's summary; peak reserved is sampled in-process by a small "
                 "`sitecustomize` probe, because the summary does not report it. A candidate "
                 "that runs out of memory is recorded and skipped, and the scan continues.")
    lines.append("")
    lines.append("The sampler still draws and discards the other buckets' rows, so a run does "
                 "more sampling work than training would. The run's log records how much: "
                 "`samples_per_step` must equal "
                 "`gradient_accumulation_steps * batch_size`, which only holds when nothing "
                 "else emitted, and any run that fails the check is recorded as `missed` and "
                 "excluded. The discarded draws cost CPU time inside the measured step; the "
                 "size of that cost is measured per bucket by a CPU-only probe and is printed "
                 "below, and it is the same for every candidate of a bucket, so it does not "
                 "decide which candidate wins.")
    lines.append("")
    lines.append("## Selections")
    lines.append("")
    rows = []
    for selection in context.selections:
        combination = selection.combination
        rows.append([
            combination.resolution_id, combination.bucket_index, combination.interval,
            f"{combination.draw_share:.3%}", combination.captions,
            selection.batch_size,
            _number(selection.candidate.ms_per_sample if selection.candidate else None),
            _number(selection.candidate.spread_ms if selection.candidate else None),
            _number(selection.candidate.ms_per_sample_all if selection.candidate else None),
            selection.source,
        ])
    lines.append(_table(rows, ["res", "bucket", "interval", "draw share", "captions",
                               "batch size", "ms/sample", "spread ms", "ms/sample all steps",
                               "source"]))
    lines.append("")
    lines.append("`source` is `measured` for every row of this table: the buckets whose "
                 "sizes are not measurements are listed under 'Not screened' instead, with "
                 "the fallback they carry.")
    lines.append("")
    lines.append("`ms/sample` is the steady per-sample time of the chosen candidate (the "
                 "median over its repeats); `spread ms` is how far its repeats ranged; "
                 "`ms/sample all steps` includes compile and start-up and is shown so the "
                 "warm-up cost is visible rather than hidden.")
    lines.append("")
    lines.append("### Why each size")
    lines.append("")
    for selection in context.selections:
        lines.append(f"- {selection.combination.label} ({selection.combination.interval}): "
                     f"batch {selection.batch_size} - {selection.reason}")
    lines.append("")
    lines.append("## Memory")
    lines.append("")
    rows = []
    for selection in context.selections:
        candidate = selection.candidate
        rows.append([
            selection.combination.label, selection.batch_size,
            _number(candidate.peak_allocated_gb if candidate else None, 1),
            _number(candidate.peak_reserved_gb if candidate else None, 1),
            _number(selection.margin_gb, 1),
            ("n/a" if selection.margin_fraction is None
             else f"{selection.margin_fraction:.1%}"),
        ])
    lines.append(_table(rows, ["bucket", "batch size", "peak allocated GB",
                               "peak reserved GB", "margin GB", "margin of device"]))
    lines.append("")
    lines.append(f"The margin is the device's total memory minus the peak reserved by that "
                 f"candidate (or minus peak allocated where the reserved probe did not "
                 f"report). A margin below {context.min_margin_fraction:.0%} of the device is "
                 f"flagged here:")
    flagged = [selection for selection in context.selections
               if selection.margin_fraction is not None
               and selection.margin_fraction < context.min_margin_fraction]
    if flagged:
        for selection in flagged:
            lines.append(f"- {selection.combination.label} at batch {selection.batch_size}: "
                         f"margin {selection.margin_fraction:.1%}")
    else:
        lines.append("- none: every chosen size leaves at least that much headroom.")
    lines.append("")
    lines.append("## Measurements")
    lines.append("")
    lines.append("Every candidate of every screened bucket, including the ones that did not "
                 "win. OOM and failed runs are listed with their status.")
    lines.append("")
    for resolution_id in context.resolutions:
        lines.append(f"### resolution {resolution_id}")
        lines.append("")
        rows = []
        for (res, bucket_index), stats_list in sorted(context.stats.items()):
            if res != resolution_id:
                continue
            for stats in stats_list:
                rows.append([
                    bucket_index, stats.batch_size, stats.runs, stats.ok_runs,
                    _number(stats.ms_per_sample), _number(stats.spread_ms),
                    _number(stats.ms_per_sample_all),
                    _number(stats.peak_allocated_gb, 1), _number(stats.peak_reserved_gb, 1),
                    stats.status,
                ])
        lines.append(_table(rows, ["bucket", "batch size", "runs", "usable", "ms/sample",
                                   "spread ms", "ms/sample all", "peak alloc GB",
                                   "peak reserv GB", "status"]))
        lines.append("")
    if context.draw_probe:
        lines.append("### Sampling cost of the isolation")
        lines.append("")
        lines.append("Measured without a GPU, by running the sampler alone against the same "
                     "screening plan: how many rows one emitted sample costs, and that work "
                     "in milliseconds per sample. It is the part of the measured per-sample "
                     "time that the discarded rows add, and it is the same for every "
                     "candidate of a bucket, so it does not decide which candidate wins. It "
                     "is the whole cost only where the sampler is not overlapped with GPU "
                     "work, which is why it is reported as an estimate rather than "
                     "subtracted from the measurements.")
        lines.append("")
        rows = []
        for (resolution_id, bucket_index), probe in sorted(context.probes.items()):
            rows.append([f"res{resolution_id}/bucket{bucket_index}", probe.emitted_batches,
                         probe.emitted_samples, probe.draws,
                         _number(probe.draws_per_sample, 1),
                         _number(probe.ms_per_sample, 3)])
        lines.append(_table(rows, ["bucket", "probe batches", "probe samples", "draws",
                                   "draws per sample", "ms/sample"]))
        lines.append("")
    lines.append("## Not screened")
    lines.append("")
    lines.append("These buckets were not measured, so the plan carries the declared fallback "
                 f"size {context.fallback_size} for them. Nothing here silently keeps the "
                 "placeholder the plan arrived with, and the fallback is a declared choice "
                 "rather than a measurement: it is the smallest candidate, which is "
                 "conservative for memory and not for throughput.")
    lines.append("")
    if context.unscreened:
        rows = []
        for combination, why in context.unscreened:
            rows.append([combination.resolution_id, combination.bucket_index,
                         combination.interval, f"{combination.draw_share:.3%}",
                         combination.captions, context.fallback_size, why])
        lines.append(_table(rows, ["res", "bucket", "interval", "draw share", "captions",
                                   "fallback", "why"]))
    else:
        lines.append("None: every bucket of every screened resolution was measured.")
    if context.kept_resolutions:
        lines.append("")
        lines.append("The plan also covers resolutions this screen was not asked for; their "
                     "sizes are neither measured nor replaced, and stay exactly as the plan "
                     "had them:")
        lines.append("")
        rows = [[resolution_id, ", ".join(str(size) for size in sizes)]
                for resolution_id, sizes in sorted(context.kept_resolutions.items())]
        lines.append(_table(rows, ["res", "batch sizes kept from the plan"]))
    lines.append("")
    lines.append("## Cache")
    lines.append("")
    lines.append(f"Measurements live in `{context.cache_path}` under key "
                 f"`{context.cache_key}`. The key covers:")
    lines.append("")
    lines.append("- the model recipe (width, depth, modulation, encoder and its exit layer, "
                 "vae) as the merged configs define it")
    lines.append("- the GPU name, its total memory and the device count")
    lines.append("- the software versions (python, torch, CUDA, cuDNN, accelerate)")
    lines.append("- the execution settings: steps, warm-up, accumulation, candidates, "
                 "`--trainer-arg` flags, seed, and the run environment")
    lines.append("- the caption-length distribution: the mix text, a digest of the loaded "
                 "caption lengths per dataset and resolution, and the bucket bounds")
    lines.append("")
    lines.append("A different setting therefore produces a different key and never reuses "
                 "another key's runs.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- One GPU, and the numbers are that GPU's. The cache key separates GPUs, so "
                 "a measurement is never silently moved to another device.")
    lines.append("- Each bucket is measured in its own process with its own `torch.compile` "
                 "warm-up; the compile cost is excluded from the steady figure and reported "
                 "separately, but a real run compiles the same shapes once rather than once "
                 "per bucket.")
    lines.append("- A short run sweeps the caption curriculum from start to end in its first "
                 "few steps, so the caption lengths inside a bucket come from a small sample "
                 "of rows. The bucket's padded width, which dominates the DiT cost, is exact; "
                 "the text encoder sees the lengths the sampler drew.")
    lines.append("- Bucket boundaries are inputs and are not re-solved here, even where the "
                 "screen shows a boundary would pay for itself.")
    lines.append("- Dropped rows are sampler work, not data work: they are drawn from the "
                 "sidecar, never fetched or decoded, and never reach the model.")
    lines.append("")
    for warning in context.warnings:
        lines.append(f"> warning: {warning}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The screen itself.
# ---------------------------------------------------------------------------


def default_output_paths(plan_path: str, out: Optional[str], report: Optional[str],
                         measurements: Optional[str], cache: Optional[str],
                         run_dir: Optional[str]) -> Dict[str, str]:
    if out is None:
        plan = Path(plan_path)
        out = str(plan.with_name(plan.stem + ".screened.json"))
    return {
        "out": out,
        "report": report or f"{out}.report.md",
        "measurements": measurements or f"{out}.measurements.json",
        "cache": cache or f"{out}.cache.json",
        "run_dir": run_dir or f"{out}.runs",
    }


def write_probe(directory: str) -> str:
    """Plant the memory probe where the training process will import it."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "sitecustomize.py").write_text(MEM_PROBE_SOURCE, encoding="utf-8")
    return str(path)


def probe_conflicts(entries: Sequence[str]) -> List[str]:
    """Existing ``sitecustomize`` modules the probe directory would shadow.

    Python imports the first ``sitecustomize`` it finds on the path, so planting
    ours in front of someone else's would silently replace it.  That is worth an
    error rather than a surprise in a training process, and ``--no-mem-probe``
    is the way out: the screen then runs without sampling reserved memory.
    """
    return [str(Path(entry) / "sitecustomize.py") for entry in entries
            if entry and (Path(entry) / "sitecustomize.py").is_file()]


def run_training_run(*, python: str, configs: Sequence[str], run_name: str, run_dir: str,
                     mix: str, plan: Mapping[int, Sequence[Bucket]], tokenizer: str,
                     vae: str, steps: int, warmup: int, accumulation: int,
                     trainer_args: Sequence[str], env_overrides: Mapping[str, str],
                     probe_dir: Optional[str], timeout: float, repo: str) -> RunOutcome:
    """One training run: temporary config, process, log, parsed result."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "screen.toml"
    config_path.write_text(
        screen_config_text(mix=mix, plan=plan, tokenizer=tokenizer, vae=vae,
                           output_dir=str(directory), steps=steps, warmup=warmup,
                           accumulation=accumulation),
        encoding="utf-8",
    )
    log_path = str(directory / "run.log")
    command = build_run_command([*configs, str(config_path)], run_name, trainer_args)
    command[0] = python
    env = build_run_env(repo, probe_dir, env_overrides)
    print(f"  running {run_name}: {' '.join(shlex.quote(part) for part in command)}",
          flush=True)
    result = execute(command, log_path, env, timeout=timeout, cwd=repo)
    outcome = RunOutcome(
        status="failed", returncode=result["returncode"], log_path=result["log_path"],
        command=command,
        summary=parse_throughput_summary(result["output"]),
        mem_probe=parse_memory_probe(result["output"]),
    )
    if result["timed_out"]:
        outcome.status = "timeout"
        outcome.note = f"the run was killed after {timeout:.0f}s"
    elif detect_out_of_memory(result["output"]):
        outcome.status = "oom"
        outcome.note = "CUDA out of memory"
    elif result["returncode"] == 0 and outcome.summary:
        outcome.status = "ok"
    elif result["returncode"] != 0:
        tail = [line for line in result["output"].strip().splitlines()[-6:]]
        outcome.note = "exit code %d; last output: %s" % (
            result["returncode"], " / ".join(line.strip() for line in tail))
    return outcome


def screen(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run the whole screen and write the plan, the measurements and the report."""
    repo = str(Path(__file__).resolve().parents[2])
    paths = default_output_paths(args.plan, args.out, args.report, args.measurements,
                                 args.cache, args.run_dir)
    configs = [args.base_config, *args.config]
    sections = read_config_sections(configs)
    data_section = sections.get("data", {})
    train_section = sections.get("train", {})

    candidates = parse_int_list(args.batch_sizes, "--batch-sizes")
    bucket_sizes = parse_bucket_batch_sizes(args.bucket_batch_sizes,
                                            "--bucket-batch-sizes")
    all_sizes = sorted({*candidates,
                        *(size for sizes in bucket_sizes.values() for size in sizes)})
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if not 0 <= args.warmup < args.steps:
        raise ValueError(
            f"--warmup {args.warmup} must be below --steps {args.steps}, otherwise the "
            "steady figure has no steps in it"
        )
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    accumulation = int(args.accumulation or train_section.get("gradient_accumulation_steps", 16))
    if accumulation < 1:
        raise ValueError("--accumulation must be at least 1")
    seed = int(args.seed if args.seed is not None else train_section.get("seed", 42))
    curriculum_start = float(data_section.get("curriculum_start", 0.0))
    policy = caption_policy_from(sections)

    plan = read_plan(args.plan)
    entries = parse_dataset_mix(args.mix)
    records = load_dataset_lengths(entries, args.text_encoder)
    data_resolutions = sorted({int(resolution) for record in records
                              for resolution in np.unique(record.metadata.resolution_ids)})
    if not data_resolutions:
        raise ValueError("the mix has no resolution ids at all")
    missing = [resolution for resolution in data_resolutions if resolution not in plan]
    if missing:
        raise ValueError(
            f"the plan has no buckets for resolution ids {missing} that the mix contains; "
            "training would refuse the same plan"
        )
    if args.resolutions is None:
        requested = data_resolutions
    else:
        requested = parse_int_list(args.resolutions, "--resolutions")
        unknown = sorted(set(requested) - set(data_resolutions))
        if unknown:
            raise ValueError(f"the mix contains no resolution ids {unknown}")
    # The trainer's own loader is the contract: a plan it refuses is not one
    # training would accept, so refuse it here rather than in the first run.
    load_plan_for_trainer(args.plan, data_resolutions)

    # The draw mass the merge step aligns the buckets with needs no GPU, so
    # writing it is a mode of its own: the caller gets the file without a
    # screen run.  It covers every resolution the mix has, not only those
    # --resolutions names, because the merge needs the whole distribution.
    if args.bucket_mass_out:
        mass_resolutions = [resolution_id for resolution_id in sorted(plan)
                            if resolution_id in set(data_resolutions)]
        write_bucket_mass(str(args.bucket_mass_out),
                          bucket_draw_mass_by_resolution(plan, mass_resolutions, records))
        unwritten = sorted(set(plan) - set(mass_resolutions))
        if unwritten:
            print(f"note: the mix carries no rows for resolution ids {unwritten}, so "
                  "the bucket mass leaves them out")
        print(f"wrote bucket mass: {args.bucket_mass_out}")
        return 0
    if not args.vae:
        raise ValueError("--vae is required to screen; it is optional only with "
                         "--bucket-mass-out, which stops before any run")

    combos = combinations(plan, requested, records)
    sink = sink_batch_size(args.steps, accumulation, all_sizes)
    load_plan_for_trainer(plan_spec(screening_plan(plan, requested[0], 0,
                                                   max(all_sizes), sink)),
                          data_resolutions)
    for combo in combos:
        if combo.draw_share >= args.min_caption_share and combo.captions == 0:
            raise ValueError(
                f"{combo.label}: the plan's bound {combo.max_length} sits below every "
                "caption of its resolution, so the bucket can never fill and the run "
                "would draw forever"
            )

    gpu, device_memory_gb = resolve_device(args.gpu, args.device_memory_gb)
    versions = software_versions()
    device_count = 0
    if args.device_count is None:
        import torch

        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    else:
        device_count = int(args.device_count)
    run_env_overrides = parse_key_values(args.env, "--env")
    # The key carries the knobs as the runs will really see them, whichever side
    # set them: an allocator setting changes both time and reserved memory.
    effective_env = build_run_env(repo, None, run_env_overrides)
    settings = build_settings(
        sections=sections, gpu=gpu, device_memory_gb=device_memory_gb, versions=versions,
        steps=args.steps, warmup=args.warmup, accumulation=accumulation,
        candidates=candidates, trainer_args=args.trainer_arg, run_env=effective_env,
        seed=seed, mix=args.mix, data_digest=distribution_digest(records),
        bounds=plan_bounds(plan), device_count=device_count,
    )
    if bucket_sizes:
        settings["execution"]["bucket_batch_sizes"] = {
            ("*:" if resolution is None else f"{resolution}:") + str(bucket): sizes
            for (resolution, bucket), sizes in bucket_sizes.items()
        }
    cache_key = settings_digest(settings)
    cache = load_cache(paths["cache"])
    if args.mem_probe:
        shadowed = probe_conflicts([*effective_env.get("PYTHONPATH", "").split(os.pathsep),
                                    repo])
        if shadowed:
            raise ValueError(
                "the memory probe would shadow an existing sitecustomize module at "
                + ", ".join(sorted(set(shadowed)))
                + "; pass --no-mem-probe to run without sampling reserved memory"
            )
    probe_dir = write_probe(str(Path(paths["run_dir"]) / "_mem_probe")) \
        if args.mem_probe else None

    fallback = int(args.fallback_batch_size or min(candidates))
    warnings: List[str] = []
    selections: List[Selection] = []
    unscreened: List[Tuple[Combination, str]] = []
    probes: Dict[Tuple[int, int], SamplingProbe] = {}
    stats_by: Dict[Tuple[int, int], List[CandidateStats]] = {}
    all_records: List[RunRecord] = list(cache_records(cache, cache_key))
    cached_runs = len(all_records)
    budget = int(args.max_runs) if args.max_runs else 0
    runs_done = 0
    exhausted = False

    for combo in sorted(combos, key=lambda item: (-item.draw_share, item.resolution_id,
                                                  item.bucket_index)):
        keyed = (combo.resolution_id, combo.bucket_index)
        if combo.draw_share < args.min_caption_share:
            unscreened.append((combo, (
                f"holds {combo.draw_share:.3%} of its resolution's draw mass "
                f"({combo.captions} captions), below --min-caption-share "
                f"{args.min_caption_share:.1%}; every in-bucket draw would come with "
                f"{1.0 / max(combo.draw_share, 1e-9) - 1.0:.0f} discarded draws")))
            continue
        if bucket_sizes:
            combo_candidates = (
                bucket_sizes.get((combo.resolution_id, combo.bucket_index))
                or bucket_sizes.get((None, combo.bucket_index)) or [])
            if not combo_candidates:
                unscreened.append((combo, (
                    "not listed in --bucket-batch-sizes; this pass only measures the "
                    "buckets it names")))
                continue
        else:
            combo_candidates = list(candidates)
        if args.draw_probe:
            probe = probe_sampling(
                records, plan, combo, max(all_sizes), sink, seed=seed, policy=policy,
                initial_stage=curriculum_start, batches=int(args.probe_batches),
                max_seconds=float(args.probe_seconds))
            probes[keyed] = probe
            if probe.unexpected_batches:
                unscreened.append((combo, (
                    "the screening plan emitted "
                    f"{probe.unexpected_batches} micro-batches from other buckets; this "
                    "buckets bounds or the sink size are wrong")))
                continue
            if probe.truncated:
                unscreened.append((combo, (
                    f"the sampler filled fewer than {args.probe_batches} micro-batches in "
                    f"{args.probe_seconds:.0f}s: the bucket is too rare to screen at this "
                    "run length")))
                continue

        existing = [record for record in all_records
                    if (record.resolution_id, record.bucket_index) == keyed]
        measured = 0
        for candidate in combo_candidates:
            if any(record.batch_size == candidate for record in existing):
                continue
            if budget and runs_done >= budget:
                exhausted = True
                break
            measured += 1
            for repeat in range(int(args.repeats)):
                if budget and runs_done >= budget:
                    exhausted = True
                    break
                run_tag = f"res{combo.resolution_id}-len{combo.bucket_index}"
                run_name = f"{run_tag}-B{candidate}-r{repeat}"
                outcome = run_training_run(
                    python=args.python, configs=configs, run_name=run_name,
                    run_dir=str(Path(paths["run_dir"]) / run_tag / f"B{candidate}-r{repeat}"),
                    mix=args.mix,
                    plan=screening_plan(plan, combo.resolution_id, combo.bucket_index,
                                        candidate, sink),
                    tokenizer=args.text_encoder, vae=args.vae, steps=args.steps,
                    warmup=args.warmup, accumulation=accumulation,
                    trainer_args=args.trainer_arg, env_overrides=run_env_overrides,
                    probe_dir=probe_dir, timeout=float(args.timeout), repo=repo)
                record = record_from_outcome(combo, candidate, repeat, outcome, accumulation)
                all_records.append(record)
                existing.append(record)
                runs_done += 1
                cache_append(cache, cache_key, settings, [record])
                store_cache(paths["cache"], cache)
                print(f"    {combo.label} B={candidate} run {repeat + 1}/{args.repeats}: "
                      f"{record.status}"
                      + (f" {record.ms_per_sample:.3f} ms/sample"
                         if record.ms_per_sample is not None else "")
                      + (f" ({record.note})" if record.note else ""), flush=True)
                if record.status != "ok":
                    # An OOM or a crash will not become a measurement by repeating
                    # it: keep the record and spend the time on the next candidate.
                    break

        stats = candidate_stats(existing, combo_candidates)
        stats_by[keyed] = stats
        chosen, reason, candidate_stats_chosen, runner_up, at_edge = choose_batch_size(stats)
        if chosen is None:
            unscreened.append((combo, reason))
            continue
        if measured and measured < len(combo_candidates):
            warnings.append(
                f"{combo.label}: only {measured} of {len(combo_candidates)} candidates were run "
                "before the run budget ended; the choice is over the ones that were"
            )
        selection = Selection(combination=combo, batch_size=int(chosen), source="measured",
                              reason=reason, candidate=candidate_stats_chosen,
                              runner_up=runner_up, at_edge=at_edge)
        peak = None
        if candidate_stats_chosen is not None:
            peak = (candidate_stats_chosen.peak_reserved_gb
                    or candidate_stats_chosen.peak_allocated_gb)
        if peak is not None and device_memory_gb > 0:
            selection.margin_gb = float(device_memory_gb - peak)
            selection.margin_fraction = float((device_memory_gb - peak) / device_memory_gb)
        selections.append(selection)
        if exhausted:
            break

    # Whatever was not measured carries a declared fallback, never the
    # placeholder the plan arrived with.  A budget that ran out leaves buckets
    # that never even started, so name them here rather than only in the log.
    selected = {(selection.combination.resolution_id, selection.combination.bucket_index):
                selection.batch_size for selection in selections}
    reported = {(combo.resolution_id, combo.bucket_index) for combo, _ in unscreened}
    for combo in combos:
        keyed = (combo.resolution_id, combo.bucket_index)
        if keyed not in selected and keyed not in reported:
            unscreened.append((combo, (
                f"the run budget (--max-runs {budget}) was reached before this bucket was "
                "screened" if budget else
                "the screen ended before this bucket was measured")))
            reported.add(keyed)
    for selection in selections:
        if selection.margin_fraction is not None \
                and selection.margin_fraction < args.min_margin_fraction:
            warnings.append(
                f"{selection.combination.label} at batch {selection.batch_size} leaves only "
                f"{selection.margin_fraction:.1%} of the device free; lower the size or "
                "check the measurement"
            )
    sizes = dict(selected)
    for combo in combos:
        keyed = (combo.resolution_id, combo.bucket_index)
        if keyed not in sizes:
            sizes[keyed] = fallback

    write_plan(paths["out"], apply_sizes(plan, sizes))
    measurements = {
        "cache_key": cache_key,
        "settings": settings,
        "plan_in": str(args.plan),
        "plan_out": paths["out"],
        "candidates": [int(value) for value in candidates],
        "bucket_batch_sizes": {
            ("*:" if resolution is None else f"{resolution}:") + str(bucket): sizes
            for (resolution, bucket), sizes in bucket_sizes.items()
        },
        "fallback_batch_size": fallback,
        "selections": [selection.to_json() for selection in selections],
        "not_screened": [
            {**combo.to_json(), "fallback_batch_size": fallback, "why": why}
            for combo, why in unscreened
        ],
        "runs": [record.to_json() for record in all_records],
        "sampling_probe": {
            f"res{resolution_id}/bucket{bucket_index}": {
                "emitted_batches": probe.emitted_batches,
                "emitted_samples": probe.emitted_samples,
                "draws": probe.draws,
                "draws_per_sample": probe.draws_per_sample,
                "ms_per_sample": probe.ms_per_sample,
                "unexpected_batches": probe.unexpected_batches,
                "truncated": probe.truncated,
            }
            for (resolution_id, bucket_index), probe in sorted(probes.items())
        },
    }
    Path(paths["measurements"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["measurements"]).write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for selection in selections:
        if selection.at_edge:
            best = (selection.candidate.batch_size if selection.candidate
                    else selection.batch_size)
            warnings.append(
                f"{selection.combination.label}: the lowest per-sample time is at batch "
                f"{best}, the edge of the screened candidates, so the optimum may lie "
                "outside them"
            )
    # Resolutions of the plan that this screen did not touch keep their own
    # sizes; say so rather than letting a reader assume every bucket was
    # measured, or that an untouched one still holds a placeholder.
    kept = {resolution_id: [bucket.batch_size for bucket in plan[resolution_id]]
            for resolution_id in sorted(plan) if resolution_id not in set(requested)}
    if kept:
        warnings.append(
            "the plan also covers resolutions "
            + ", ".join(str(resolution_id) for resolution_id in sorted(kept))
            + ", which this screen did not measure; their batch sizes are kept as the "
            "plan had them"
        )
    print(f"runs executed: {runs_done}; measurements reused from the cache: "
          f"{cached_runs}", flush=True)
    context = ReportContext(
        argv=argv, plan_path=str(args.plan), out_path=paths["out"],
        measurements_path=paths["measurements"], cache_path=paths["cache"],
        run_dir=paths["run_dir"], mix=args.mix, entries=list(entries),
        base_config=str(args.base_config), extra_configs=list(args.config),
        trainer_args=list(args.trainer_arg), vae=str(args.vae),
        text_encoder=str(args.text_encoder), gpu=gpu,
        device_memory_gb=float(device_memory_gb), candidates=candidates,
        steps=int(args.steps), warmup=int(args.warmup), repeats=int(args.repeats),
        accumulation=accumulation, sink=sink,
        min_caption_share=float(args.min_caption_share),
        min_margin_fraction=float(args.min_margin_fraction),
        cache_key=cache_key, settings=settings, selections=selections,
        unscreened=unscreened, stats=stats_by, records=all_records,
        resolutions=requested, kept_resolutions=kept, fallback_size=fallback,
        warnings=warnings, draw_probe=bool(args.draw_probe), probes=probes,
    )
    Path(paths["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["report"]).write_text(render_report(context), encoding="utf-8")

    print(f"wrote plan: {paths['out']}")
    print(f"wrote measurements: {paths['measurements']}")
    print(f"wrote report: {paths['report']}")
    for selection in selections:
        combo = selection.combination
        print(f"  {combo.label} {combo.interval}: batch {selection.batch_size} "
              f"({selection.reason.split(';')[0]})")
    if unscreened:
        print(f"not screened: {len(unscreened)} bucket(s), fallback size {fallback} "
              f"(see the report's 'Not screened' section)")
    for warning in warnings:
        print(f"batch_size_screen: warning: {warning}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen a micro-batch size per (resolution, length bucket) on the GPU.",
    )
    parser.add_argument("--plan", required=True,
                        help="bucket plan JSON with the boundaries to keep and "
                             "placeholder batch sizes to replace")
    parser.add_argument("--mix", required=True,
                        help='dataset mix, same syntax as [data] mix: "path:weight ..."')
    parser.add_argument("--resolutions", nargs="+", default=None,
                        help="resolution ids to screen, or 'all' (the default)")
    parser.add_argument("--batch-sizes", nargs="+",
                        default=[str(value) for value in DEFAULT_CANDIDATES],
                        help="candidate batch sizes, e.g. --batch-sizes 4 8 12 16 24 32")
    parser.add_argument("--bucket-batch-sizes", nargs="+", default=[],
                        metavar="BUCKET:SIZES",
                        help="per-bucket candidates for a follow-up pass, e.g. "
                             "--bucket-batch-sizes 0:80,96 5:40,48 (prefix with the "
                             "resolution id, as 4:0:80,96, to limit one resolution). "
                             "Buckets no entry names are skipped, so a dense pass can "
                             "probe only the sizes between one that passed and one "
                             "that failed")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="optimizer steps per run")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="optimizer steps dropped from the steady figure; leave room "
                             "for the first torch.compile of the shape")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="runs per candidate, so the report can state the spread")
    parser.add_argument("--accumulation", type=int, default=None,
                        help="micro-batches per optimizer step; defaults to "
                             "[train] gradient_accumulation_steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampler seed; defaults to [train] seed")
    parser.add_argument("--base-config", default="configs/base.toml",
                        help="training recipe the measurements are taken in")
    parser.add_argument("--config", action="append", default=[],
                        help="extra config file, applied after the base one (repeatable)")
    parser.add_argument("--trainer-arg", action="append", default=[],
                        help="extra flag for src.train.train, e.g. --trainer-arg --no-compile "
                             "(repeatable; part of the cache key)")
    parser.add_argument("--vae", default=None, help="vae checkpoint or directory")
    parser.add_argument("--text-encoder", required=True,
                        help="frozen text encoder directory (also tokenizes a missing "
                             "caption-length sidecar)")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the training runs")
    parser.add_argument("--env", action="append", default=[],
                        help="extra KEY=VALUE environment for the runs (repeatable; part "
                             "of the cache key)")
    parser.add_argument("--out", default=None, help="plan to write with measured sizes")
    parser.add_argument("--report", default=None, help="report to write")
    parser.add_argument("--measurements", default=None, help="measurement detail to write")
    parser.add_argument("--cache", default=None, help="measurement cache file")
    parser.add_argument("--run-dir", default=None, help="where run logs and configs go")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="seconds before a single run is killed and recorded as such")
    parser.add_argument("--min-caption-share", type=float, default=DEFAULT_MIN_CAPTION_SHARE,
                        help="do not screen a bucket holding less than this share of its "
                             "resolution's draws: filling a micro-batch would cost far more "
                             "discarded draws than measured ones")
    parser.add_argument("--bucket-mass-out", default=None, metavar="PATH",
                        help="write the per-bucket draw mass implied by --mix and "
                             "--plan here and stop without screening.  This is the "
                             "input merge_screen_results.py needs to line the buckets "
                             "up in time; it covers every resolution the mix has, not "
                             "only those --resolutions names")
    parser.add_argument("--min-margin-fraction", type=float,
                        default=DEFAULT_MIN_MARGIN_FRACTION,
                        help="warn when the chosen size leaves less than this fraction of "
                             "the device memory free")
    parser.add_argument("--fallback-batch-size", type=int, default=None,
                        help="size written for buckets that were not screened; defaults to "
                             "the smallest candidate")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="stop after this many training runs and report the rest as not "
                             "screened (0 = no limit)")
    parser.add_argument("--gpu", default=None, help="device name for the cache key")
    parser.add_argument("--device-memory-gb", type=float, default=None,
                        help="total device memory, for the safety margin")
    parser.add_argument("--device-count", type=int, default=None,
                        help="visible device count for the cache key")
    parser.add_argument("--mem-probe", action=argparse.BooleanOptionalAction, default=False,
                        help="sample peak reserved memory inside the training process; "
                             "off by default because the trainer's [throughput-summary] "
                             "already reports it, and injecting a probe into the "
                             "training process is one more thing that can break a run")
    parser.add_argument("--draw-probe", action=argparse.BooleanOptionalAction, default=True,
                        help="measure the sampler's discard cost per bucket on the CPU")
    parser.add_argument("--probe-batches", type=int, default=16,
                        help="micro-batches the sampling probe fills before it stops")
    parser.add_argument("--probe-seconds", type=float, default=30.0,
                        help="seconds the sampling probe may take before it gives up")
    return parser.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.resolutions is not None and [value.lower() for value in args.resolutions] == ["all"]:
        args.resolutions = None
    try:
        return screen(args, argv)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"batch_size_screen: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
