#!/usr/bin/env python3
"""Merge batch-size screen results into the plan training runs with.

The sweep measures, for each (resolution, caption-length bucket), the steady
cost of a micro-batch at several candidate sizes.  This reads every cache and
writes the plan training uses: one measured size per bucket.

The sizes are chosen so the buckets line up in time.  Every rank draws its own
micro-batches and the reduction at the optimizer-step boundary waits for the
slowest rank, so a step costs what the slowest rank's micro-batches cost, and a
bucket that runs far faster than the rest buys nothing - its time hides behind
the wait.  So for a target micro-batch time ``T`` each bucket takes the largest
measured candidate whose micro-batch time (``batch_size * ms_per_sample``)
stays at or below ``T``.  A small target means short steps; a large one means
steps that carry more samples, because the bigger micro-batches already
measured a lower time per sample.  Sweeping ``T`` over a list of targets,
simulating a step for each one (``ranks`` ranks, each drawing ``accumulation``
micro-batches from the bucket mix) and keeping the target whose step wall clock
per sample is lowest picks every size at once.  Only finished runs are
candidates: an out-of-memory run is a fact about the candidate, not a time.

Two inputs make that decision possible.  The caches hold the measured times,
and ``--bucket-mass`` holds how often each bucket is drawn: a JSON file mapping
each resolution id to one value per bucket of the plan's resolution, the
bucket's share of the draws the mix makes.  It is required.  Without the draw
mass the alignment cannot be simulated, and this script stops rather than
falling back to picking each bucket's own fastest candidate, which is the rule
it replaced.  Generate the file on the machine that holds the corpus with
``batch_size_screen.py --bucket-mass-out``, which derives it from the same mix
and the same plan.

A bucket with no finished run takes no part in the choice: it keeps the batch
size the plan declares for it (the fallback), and the simulation gives it the
per-sample time of its nearest measured neighbour at the same resolution.

Usage:
  python -m scripts.bench.merge_screen_results \
      --plan bucket_plans/256p-k10-13src.json \
      --cache /tmp/screen_cache/a/cache-res1.json /tmp/screen_cache/c/cache-res1.json \
      --bucket-mass /tmp/bucket-mass.json \
      --out bucket_plans/256p-k10-13src.screened.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Plan = Dict[str, List[Dict[str, Any]]]
BucketKey = Tuple[int, int]
Candidates = Dict[BucketKey, Dict[int, float]]
Shares = Dict[BucketKey, float]

# One simulated optimizer step: this many ranks, each drawing this many
# micro-batches, which is what makes the step's wall clock a slowest-rank
# maximum.  The defaults match the training recipe; ``trials`` bounds the
# simulation's own noise and ``seed`` keeps a scan reproducible.
DEFAULT_RANKS = 8
DEFAULT_ACCUMULATION = 16
DEFAULT_TRIALS = 20000
DEFAULT_SEED = 0
# Targets to sweep, in milliseconds of micro-batch time.  The optimum is a
# plateau - a range of targets produces the same sizes - so round numbers read
# better than the breakpoints themselves.
DEFAULT_TARGETS = (500, 600, 700, 800, 900, 1000, 1100, 1300, 1500, 2000, 3000)


def read_plan(path: str) -> Plan:
    with open(path) as handle:
        raw = json.load(handle)
    plan: Plan = {}
    for resolution_id, buckets in raw.items():
        plan[str(int(resolution_id))] = [
            {"max_length": int(bucket["max_length"]),
             "batch_size": int(bucket["batch_size"])}
            for bucket in buckets
        ]
    return plan


def read_cache(path: str) -> Iterable[Dict[str, Any]]:
    """Yield one dict per finished measurement in a screen cache.

    A cache holds ``{"version": int, "entries": {key: {"records": [...]}}}``,
    where each record describes one candidate's run for one bucket.
    """
    with open(path) as handle:
        payload = json.load(handle)
    entries = payload.get("entries", {})
    if isinstance(entries, list):  # tolerate an older flat layout
        entries = {"": {"records": entries}}
    for entry in entries.values():
        for record in entry.get("records", []):
            yield record


def read_bucket_mass(path: str) -> Dict[str, List[float]]:
    """Read the per-bucket draw mass.

    The file maps a resolution id to one value per bucket of the plan's
    resolution: the bucket's share of the draws the mix makes.  Only the ratios
    between the values matter, since ``share_by_bucket`` renormalizes them over
    the whole table.  ``batch_size_screen.py --bucket-mass-out`` writes the file
    from the same mix and plan, using the sampler's own draw rule.
    """
    with open(path) as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("the bucket mass must be a JSON object keyed by resolution id")
    mass: Dict[str, List[float]] = {}
    for key, values in raw.items():
        try:
            resolution_id = str(int(key))
        except (TypeError, ValueError):
            raise ValueError(f"bucket mass key {key!r} is not a resolution id") from None
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"bucket mass for resolution {resolution_id} must be a non-empty list")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f"bucket mass for resolution {resolution_id} holds a non-numeric "
                    f"or negative value {value!r}")
        mass[resolution_id] = [float(value) for value in values]
    return mass


def share_by_bucket(mass: Dict[str, List[float]], plan: Plan) -> Shares:
    """Join the mass file to the plan's buckets and normalize to shares."""
    missing = sorted(set(plan) - set(mass), key=int)
    if missing:
        raise ValueError(f"the bucket mass has no entry for resolution ids {missing}")
    unknown = sorted(set(mass) - set(plan), key=int)
    if unknown:
        raise ValueError(
            f"the bucket mass names resolution ids {unknown} that the plan does not have")
    shares: Shares = {}
    for resolution_id, buckets in plan.items():
        values = mass[resolution_id]
        if len(values) != len(buckets):
            raise ValueError(
                f"the bucket mass for resolution {resolution_id} has {len(values)} "
                f"value(s) but the plan has {len(buckets)} bucket(s)")
        for index, value in enumerate(values):
            shares[(int(resolution_id), index)] = float(value)
    total = sum(shares.values())
    if total <= 0:
        raise ValueError("the bucket mass sums to zero, so no bucket is ever drawn")
    return {key: value / total for key, value in shares.items()}


def measured_candidates(caches: Sequence[str]) -> Candidates:
    """The steady per-sample time of every candidate whose run finished.

    Only records with ``status == "ok"`` and a numeric ``ms_per_sample`` take
    part; an out-of-memory, timed-out or otherwise failed run says nothing about
    the candidate's speed.  A candidate measured more than once - over repeats
    within a round or across rounds - is represented by the median of its
    measurements, the same figure the screening report selects on.
    """
    measurements: Dict[BucketKey, Dict[int, List[float]]] = {}
    for path in caches:
        for record in read_cache(path):
            if record.get("status") != "ok":
                continue
            ms = record.get("ms_per_sample")
            if isinstance(ms, bool) or not isinstance(ms, (int, float)):
                continue
            key = (int(record["resolution_id"]), int(record["bucket_index"]))
            by_size = measurements.setdefault(key, {})
            by_size.setdefault(int(record["batch_size"]), []).append(float(ms))
    return {key: {size: float(statistics.median(times)) for size, times in sizes.items()}
            for key, sizes in measurements.items()}


@dataclass(frozen=True)
class BucketChoice:
    """The micro-batch size one bucket runs with, and what it costs."""

    batch_size: int
    ms_per_sample: float
    measured: bool

    @property
    def micro_batch_ms(self) -> float:
        return self.batch_size * self.ms_per_sample


def _neighbour_rate(candidates: Candidates, resolution_id: int, bucket_index: int,
                    fallback_size: int) -> Optional[float]:
    """The per-sample time a measured neighbour implies for a fallback size.

    The neighbour taken is the closest measured bucket of the same resolution
    (a tie goes to the longer bucket, whose per-sample time is the larger and
    therefore the safe side for a slowest-rank model).  The neighbour's time at
    the same batch size is used when it has one, and its time at the closest
    measured size otherwise.
    """
    measured = sorted(index for (rid, index) in candidates
                      if rid == resolution_id and candidates[(rid, index)])
    if not measured:
        return None
    nearest = min(measured, key=lambda index: (abs(index - bucket_index), -index))
    sizes = candidates[(resolution_id, nearest)]
    if fallback_size in sizes:
        return sizes[fallback_size]
    size = min(sizes, key=lambda candidate: (abs(candidate - fallback_size), candidate))
    return sizes[size]


def _pick(sizes: Dict[int, float], target: Optional[float]) -> Tuple[int, float]:
    """The size one bucket takes under the rule, and its per-sample time."""
    if target is None:
        size = min(sizes, key=lambda candidate: (sizes[candidate], candidate))
    else:
        fitting = [candidate for candidate, ms in sizes.items() if candidate * ms <= target]
        size = max(fitting) if fitting else min(sizes)
    return size, sizes[size]


def choice_for(plan: Plan, candidates: Candidates,
               target: Optional[float] = None) -> Dict[BucketKey, BucketChoice]:
    """The size every bucket runs with under one rule.

    ``target`` in milliseconds gives the aligned rule: each bucket takes the
    largest measured candidate whose micro-batch time stays at or below the
    target, and a bucket whose smallest candidate is already over the target
    takes that smallest candidate, because a bucket cannot go lower than what
    was measured.  ``target=None`` gives each bucket its own fastest measured
    candidate, the baseline the alignment is compared against.

    A bucket with no finished run keeps the size the plan declares for it and
    is modelled at its nearest measured neighbour's per-sample time; a
    resolution with no measured bucket at all cannot be modelled and is an
    error rather than a guess.
    """
    choice: Dict[BucketKey, BucketChoice] = {}
    for resolution_id, buckets in plan.items():
        rid = int(resolution_id)
        for index, bucket in enumerate(buckets):
            sizes = candidates.get((rid, index))
            if sizes:
                size, ms = _pick(sizes, target)
                choice[(rid, index)] = BucketChoice(size, ms, True)
            else:
                fallback = int(bucket["batch_size"])
                rate = _neighbour_rate(candidates, rid, index, fallback)
                if rate is None:
                    raise ValueError(
                        f"resolution {rid} has no measured bucket at all, so its "
                        "draws cannot be modelled; screen at least one of its "
                        "buckets first")
                choice[(rid, index)] = BucketChoice(fallback, rate, False)
    return choice


@dataclass(frozen=True)
class StepStats:
    """What one optimizer step costs under one choice of sizes."""

    mean_step_ms: float
    slowest_step_ms: float
    mean_samples: float

    @property
    def premium(self) -> float:
        """How much slower the slowest rank is than the mean rank."""
        return self.slowest_step_ms / self.mean_step_ms - 1.0 if self.mean_step_ms else 0.0

    @property
    def effective_ms_per_sample(self) -> float:
        """The step wall clock per sample the step carried."""
        return self.slowest_step_ms / self.mean_samples if self.mean_samples else 0.0


def simulate(choice: Dict[BucketKey, BucketChoice], shares: Shares, *,
             ranks: int = DEFAULT_RANKS, accumulation: int = DEFAULT_ACCUMULATION,
             trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED) -> StepStats:
    """A step's wall clock and samples, averaged over many simulated steps.

    Every rank draws ``accumulation`` micro-batches from the bucket mix, so its
    step time is the sum of their micro-batch times; the step's wall clock is
    the slowest rank's, because the reduction at the boundary waits for it,
    while the samples the step carries are the mean over ranks.  The effective
    time per sample is the wall clock over those samples.

    Every target of a scan shares the seed, which keeps the comparison between
    targets paired rather than independent, and keeps a scan reproducible.
    """
    keys = sorted(shares)
    if any(key not in choice for key in keys):
        raise ValueError("every bucket needs a choice before it can be simulated")
    weights = [shares[key] for key in keys]
    times = [choice[key].micro_batch_ms for key in keys]
    sizes = [choice[key].batch_size for key in keys]
    population = list(range(len(keys)))
    rng = random.Random(seed)
    total_mean = total_slowest = total_samples = 0.0
    for _ in range(trials):
        step_times: List[float] = []
        step_samples: List[int] = []
        for _rank in range(ranks):
            elapsed = 0.0
            carried = 0
            for _ in range(accumulation):
                index = rng.choices(population, weights=weights, k=1)[0]
                elapsed += times[index]
                carried += sizes[index]
            step_times.append(elapsed)
            step_samples.append(carried)
        total_mean += sum(step_times) / ranks
        total_slowest += max(step_times)
        total_samples += sum(step_samples) / ranks
    return StepStats(mean_step_ms=total_mean / trials,
                     slowest_step_ms=total_slowest / trials,
                     mean_samples=total_samples / trials)


@dataclass(frozen=True)
class ScanRow:
    """One target of the sweep, and what it costs.

    ``target=None`` is the reference row: every bucket at its own fastest
    measured candidate, the choice the alignment is meant to beat.
    """

    target: Optional[int]
    stats: StepStats


def scan_targets(plan: Plan, candidates: Candidates, shares: Shares,
                 targets: Sequence[int], *, ranks: int = DEFAULT_RANKS,
                 accumulation: int = DEFAULT_ACCUMULATION,
                 trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED
                 ) -> Tuple[List[ScanRow], ScanRow]:
    """Sweep the targets; return every row and the best one.

    The first target to reach the lowest effective time wins: the choice is a
    plateau - a range of targets produces the same sizes - and the sweep runs
    in increasing order, so the smallest target in the plateau is the answer.
    """
    rows = [ScanRow(None, simulate(choice_for(plan, candidates), shares,
                                   ranks=ranks, accumulation=accumulation,
                                   trials=trials, seed=seed))]
    for target in targets:
        rows.append(ScanRow(
            target,
            simulate(choice_for(plan, candidates, target), shares,
                     ranks=ranks, accumulation=accumulation, trials=trials, seed=seed)))
    best = rows[1]
    for row in rows[2:]:
        if row.stats.effective_ms_per_sample < best.stats.effective_ms_per_sample:
            best = row
    return rows, best


def render_scan(rows: Sequence[ScanRow], best: ScanRow) -> List[str]:
    """The sweep as a table: what each target costs and which one won."""
    lines = [f"{'target':>19}  {'step ms':>9}  {'slowest ms':>10}  {'premium':>8}  "
             f"{'samples/step':>12}  {'ms/sample':>9}"]
    for row in rows:
        label = "each at its fastest" if row.target is None else f"{row.target} ms"
        mark = "  <- best" if row is best else ""
        lines.append(
            f"{label:>19}  {row.stats.mean_step_ms:9.0f}  {row.stats.slowest_step_ms:10.0f}  "
            f"{row.stats.premium:+8.2%}  {row.stats.mean_samples:12.0f}  "
            f"{row.stats.effective_ms_per_sample:9.3f}{mark}")
    return lines


def apply(plan: Plan, choice: Dict[BucketKey, BucketChoice],
          candidates: Candidates) -> Tuple[Plan, List[str]]:
    """Write the chosen sizes into a copy of the plan; describe each choice."""
    merged: Plan = {}
    lines: List[str] = []
    for resolution_id, buckets in plan.items():
        rid = int(resolution_id)
        merged[resolution_id] = []
        for index, bucket in enumerate(buckets):
            entry = choice[(rid, index)]
            merged[resolution_id].append(
                {"max_length": bucket["max_length"], "batch_size": int(entry.batch_size)})
            if candidates.get((rid, index)):
                lines.append(
                    f"res{resolution_id} bucket{index} bound={bucket['max_length']}: "
                    f"batch {entry.batch_size} at {entry.micro_batch_ms:.0f} ms per "
                    f"micro-batch ({entry.ms_per_sample:.2f} ms/sample)")
            else:
                lines.append(
                    f"res{resolution_id} bucket{index} bound={bucket['max_length']}: "
                    f"batch {entry.batch_size} (not screened, declared fallback; "
                    f"modelled at {entry.ms_per_sample:.2f} ms/sample)")
    return merged, lines


def parse_targets(values: Sequence[str]) -> List[int]:
    """Parse ``--targets``; an empty list means the standard sweep."""
    if not values:
        return list(DEFAULT_TARGETS)
    targets: List[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            try:
                target = int(part)
            except ValueError:
                raise ValueError(
                    f"--targets: {part!r} is not a whole number of milliseconds") from None
            if target < 1:
                raise ValueError(f"--targets: {target} must be positive")
            targets.append(target)
    if len(set(targets)) != len(targets):
        raise ValueError(f"--targets: duplicate values in {targets}")
    return targets


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True,
                        help="plan whose bucket bounds and fallbacks to keep")
    parser.add_argument("--cache", required=True, nargs="+",
                        help="screen caches to read, in any order")
    parser.add_argument("--bucket-mass", required=True, metavar="JSON",
                        help="per-bucket draw mass, one value per bucket of each "
                             "plan resolution; required because the alignment is "
                             "simulated over the bucket mix, and there is no "
                             "sensible default without it (write it with "
                             "batch_size_screen.py --bucket-mass-out)")
    parser.add_argument("--out", required=True,
                        help="plan to write with the aligned sizes")
    parser.add_argument("--targets", nargs="+", default=[], metavar="MS",
                        help="micro-batch times to sweep, in milliseconds "
                             "(default: the standard grid)")
    parser.add_argument("--ranks", type=int, default=DEFAULT_RANKS,
                        help="ranks per simulated step (default: 8)")
    parser.add_argument("--accumulation", type=int, default=DEFAULT_ACCUMULATION,
                        help="micro-batches per rank per step (default: 16)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="simulated steps per target (default: 20000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed of the step simulation")
    args = parser.parse_args(argv)

    try:
        plan = read_plan(args.plan)
        shares = share_by_bucket(read_bucket_mass(args.bucket_mass), plan)
        candidates = measured_candidates(args.cache)
        targets = parse_targets(args.targets)
        if args.ranks < 1 or args.accumulation < 1 or args.trials < 1:
            raise ValueError("--ranks, --accumulation and --trials must be positive")
        rows, best = scan_targets(plan, candidates, shares, targets,
                                  ranks=args.ranks, accumulation=args.accumulation,
                                  trials=args.trials, seed=args.seed)
        choice = choice_for(plan, candidates, best.target)
        merged, lines = apply(plan, choice, candidates)
        with open(args.out, "w") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"merge_screen_results: error: {exc}", file=sys.stderr)
        return 2

    for line in render_scan(rows, best):
        print(line)
    reference = rows[0].stats
    print(f"\nbest target: {best.target} ms at "
          f"{best.stats.effective_ms_per_sample:.3f} ms/sample "
          f"(slowest rank +{best.stats.premium:.2%}; "
          f"each bucket at its fastest: {reference.effective_ms_per_sample:.3f} ms/sample)")
    print()
    for line in lines:
        print(line)
    screened = sum(1 for entry in choice.values() if entry.measured)
    print(f"\n{screened} of {len(choice)} (resolution, bucket) pairs had a finished run; "
          f"the rest keep the fallback in {args.plan}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
