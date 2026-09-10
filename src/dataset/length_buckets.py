"""Choose retained-length bucket boundaries for the training sampler.

The sampler pads every caption in a micro-batch up to its bucket's upper bound,
so a bucket whose bound sits far above the lengths it actually holds wastes
compute on padding.  For a fixed distribution ``p(l)`` over retained lengths and
a per-sample cost ``C(l)``, the expected wasted compute is

    W(h_1..h_K) = sum_i sum_{l = h_(i-1)+1}^{h_i} p(l) * (C(h_i) - C(l))

and the unpadded term ``sum_l p(l) C(l)`` does not depend on the boundaries, so
minimising ``W`` is the same as minimising expected padded compute.  This module
solves that partition exactly with dynamic programming, and also accepts a
measured per-interval timing table for the refinement pass that follows GPU
screening.

Buckets are closed: bucket ``i`` holds lengths ``h_(i-1)+1 .. h_i``, matching
:meth:`src.dataset.sampler.BucketPlan.bucket_for`, which assigns a length to the
first bucket whose bound is at least as large.  The last bound is always the
configured cap, so a rare long caption is never silently dropped.

Cost proxy
----------

``C(l) = alpha * (I + l) + gamma * (I + l) ** 2`` with ``I`` the number of image
tokens and ``l`` the retained caption tokens.  The linear term stands for the
projections and MLP, the quadratic term for attention over the concatenated
sequence.  :func:`architecture_cost` derives both from the transformer width and
depth; the optimal boundaries depend on their ratio, not their scale, so a
rough coefficient is enough as long as the ratio is not wildly wrong.  The proxy
is not a latency model: it ignores kernels, the frozen text encoder and the
optimizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..utils.prompt_contract import MAX_SEQUENCE_LENGTH


@dataclass(frozen=True)
class CostModel:
    """Per-sample compute proxy for a caption of a given retained length."""

    alpha: float
    gamma: float
    image_tokens: int

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.gamma < 0:
            raise ValueError("cost coefficients must be non-negative")
        if self.image_tokens < 1:
            raise ValueError("image_tokens must be positive")
        if self.alpha == 0 and self.gamma == 0:
            raise ValueError("a cost model needs at least one non-zero coefficient")

    def per_sample(self, length: int) -> float:
        sequence = self.image_tokens + int(length)
        return self.alpha * sequence + self.gamma * sequence * sequence

    def over_lengths(self, max_length: int) -> np.ndarray:
        """Cost of every retained length 1..``max_length``, index ``l-1``."""
        lengths = np.arange(1, int(max_length) + 1, dtype=np.float64)
        sequence = self.image_tokens + lengths
        return self.alpha * sequence + self.gamma * sequence * sequence


def architecture_cost(width: int, layers: int, image_tokens: int) -> CostModel:
    """Cost proxy coefficients for a transformer of the given width and depth.

    Per layer, per token, the projections and the MLP cost about ``24 * width**2``
    multiply-accumulates; attention scores and the value mixing cost about
    ``4 * width`` per token per token of sequence length.  Both are multiplied by
    the layer count.  Only the ratio matters for the optimum, so the constants
    (which ignore modulation blocks and the final head) are deliberately rough.
    """
    if width < 1 or layers < 1:
        raise ValueError("width and layers must be positive")
    return CostModel(alpha=24.0 * width * width * layers,
                     gamma=4.0 * width * layers,
                     image_tokens=int(image_tokens))


def _prefix_sums(probabilities: np.ndarray, costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return F and G with ``F[b] = sum_{l<=b} p(l)`` and ``G[b] = sum p(l) C(l)``."""
    weights = np.concatenate(([0.0], np.asarray(probabilities, dtype=np.float64)))
    weighted = np.concatenate(([0.0], np.asarray(probabilities, dtype=np.float64)
                               * np.asarray(costs, dtype=np.float64)))
    return np.cumsum(weights), np.cumsum(weighted)


def interval_cost(prefix_f: np.ndarray, prefix_g: np.ndarray, costs: np.ndarray,
                  lower: int, upper: int) -> float:
    """Expected padding waste of one bucket holding lengths ``lower+1..upper``."""
    return float(costs[upper - 1] * (prefix_f[upper] - prefix_f[lower])
                 - (prefix_g[upper] - prefix_g[lower]))


def padding_waste(probabilities: Sequence[float], boundaries: Sequence[int],
                  costs: Sequence[float]) -> float:
    """Expected wasted compute for a partition, in the same units as ``costs``."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    boundaries = [int(bound) for bound in boundaries]
    _validate_boundaries(boundaries, len(probabilities))
    prefix_f, prefix_g = _prefix_sums(probabilities, costs)
    lower = 0
    total = 0.0
    for upper in boundaries:
        total += interval_cost(prefix_f, prefix_g, costs, lower, upper)
        lower = upper
    return total


def per_bucket_waste(probabilities: Sequence[float], boundaries: Sequence[int],
                     costs: Sequence[float]) -> List[float]:
    """Padding waste of each bucket, in order."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    boundaries = [int(bound) for bound in boundaries]
    _validate_boundaries(boundaries, len(probabilities))
    prefix_f, prefix_g = _prefix_sums(probabilities, costs)
    out: List[float] = []
    lower = 0
    for upper in boundaries:
        out.append(interval_cost(prefix_f, prefix_g, costs, lower, upper))
        lower = upper
    return out


def _validate_boundaries(boundaries: Sequence[int], max_length: int) -> None:
    if not boundaries:
        raise ValueError("a partition needs at least one boundary")
    if boundaries[-1] != max_length:
        raise ValueError(f"the last boundary must be {max_length}, got {boundaries[-1]}")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries must be strictly increasing")
    if boundaries[0] < 1:
        raise ValueError("boundaries must be positive")


def optimal_boundaries(probabilities: Sequence[float], num_buckets: int,
                       costs: Sequence[float]) -> List[int]:
    """Exact minimiser of :func:`padding_waste` for ``num_buckets`` buckets.

    Dynamic programming over the ordered partition.  Ties are broken towards the
    smaller boundary, which keeps the result stable in zero-mass regions where
    many partitions cost the same.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.shape != costs.shape:
        raise ValueError("probabilities and costs must be 1-D and the same length")
    if probabilities.size == 0:
        raise ValueError("probabilities must not be empty")
    if np.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")
    num_buckets = int(num_buckets)
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1")
    max_length = probabilities.size
    if num_buckets > max_length:
        raise ValueError("cannot have more buckets than retained lengths")

    prefix_f, prefix_g = _prefix_sums(probabilities, costs)
    inf = np.inf
    best = np.full((num_buckets + 1, max_length + 1), inf, dtype=np.float64)
    parent = np.full((num_buckets + 1, max_length + 1), -1, dtype=np.int64)
    best[0, 0] = 0.0

    for k in range(1, num_buckets + 1):
        for b in range(k, max_length + 1):
            # predecessor a ranges over k-1 .. b-1 so every earlier bucket is non-empty
            predecessors = np.arange(k - 1, b)
            if predecessors.size == 0:
                continue
            interval = costs[b - 1] * (prefix_f[b] - prefix_f[predecessors]) \
                - (prefix_g[b] - prefix_g[predecessors])
            candidates = best[k - 1, predecessors] + interval
            if not np.any(np.isfinite(candidates)):
                continue
            index = int(np.argmin(candidates))
            best[k, b] = candidates[index]
            parent[k, b] = predecessors[index]

    if not np.isfinite(best[num_buckets, max_length]):
        raise ValueError("no feasible partition for these lengths and bucket count")

    boundaries: List[int] = []
    b = max_length
    for k in range(num_buckets, 0, -1):
        boundaries.append(int(b))
        b = int(parent[k, b])
    boundaries.reverse()
    _validate_boundaries(boundaries, max_length)
    return boundaries


def optimal_boundaries_from_table(
    probabilities: Sequence[float],
    num_buckets: int,
    interval_time: Callable[[int, int], float],
) -> List[int]:
    """Exact minimiser when the cost of a bucket is a measured time, not a proxy.

    ``interval_time(a, b)`` must return the measured (or estimated) seconds per
    sample for lengths ``a+1..b`` padded to ``b``.  Every allowed interval needs a
    finite cost; a measured table is normally only available near the boundaries
    the proxy suggested, and unmeasured intervals must be estimated explicitly
    rather than treated as free.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty 1-D sequence")
    max_length = probabilities.size
    num_buckets = int(num_buckets)
    if not 1 <= num_buckets <= max_length:
        raise ValueError("num_buckets must be in [1, number of lengths]")

    mass = np.concatenate(([0.0], np.cumsum(probabilities)))
    inf = np.inf
    best = np.full((num_buckets + 1, max_length + 1), inf, dtype=np.float64)
    parent = np.full((num_buckets + 1, max_length + 1), -1, dtype=np.int64)
    best[0, 0] = 0.0

    for k in range(1, num_buckets + 1):
        for b in range(k, max_length + 1):
            for a in range(k - 1, b):
                if not np.isfinite(best[k - 1, a]):
                    continue
                seconds = float(interval_time(a, b))
                if not np.isfinite(seconds):
                    continue
                candidate = best[k - 1, a] + seconds * (mass[b] - mass[a])
                if candidate < best[k, b]:
                    best[k, b] = candidate
                    parent[k, b] = a

    if not np.isfinite(best[num_buckets, max_length]):
        raise ValueError("no feasible partition for this timing table")

    boundaries: List[int] = []
    b = max_length
    for k in range(num_buckets, 0, -1):
        boundaries.append(int(b))
        b = int(parent[k, b])
    boundaries.reverse()
    _validate_boundaries(boundaries, max_length)
    return boundaries


def uniform_boundaries(max_length: int, num_buckets: int) -> List[int]:
    """Equal-width partition, the baseline the optimiser is compared against."""
    if not 1 <= num_buckets <= max_length:
        raise ValueError("num_buckets must be in [1, max_length]")
    step = max_length / num_buckets
    boundaries = [int(round(step * (index + 1))) for index in range(num_buckets)]
    boundaries[-1] = max_length
    return _repair(boundaries, max_length)


def equal_mass_boundaries(probabilities: Sequence[float], num_buckets: int) -> List[int]:
    """Partition that puts equal probability mass in each bucket."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    max_length = probabilities.size
    if not 1 <= num_buckets <= max_length:
        raise ValueError("num_buckets must be in [1, max_length]")
    cumulative = np.cumsum(probabilities)
    if cumulative[-1] <= 0:
        return uniform_boundaries(max_length, num_buckets)
    targets = cumulative[-1] * (np.arange(1, num_buckets + 1) / num_buckets)
    boundaries = [int(np.searchsorted(cumulative, target, side="left")) + 1
                  for target in targets]
    boundaries[-1] = max_length
    return _repair(boundaries, max_length)


def _repair(boundaries: List[int], max_length: int) -> List[int]:
    """Force strict increase while keeping the final boundary at the cap."""
    fixed: List[int] = []
    for index, bound in enumerate(boundaries):
        bound = max(int(bound), 1)
        if fixed:
            bound = max(bound, fixed[-1] + 1)
        fixed.append(bound)
    fixed[-1] = max_length
    for index in range(len(fixed) - 2, -1, -1):
        fixed[index] = min(fixed[index], fixed[index + 1] - 1)
    if fixed[0] < 1:
        raise ValueError("cannot repair boundaries into a valid partition")
    return fixed


def histogram_from_lengths(lengths: Iterable[int], max_length: int = MAX_SEQUENCE_LENGTH,
                           weights: Optional[Iterable[float]] = None) -> np.ndarray:
    """Probability of each retained length 1..``max_length``, index ``l-1``.

    Lengths above the cap are counted at the cap, which is what the training
    contract does when it truncates.  Lengths below 1 are ignored.
    """
    counts = np.zeros(int(max_length), dtype=np.float64)
    lengths = list(lengths)
    if weights is None:
        weights = [1.0] * len(lengths)
    for length, weight in zip(lengths, weights):
        value = int(length)
        if value < 1:
            continue
        counts[min(value, int(max_length)) - 1] += float(weight)
    total = counts.sum()
    if total <= 0:
        raise ValueError("no positive-length captions in the histogram input")
    return counts / total


def plan_json(boundaries_by_resolution: Mapping[int, Sequence[int]],
              batch_sizes_by_resolution: Mapping[int, Sequence[int]]) -> Dict[str, List[dict]]:
    """Bucket plan in the shape ``src.train.train.load_bucket_plan`` expects."""
    plan: Dict[str, List[dict]] = {}
    for resolution_id, boundaries in boundaries_by_resolution.items():
        batch_sizes = batch_sizes_by_resolution.get(int(resolution_id))
        if batch_sizes is None:
            raise ValueError(f"resolution {resolution_id} has no batch sizes")
        if len(batch_sizes) != len(boundaries):
            raise ValueError(f"resolution {resolution_id}: "
                             f"{len(boundaries)} buckets but {len(batch_sizes)} batch sizes")
        plan[str(int(resolution_id))] = [
            {"max_length": int(bound), "batch_size": int(size)}
            for bound, size in zip(boundaries, batch_sizes)
        ]
    return plan


def dump_plan(path: str, boundaries_by_resolution: Mapping[int, Sequence[int]],
              batch_sizes_by_resolution: Mapping[int, Sequence[int]]) -> None:
    """Write a plan file in the loader's shape.

    No extra keys are written: the loader treats every top-level key as a
    resolution id, so provenance belongs in a separate report next to the plan.
    """
    payload = plan_json(boundaries_by_resolution, batch_sizes_by_resolution)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
