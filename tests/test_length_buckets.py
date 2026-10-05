"""Bucket optimization checked against exhaustive search; plans against the trainer reader."""

import itertools

import numpy as np
import pytest

from src.dataset.length_buckets import dump_plan, optimal_boundaries
from src.dataset.sampler import load_bucket_plan


@pytest.mark.parametrize('seed', [0, 1, 7])
def test_optimal_partition_matches_exhaustive_assignment(seed):
    rng = np.random.default_rng(seed)
    probabilities = rng.random(9)
    probabilities[2:5] = 0  # Include a gap with multiple equivalent boundaries.
    probabilities /= probabilities.sum()
    costs = np.array([1, 3, 4, 8, 10, 17, 25, 36, 50], dtype=float)

    def expected_cost(boundaries):
        # Assign each length to its enclosing bucket directly. This is separate
        # from the production dynamic program and its cumulative cost tables.
        assigned = np.searchsorted(boundaries, np.arange(1, 10))
        return np.average(costs[np.asarray(boundaries)[assigned] - 1], weights=probabilities)

    for count in (1, 2, 3, 4):
        candidates = [list(cuts) + [9] for cuts in itertools.combinations(range(1, 9), count - 1)]
        actual = optimal_boundaries(probabilities, count, costs)
        assert actual in candidates
        assert expected_cost(actual) == pytest.approx(min(map(expected_cost, candidates)))


def test_written_plan_preserves_bucket_assignments_in_trainer(tmp_path):
    path = tmp_path / 'plan.json'
    dump_plan(str(path), {1: [8, 32, 64], 2: [16, 64]}, {1: [8, 4, 2], 2: [6, 3]})
    plan = load_bucket_plan(str(path), resolution_ids=[1, 2])
    for resolution, length, bound, batch in [(1, 8, 8, 8), (1, 9, 32, 4),
                                             (1, 64, 64, 2), (2, 16, 16, 6), (2, 17, 64, 3)]:
        _, bucket = plan.bucket_for(resolution, length)
        assert (bucket.max_length, bucket.batch_size) == (bound, batch)


@pytest.mark.parametrize('payload', [
    {"by_resolution": {"1": [{"max_length": 8, "batch_size": 2}]}},
    {"1": [[8, 2]]}, {"1": [{"max_length": 8}]},
    {"1": [{"max_length": 8.5, "batch_size": 2}]},
])
def test_plan_rejects_noncanonical_file_formats(tmp_path, payload):
    import json
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_bucket_plan(path)


def test_inline_json_is_not_a_plan_file():
    with pytest.raises(FileNotFoundError):
        load_bucket_plan('{"1": [{"max_length": 8, "batch_size": 2}]}')
