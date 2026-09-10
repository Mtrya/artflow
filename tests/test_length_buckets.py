"""Tests for the retained-length bucket planner."""

import itertools
import json
import random

import numpy as np
import pytest

from src.dataset.length_buckets import (
    CostModel,
    architecture_cost,
    dump_plan,
    equal_mass_boundaries,
    histogram_from_lengths,
    optimal_boundaries,
    optimal_boundaries_from_table,
    padding_waste,
    per_bucket_waste,
    plan_json,
    uniform_boundaries,
)
from src.train.train import load_bucket_plan


def brute_force_optimum(probabilities, num_buckets, costs):
    """Smallest waste over every ordered partition, for small problems only."""
    max_length = len(probabilities)
    best = None
    for cuts in itertools.combinations(range(1, max_length), num_buckets - 1):
        boundaries = list(cuts) + [max_length]
        waste = padding_waste(probabilities, boundaries, costs)
        if best is None or waste < best[0] - 1e-12:
            best = (waste, boundaries)
    return best


def test_cost_model_is_monotone_and_quadratic_in_the_tail():
    model = CostModel(alpha=1.0, gamma=0.0, image_tokens=100)
    linear = model.over_lengths(50)
    assert np.all(np.diff(linear) > 0)

    model = CostModel(alpha=0.0, gamma=1.0, image_tokens=10)
    costs = model.over_lengths(10)
    assert costs[1] - costs[0] == pytest.approx(2 * 11 + 1)


def test_architecture_cost_rejects_nonsense():
    with pytest.raises(ValueError):
        architecture_cost(0, 24, 1024)
    with pytest.raises(ValueError):
        architecture_cost(1152, 0, 1024)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dp_matches_brute_force(seed):
    rng = random.Random(seed)
    max_length = 9
    probabilities = [rng.random() for _ in range(max_length)]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    model = CostModel(alpha=3.0, gamma=0.5, image_tokens=5)
    costs = model.over_lengths(max_length)

    for num_buckets in (1, 2, 3, 4):
        expected, _ = brute_force_optimum(probabilities, num_buckets, costs)
        boundaries = optimal_boundaries(probabilities, num_buckets, costs)
        assert len(boundaries) == num_buckets
        assert boundaries[-1] == max_length
        assert all(right > left for left, right in zip(boundaries, boundaries[1:]))
        assert padding_waste(probabilities, boundaries, costs) == pytest.approx(expected, abs=1e-12)


def test_optimal_never_worse_than_uniform_or_equal_mass():
    rng = np.random.default_rng(3)
    max_length = 200
    probabilities = rng.gamma(2.0, 1.0, max_length)
    probabilities[40:120] = 0.0            # a zero-mass gap, as real captions have
    probabilities /= probabilities.sum()
    model = CostModel(alpha=24.0 * 1152 ** 2 * 24, gamma=4.0 * 1152 * 24, image_tokens=4096)
    costs = model.over_lengths(max_length)

    optimal = optimal_boundaries(probabilities, 4, costs)
    best = padding_waste(probabilities, optimal, costs)
    assert best <= padding_waste(probabilities, uniform_boundaries(max_length, 4), costs) + 1e-9
    assert best <= padding_waste(probabilities, equal_mass_boundaries(probabilities, 4), costs) + 1e-9


def test_boundaries_are_deterministic_in_zero_mass_regions():
    probabilities = np.zeros(32)
    probabilities[0] = 0.5
    probabilities[31] = 0.5
    model = CostModel(alpha=1.0, gamma=1.0, image_tokens=1)
    costs = model.over_lengths(32)
    first = optimal_boundaries(probabilities, 3, costs)
    second = optimal_boundaries(probabilities, 3, costs)
    assert first == second
    assert first[-1] == 32


def test_per_bucket_waste_sums_to_total():
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    costs = np.array([1.0, 4.0, 9.0, 16.0])
    boundaries = [2, 4]
    parts = per_bucket_waste(probabilities, boundaries, costs)
    assert sum(parts) == pytest.approx(padding_waste(probabilities, boundaries, costs))


def test_padding_waste_is_zero_for_point_mass_on_bounds():
    probabilities = np.array([0.0, 1.0, 0.0, 0.0])
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    assert padding_waste(probabilities, [2, 4], costs) == pytest.approx(0.0)


def test_boundaries_must_cover_the_cap():
    probabilities = np.array([0.25] * 4)
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        padding_waste(probabilities, [2], costs)
    with pytest.raises(ValueError):
        padding_waste(probabilities, [3, 2], costs)


def test_histogram_truncates_above_the_cap_and_normalises():
    histogram = histogram_from_lengths([1, 3, 9, 12], max_length=10)
    assert histogram.shape == (10,)
    assert histogram[0] == pytest.approx(0.25)
    assert histogram[2] == pytest.approx(0.25)
    assert histogram[8] == pytest.approx(0.25)   # length 9 stays at 9
    assert histogram[9] == pytest.approx(0.25)   # length 12 is counted at the cap
    assert histogram.sum() == pytest.approx(1.0)


def test_histogram_honours_weights():
    histogram = histogram_from_lengths([2, 2], weights=[3.0, 1.0], max_length=4)
    assert histogram[1] == pytest.approx(1.0)


def test_timing_table_matches_brute_force():
    rng = random.Random(11)
    max_length = 8
    probabilities = [rng.random() for _ in range(max_length)]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    table = {(a, b): rng.uniform(0.5, 2.0)
             for a in range(0, max_length) for b in range(a + 1, max_length + 1)}

    def interval_time(a, b):
        return table[(a, b)]

    def brute_force_time():
        best = None
        for cuts in itertools.combinations(range(1, max_length), 2):
            boundaries = list(cuts) + [max_length]
            mass = 0.0
            lower = 0
            for upper in boundaries:
                mass += table[(lower, upper)] * sum(probabilities[lower:upper])
                lower = upper
            if best is None or mass < best:
                best = mass
        return best

    boundaries = optimal_boundaries_from_table(probabilities, 3, interval_time)
    mass = 0.0
    lower = 0
    for upper in boundaries:
        mass += table[(lower, upper)] * sum(probabilities[lower:upper])
        lower = upper
    assert mass == pytest.approx(brute_force_time(), abs=1e-12)


def test_timing_table_needs_finite_costs():
    probabilities = [0.5, 0.5]
    with pytest.raises(ValueError):
        optimal_boundaries_from_table(probabilities, 2, lambda a, b: float("inf"))


def test_plan_json_round_trips_through_the_trainer_loader(tmp_path):
    boundaries = {1: [8, 32, 64], 2: [16, 64]}
    batches = {1: [8, 4, 2], 2: [6, 3]}
    payload = plan_json(boundaries, batches)
    assert payload["1"][0] == {"max_length": 8, "batch_size": 8}

    path = tmp_path / "plan.json"
    dump_plan(str(path), boundaries, batches)
    plan = load_bucket_plan(str(path), resolution_ids=[1, 2])
    assert [bucket.max_length for bucket in plan.buckets_for(1)] == [8, 32, 64]
    assert [bucket.batch_size for bucket in plan.buckets_for(2)] == [6, 3]
    assert plan.bucket_for(1, 0)[1].max_length == 8
    assert plan.bucket_for(1, 33)[1].max_length == 64

    with open(path, encoding="utf-8") as handle:
        assert set(json.load(handle)) == {"1", "2"}


def test_plan_json_requires_matching_batch_sizes():
    with pytest.raises(ValueError):
        plan_json({1: [8, 32]}, {1: [8]})
    with pytest.raises(ValueError):
        plan_json({1: [8, 32]}, {})


def test_uniform_and_equal_mass_are_valid_partitions():
    probabilities = np.array([0.0, 0.0, 0.5, 0.5])
    assert uniform_boundaries(4, 3) == [1, 3, 4]
    assert equal_mass_boundaries(probabilities, 2)[-1] == 4
    for boundaries in (uniform_boundaries(4, 3), equal_mass_boundaries(probabilities, 2)):
        assert all(right > left for left, right in zip(boundaries, boundaries[1:]))
