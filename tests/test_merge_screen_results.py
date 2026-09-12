"""The merge must pick sizes that line the buckets up in time, and say so.

What is tested here is the rule the frozen plan uses: for a target micro-batch
time each bucket takes the largest measured candidate under it, the target is
swept against a simulation of the slowest rank, and a bucket with no finished
run keeps the plan's declared fallback.  The simulation's own numbers are
allowed to move with the seed; the choices it produces are not.
"""

import json

import pytest

from scripts.bench.merge_screen_results import (
    apply, choice_for, main, measured_candidates, read_bucket_mass, read_plan,
    scan_targets, share_by_bucket, simulate)


def write_cache(tmp_path, name, records):
    path = tmp_path / name
    path.write_text(json.dumps({"version": 1, "entries": {
        f"key{i}": {"records": [record]} for i, record in enumerate(records)}}))
    return str(path)


def record(resolution_id, bucket_index, batch_size, ms, status="ok"):
    return {"resolution_id": resolution_id, "bucket_index": bucket_index,
            "batch_size": batch_size, "ms_per_sample": ms, "status": status}


def plan_with(bounds):
    """A plan over the given bounds, fallback 16 for every bucket."""
    return {str(resolution_id): [{"max_length": bound, "batch_size": 16}
                                 for bound in bounds]
            for resolution_id, bounds in bounds.items()}


def sizes(plan, candidates, target=None):
    return {key: entry.batch_size
            for key, entry in choice_for(plan, candidates, target).items()}


# ---------------------------------------------------------------------------
# The draw mass, and what makes it usable.
# ---------------------------------------------------------------------------


def test_the_mass_file_is_read_and_turned_into_shares(tmp_path):
    path = tmp_path / "mass.json"
    path.write_text(json.dumps({"1": [2.0, 1.0], "2": [1.0]}))

    mass = read_bucket_mass(str(path))
    shares = share_by_bucket(mass, plan_with({1: [26, 47], 2: [46]}))

    assert mass == {"1": [2.0, 1.0], "2": [1.0]}
    assert shares[(1, 0)] == pytest.approx(0.5)
    assert shares[(1, 1)] == pytest.approx(0.25)
    assert shares[(2, 0)] == pytest.approx(0.25)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_a_mass_table_that_does_not_match_the_plan_is_refused(tmp_path):
    plan = plan_with({1: [26, 47], 2: [46]})

    with pytest.raises(ValueError, match="no entry for resolution ids"):
        share_by_bucket({"1": [1.0, 1.0]}, plan)
    with pytest.raises(ValueError, match="does not have"):
        share_by_bucket({"1": [1.0, 1.0], "2": [1.0], "3": [1.0]}, plan)
    with pytest.raises(ValueError, match="but the plan has"):
        share_by_bucket({"1": [1.0], "2": [1.0]}, plan)
    with pytest.raises(ValueError, match="sums to zero"):
        share_by_bucket({"1": [0.0, 0.0], "2": [0.0]}, plan)


def test_a_mass_file_that_is_not_a_mass_is_refused(tmp_path):
    path = tmp_path / "mass.json"

    path.write_text(json.dumps({"1": [1.0, "many"]}))
    with pytest.raises(ValueError, match="non-numeric"):
        read_bucket_mass(str(path))
    path.write_text(json.dumps({"1": [-1.0]}))
    with pytest.raises(ValueError, match="non-numeric or negative"):
        read_bucket_mass(str(path))
    path.write_text(json.dumps([1.0, 2.0]))
    with pytest.raises(ValueError, match="keyed by resolution"):
        read_bucket_mass(str(path))


# ---------------------------------------------------------------------------
# What counts as a measurement.
# ---------------------------------------------------------------------------


def test_candidates_pool_across_caches_and_keep_the_median(tmp_path):
    first = write_cache(tmp_path, "a.json", [record(1, 0, 64, 9.0),
                                             record(1, 0, 32, 9.5)])
    second = write_cache(tmp_path, "b.json", [record(1, 0, 64, 11.0),
                                              record(1, 0, 16, 10.0)])

    assert measured_candidates([first, second]) == {
        (1, 0): {64: 10.0, 32: 9.5, 16: 10.0}}


def test_failed_runs_are_not_measurements(tmp_path):
    cache = write_cache(tmp_path, "cache.json", [
        record(1, 0, 128, None, status="oom"),
        record(1, 0, 64, None, status="timed_out"),
        record(1, 0, 32, 11.0, status="missed"),
        record(1, 0, 16, 12.0),
    ])

    assert measured_candidates([cache]) == {(1, 0): {16: 12.0}}


# ---------------------------------------------------------------------------
# The rule: the largest candidate under the target.
# ---------------------------------------------------------------------------


def test_the_aligned_rule_takes_the_largest_candidate_under_the_target():
    plan = plan_with({1: [26]})
    # Micro-batch times: 16 -> 160 ms, 32 -> 304 ms, 64 -> 576 ms.
    candidates = {(1, 0): {16: 10.0, 32: 9.5, 64: 9.0}}

    assert sizes(plan, candidates, 400) == {(1, 0): 32}
    assert sizes(plan, candidates, 600) == {(1, 0): 64}
    # Nothing is under the target: the smallest measured size is the closest
    # a bucket can come to it, since nothing below it was ever measured.
    assert sizes(plan, candidates, 100) == {(1, 0): 16}
    # Without a target every bucket takes its fastest candidate.
    assert sizes(plan, candidates) == {(1, 0): 64}


def test_the_target_is_a_cap_on_micro_batch_time_not_on_size():
    plan = plan_with({1: [26]})
    # A larger candidate can be slower in total even though it is faster per
    # sample, so the cap is on batch_size * ms_per_sample.
    candidates = {(1, 0): {16: 10.0, 64: 9.0}}

    assert sizes(plan, candidates, 500) == {(1, 0): 16}


# ---------------------------------------------------------------------------
# Buckets without a finished run.
# ---------------------------------------------------------------------------


def test_an_unscreened_bucket_keeps_the_fallback_and_borrows_a_rate():
    plan = {str(1): [{"max_length": 26, "batch_size": 8},
                     {"max_length": 47, "batch_size": 4},
                     {"max_length": 90, "batch_size": 2}]}
    candidates = {(1, 0): {32: 9.0}}       # only bucket 0 was ever measured

    choice = choice_for(plan, candidates, 1000)

    assert choice[(1, 0)].batch_size == 32 and choice[(1, 0)].measured
    for index in (1, 2):
        assert choice[(1, index)].batch_size in (4, 2), "the plan's own fallback"
        assert not choice[(1, index)].measured
        assert choice[(1, index)].ms_per_sample == pytest.approx(9.0)

    merged, lines = apply(plan, choice, candidates)

    assert merged["1"] == [{"max_length": 26, "batch_size": 32},
                           {"max_length": 47, "batch_size": 4},
                           {"max_length": 90, "batch_size": 2}]
    assert "declared fallback" in lines[1] and "declared fallback" in lines[2]
    assert "declared fallback" not in lines[0]


def test_a_resolution_with_no_measurement_at_all_is_refused():
    plan = plan_with({1: [26], 2: [46]})
    candidates = {(1, 0): {16: 10.0}}

    with pytest.raises(ValueError, match="resolution 2 has no measured bucket"):
        choice_for(plan, candidates, 1000)


# ---------------------------------------------------------------------------
# The simulation, and the target sweep.
# ---------------------------------------------------------------------------


def test_one_bucket_has_no_slowest_rank_premium():
    plan = plan_with({1: [26]})
    candidates = {(1, 0): {16: 10.0}}

    stats = simulate(choice_for(plan, candidates, 1000), {(1, 0): 1.0},
                     ranks=8, accumulation=4, trials=50, seed=0)

    # Every rank draws the same bucket, so the slowest rank is the mean rank;
    # a step is 4 micro-batches of 16 samples at 160 ms each.
    assert stats.mean_step_ms == pytest.approx(640.0)
    assert stats.slowest_step_ms == pytest.approx(640.0)
    assert stats.mean_samples == pytest.approx(64.0)
    assert stats.premium == pytest.approx(0.0)
    assert stats.effective_ms_per_sample == pytest.approx(10.0)


def test_the_scan_prefers_the_target_that_pays_for_the_wait():
    # Two equally drawn buckets.  Bucket 0: 16 -> 160 ms, 64 -> 256 ms.
    # Bucket 1: 16 -> 320 ms, 32 -> 512 ms.  Under a 320 ms target bucket 0
    # moves up to 64 while bucket 1 stays at 16, so every step carries the
    # same 80 samples per rank behind a 320 ms longest path; under 512 ms the
    # step is longer but not enough richer to pay for itself.
    plan = plan_with({1: [26, 47]})
    candidates = {(1, 0): {16: 10.0, 64: 4.0}, (1, 1): {16: 20.0, 32: 16.0}}
    shares = {(1, 0): 0.5, (1, 1): 0.5}

    rows, best = scan_targets(plan, candidates, shares, [320, 512],
                              ranks=8, accumulation=16, trials=400, seed=3)

    assert [row.target for row in rows] == [None, 320, 512]
    assert best.target == 320
    assert best.stats.effective_ms_per_sample < rows[2].stats.effective_ms_per_sample
    assert best.stats.effective_ms_per_sample < rows[0].stats.effective_ms_per_sample
    assert best.stats.premium < rows[0].stats.premium
    assert sizes(plan, candidates, best.target) == {(1, 0): 64, (1, 1): 16}


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


def test_the_cli_refuses_to_run_without_a_bucket_mass(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_with({1: [26]})))

    with pytest.raises(SystemExit):
        main(["--plan", str(plan_path), "--cache", "cache.json", "--out", "out.json"])

    assert "--bucket-mass" in capsys.readouterr().err


def test_the_cli_writes_the_aligned_plan(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "1": [{"max_length": 26, "batch_size": 16},
              {"max_length": 47, "batch_size": 16}],
        "2": [{"max_length": 46, "batch_size": 16}]}))
    cache = write_cache(tmp_path, "cache.json", [
        record(1, 0, 16, 10.0), record(1, 0, 32, 12.0), record(1, 0, 64, 9.0),
        record(1, 1, 16, 12.0),
        record(2, 0, 16, 11.0),
    ])
    mass_path = tmp_path / "mass.json"
    mass_path.write_text(json.dumps({"1": [1.0, 1.0], "2": [1.0]}))
    out = tmp_path / "out.json"

    assert main(["--plan", str(plan_path), "--cache", cache,
                 "--bucket-mass", str(mass_path), "--out", str(out),
                 "--targets", "400", "--trials", "50"]) == 0

    merged = json.loads(out.read_text())
    # Bucket 0 takes 32 (384 ms, the largest under the 400 ms target); bucket
    # 1 has no candidate under it and keeps its smallest measurement; the
    # bounds are the plan's and are never touched.
    assert merged == {
        "1": [{"max_length": 26, "batch_size": 32},
              {"max_length": 47, "batch_size": 16}],
        "2": [{"max_length": 46, "batch_size": 16}]}
    assert read_plan(str(out))["1"][0]["max_length"] == 26
