"""The plan sizer must turn caption lengths and a measured sweep into a plan.

These tests cover the three parts the pipeline turns on: the least-squares
memory and time fits (do they recover coefficients the synthetic data was built
with?), the closed-form batch solve (does the predicted peak stay inside the
budget?), and the alignment pass (is it ever predicted worse than the memory
solution?).  The end-to-end test runs the CLI on synthetic sidecars and a
synthetic sweep and reads the plan back through the trainer's own loader.
"""

import json

import numpy as np
import pytest

from scripts.bench.plan_buckets import (
    AlignTarget,
    CalibrationPoint,
    MemoryModel,
    align_batches,
    bucket_boundaries,
    calibration_points,
    fit_memory_model,
    fit_time_model,
    length_histogram,
    load_sidecar_lengths,
    mean_emitted_batch,
    main,
    parse_image_tokens,
    predict_throughput,
    resolution_lengths,
    solve_batch_size,
)
from src.dataset.length_buckets import equal_mass_boundaries, padding_waste
from src.dataset.captions import CaptionPolicy, caption_probabilities_from_lengths
from src.dataset.length_metadata import RowLengthMetadata, sidecar_path
from src.dataset.mix import parse_dataset_mix
from src.pretrain.train import load_bucket_plan
from src.utils.prompt_contract import MAX_SEQUENCE_LENGTH

# The memory model the synthetic sweeps are built from: 9.5 GB of weights,
# optimizer state and kernels, 0.00101 GB per sample per padded token.
M0 = 9.5
M1 = 0.00101
T0 = 3.0
T1 = 0.0004
T2 = 0.0000008


def sweep_points(*, img_tokens=256, latent_hw=(32, 32), txt_lens=(128, 512),
                 micros=(8, 16)) -> list:
    """A grid of synthetic calibration points built from the module constants."""
    points = []
    for txt_len in txt_lens:
        for micro in micros:
            total = img_tokens + txt_len
            x = micro * total
            points.append(CalibrationPoint(
                latent_hw=latent_hw, img_tokens=img_tokens, txt_len=txt_len,
                micro_batch=micro, peak_mem_gb=M0 + M1 * x,
                ms_per_step=T0 + micro * (T1 * total + T2 * total * total)))
    return points


def write_sidecar(root, rows):
    """Write one dataset's companion length file; ``rows`` is (id, [lengths])."""
    offsets = [0]
    lengths = []
    for _, row_lengths in rows:
        lengths.extend(row_lengths)
        offsets.append(offsets[-1] + len(row_lengths))
    metadata = RowLengthMetadata(
        resolution_ids=np.array([resolution for resolution, _ in rows], dtype=np.int64),
        caption_offsets=np.array(offsets, dtype=np.int64),
        prompt_lengths=np.array(lengths, dtype=np.int64),
    )
    root.mkdir(parents=True, exist_ok=True)
    metadata.save(sidecar_path(str(root)))
    return root


def write_sweep(path, *, shapes=None):
    """A calibration file with explicit measured shapes."""
    shapes = shapes or {
        (32, 32): {"img_tokens": 256, "txt_lens": (128, 512), "micros": (8, 16)},
        (80, 80): {"img_tokens": 1600, "txt_lens": (128, 512), "micros": (4, 8)},
    }
    results = {}
    for (height, width), spec in shapes.items():
        latent = f"{height}x{width}"
        img_tokens = spec["img_tokens"]
        for txt_len in spec["txt_lens"]:
            for micro in spec["micros"]:
                total = img_tokens + txt_len
                x = micro * total
                results.setdefault(latent, {}).setdefault(str(txt_len), {})[str(micro)] = {
                    "peak_mem_gb": M0 + M1 * x,
                    "ms_per_step": T0 + micro * (T1 * total + T2 * total * total),
                    "samples_per_sec": 1000.0,
                    "latent_hw": [height, width],
                    "img_tokens": img_tokens,
                    "txt_seq": txt_len,
                    "micro_batch": micro,
                }
    path.write_text(json.dumps({"arch": {}, "results": results}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fits.
# ---------------------------------------------------------------------------


def costly_points():
    """A sweep whose memory is flat and then jumps — no line fits it well.

    Four points with peaks of 12.0/12.2/12.6/17.0 GB at batch*L of
    2000/6000/16000/40000: the line through them has R^2 0.94, the case the
    quadratic fallback exists for.
    """
    peaks = (12.0, 12.2, 12.6, 17.0)
    points = []
    for index, peak in enumerate(peaks):
        micro = 2 * 2 ** index
        total = 1000 + 500 * index
        points.append(CalibrationPoint(latent_hw=(80, 80), img_tokens=total // 2,
                                       txt_len=total - total // 2, micro_batch=micro,
                                       peak_mem_gb=peak, ms_per_step=10.0 * micro))
    return points


def quadratic_model():
    """A well-behaved quadratic memory model, built rather than fitted."""
    return MemoryModel(m0=M0, m1=M1, m2=1e-6, quadratic=True, r_squared=1.0,
                       max_residual_gb=0.0, points=8, rank=3, columns=3,
                       tokens_min=384.0, tokens_max=3648.0)


def test_linear_fit_recovers_the_coefficients_it_was_built_from():
    model = fit_memory_model(sweep_points())
    assert model.quadratic is False
    assert model.r_squared == pytest.approx(1.0)
    assert model.max_residual_gb == pytest.approx(0.0, abs=1e-9)
    assert model.m0 == pytest.approx(M0)
    assert model.m1 == pytest.approx(M1)
    assert model.identifiable


def test_quadratic_term_is_added_only_when_the_linear_fit_is_poor():
    linear = fit_memory_model(sweep_points())
    assert linear.quadratic is False
    assert linear.r_squared > 0.95
    curved = fit_memory_model(costly_points())
    assert curved.quadratic is True
    assert curved.r_squared > 0.99
    assert curved.max_residual_gb < 0.5


def test_time_fit_recovers_the_coefficients_it_was_built_from():
    model = fit_time_model(sweep_points())
    assert model.r_squared == pytest.approx(1.0)
    assert model.t0 == pytest.approx(T0)
    assert model.t1 == pytest.approx(T1)
    assert model.t2 == pytest.approx(T2)


def test_a_thin_sweep_is_reported_as_not_identifiable():
    # One shape, one micro-batch: a straight line is the most the fit can claim.
    points = sweep_points(txt_lens=(128,), micros=(8, 8))
    model = fit_memory_model(points)
    assert not model.identifiable


# ---------------------------------------------------------------------------
# The batch solve.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_tokens", [282, 1252, 2304, 2304 + 1600])
def test_solved_batch_is_the_largest_one_inside_the_budget(total_tokens):
    model = fit_memory_model(sweep_points())
    budget = 42.24
    size = solve_batch_size(model, budget_gb=budget, total_tokens=total_tokens,
                            min_batch=2, max_batch=128)
    assert model.predict_gb(size, total_tokens) <= budget
    if size < 128:
        assert model.predict_gb(size + 1, total_tokens) > budget


def test_solved_batch_is_clamped_to_the_configured_range():
    model = fit_memory_model(sweep_points())
    assert solve_batch_size(model, budget_gb=42.24, total_tokens=282,
                            min_batch=2, max_batch=10) == 10
    # A 640p bucket whose caption fills the cap needs 3648-token sequences: at a
    # 24 GB budget the fit leaves room for 3 samples, so the smallest allowed
    # batch is what comes back — and the caller can see it does not fit.
    total = 1600 + MAX_SEQUENCE_LENGTH
    size = solve_batch_size(model, budget_gb=24.0, total_tokens=total,
                            min_batch=4, max_batch=128)
    assert size == 4
    assert model.predict_gb(size, total) > 24.0


def test_solved_batch_uses_the_quadratic_root_when_the_fit_has_curvature():
    model = quadratic_model()
    budget = 42.24
    total = 1500
    size = solve_batch_size(model, budget_gb=budget, total_tokens=total,
                            min_batch=1, max_batch=128)
    assert model.predict_gb(size, total) <= budget
    if size < 128:
        assert model.predict_gb(size + 1, total) > budget
    # Curvature only ever pulls the size below the linear answer.
    linear = fit_memory_model(sweep_points())
    assert size < solve_batch_size(linear, budget_gb=budget, total_tokens=total,
                                   min_batch=1, max_batch=128)


def test_a_flat_memory_fit_is_refused_rather_than_sized_from():
    flat = sweep_points()[0]
    degenerate = [flat, CalibrationPoint(latent_hw=flat.latent_hw,
                                         img_tokens=flat.img_tokens,
                                         txt_len=flat.txt_len, micro_batch=16,
                                         peak_mem_gb=flat.peak_mem_gb,
                                         ms_per_step=flat.ms_per_step)]
    model = fit_memory_model(degenerate)
    with pytest.raises(ValueError, match="no meaningful change"):
        solve_batch_size(model, budget_gb=42.24, total_tokens=1252,
                         min_batch=2, max_batch=128)


# ---------------------------------------------------------------------------
# Alignment.
# ---------------------------------------------------------------------------


def alignment_fixture(max_batches, totals, weights=None):
    model = fit_time_model(sweep_points())
    weights = weights or [1.0 / len(max_batches)] * len(max_batches)
    return [AlignTarget(key=f"bucket{index}", total_tokens=tokens,
                        max_batch_size=size, weight=weight, model=model)
            for index, (size, tokens, weight) in
            enumerate(zip(max_batches, totals, weights))]


def test_alignment_gives_up_size_in_the_slow_bucket_to_shorten_the_step():
    # A common short bucket that can hold 108 samples, and a rare long one that
    # can only hold 14 but costs 75 ms against the short bucket's 24 ms.  The
    # step waits on the long bucket, so shrinking it to a few samples buys the
    # whole step, at the price of 10% of the draws.
    targets = alignment_fixture([108, 14], [300, 2300], weights=[0.9, 0.1])
    before = predict_throughput(targets, [108, 14])
    outcome = align_batches(targets, min_batch=1, max_batch=128, gain_threshold=0.03)
    assert outcome.aligned is True
    assert outcome.target_ms is not None
    assert 100 <= outcome.sizes[0] <= 108
    assert outcome.sizes[1] < 14
    assert outcome.gain > 0.03
    assert outcome.throughput_after > before
    assert outcome.target_ms <= targets[0].ms(108)


def test_throughput_weights_the_samples_and_charges_the_slowest_bucket():
    targets = alignment_fixture([10, 20], [1000, 1500], weights=[0.75, 0.25])
    sizes = [10, 20]
    expected = 1 / (0.75 / 10 + 0.25 / 20) / max(targets[0].ms(10), targets[1].ms(20))
    assert predict_throughput(targets, sizes) == pytest.approx(expected)
    # No draws at all is not a throughput.
    assert predict_throughput([], []) == 0.0


def test_alignment_is_never_predicted_worse_than_the_memory_solution():
    for max_batches, totals in (
        ([64, 4], [300, 2000]),
        ([16, 16, 16], [300, 600, 1200]),
        ([128, 2, 8], [280, 2300, 900]),
        ([8, 8], [1000, 1000]),
    ):
        targets = alignment_fixture(max_batches, totals)
        outcome = align_batches(targets, min_batch=2, max_batch=128,
                                gain_threshold=0.0)
        assert all(1 <= size <= cap for size, cap in zip(outcome.sizes, max_batches))
        assert predict_throughput(targets, outcome.sizes) >= \
            outcome.throughput_before * (1 - 1e-12)


def test_alignment_is_dropped_when_the_gain_is_below_the_threshold():
    # Two buckets of the same shape cost the same per sample, so no target time
    # can buy anything: the memory sizes must survive untouched.
    targets = alignment_fixture([8, 8], [1000, 1000])
    outcome = align_batches(targets, min_batch=2, max_batch=128,
                            gain_threshold=0.03)
    assert outcome.aligned is False
    assert list(outcome.sizes) == [8, 8]
    assert outcome.gain == pytest.approx(0.0)
    assert "kept" in outcome.note


def test_alignment_weights_decide_which_bucket_is_worth_shrinking():
    # The same geometry under two weightings: the bucket the alignment shrinks
    # is the one whose micro-batch time sets the wait, and it never costs the
    # weighted mean more samples than the step time it saves.
    for weights in ([0.9, 0.1], [0.5, 0.5], [0.1, 0.9]):
        targets = alignment_fixture([108, 14], [300, 2300], weights=weights)
        outcome = align_batches(targets, min_batch=1, max_batch=128,
                                gain_threshold=0.03)
        assert predict_throughput(targets, outcome.sizes) >= \
            outcome.throughput_before * (1 - 1e-12)


# ---------------------------------------------------------------------------
# Bounds and the corpus side.
# ---------------------------------------------------------------------------


def _boundary_time_model():
    return fit_time_model(sweep_points())


def test_boundaries_minimise_padding_and_end_at_the_cap():
    values = np.arange(1, 1001)
    weights = np.ones(values.size)
    probabilities = length_histogram(values, weights, 1000)
    time_model = _boundary_time_model()
    boundaries = bucket_boundaries(probabilities, 10, 256, time_model)
    assert boundaries[-1] == 1000
    assert all(right > left for left, right in zip(boundaries, boundaries[1:]))

    # The DP result never pads more than an equal-mass cut of the same mass.
    lengths = np.arange(1, 1001, dtype=np.float64)
    costs = time_model.t1 * (256 + lengths) + time_model.t2 * (256 + lengths) ** 2
    equal_mass = equal_mass_boundaries(probabilities, 10)
    assert padding_waste(probabilities, boundaries, costs) <= \
        padding_waste(probabilities, equal_mass, costs) + 1e-9


def test_boundaries_cut_the_long_tail_finely():
    # 95% of mass below length 100 and a long tail to the cap: an equal-mass
    # cut parks the whole tail in one bucket padded to the cap, while the
    # compute-optimal cut must spend several bounds inside the tail.
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.integers(1, 100, 9500),
                             rng.integers(100, 2048, 500)])
    probabilities = length_histogram(values, np.ones(values.size),
                                     MAX_SEQUENCE_LENGTH)
    boundaries = bucket_boundaries(probabilities, 10, 256,
                                   _boundary_time_model())
    tail_bounds = [bound for bound in boundaries if bound > 100]
    assert len(tail_bounds) >= 4


def test_boundaries_cap_the_histogram_at_the_contract_maximum():
    # Captions longer than the cap are counted at the cap, as training truncates
    # them, so the last bound is the loader's own maximum.
    values = np.array([10, 20, 30, 4000])
    probabilities = length_histogram(values, np.ones(4), MAX_SEQUENCE_LENGTH)
    boundaries = bucket_boundaries(probabilities, 3, 256, _boundary_time_model())
    assert boundaries[-1] == MAX_SEQUENCE_LENGTH
    assert all(right > left for left, right in zip(boundaries, boundaries[1:]))


def test_sidecar_lengths_weight_captions_by_dataset_and_rows(tmp_path):
    first = write_sidecar(tmp_path / "a", [(1, [10, 20]), (1, [30])])
    second = write_sidecar(tmp_path / "b", [(2, [40, 50])])
    entries = parse_dataset_mix(f"{first}:0.75 {second}:0.25")
    pooled = resolution_lengths(load_sidecar_lengths(entries))

    assert sorted(pooled) == [1, 2]
    assert pooled[1].captions == 3
    assert pooled[2].captions == 2
    # Each row gets 0.75/2, divided among its own captions.
    assert pooled[1].weights == pytest.approx([0.75 / 4, 0.75 / 4, 0.75 / 2])
    assert pooled[1].weights.sum() == pytest.approx(0.75)
    assert pooled[2].weights.sum() == pytest.approx(0.25)


def test_mean_batch_matches_queue_emissions():
    assert mean_emitted_batch([0.9, 0.1], [100, 10]) == pytest.approx(1000 / 19)
    assert mean_emitted_batch([9, 1], [100, 10]) == pytest.approx(1000 / 19)
    assert mean_emitted_batch([1, 0], [100, 10]) == 100


def test_alignment_preserves_required_mean_batch():
    targets = alignment_fixture([108, 14], [300, 2300], weights=[.9, .1])
    result = align_batches(targets, min_batch=1, max_batch=128, min_mean_batch=40)
    assert mean_emitted_batch([.9, .1], result.sizes) >= 40
    assert all(b <= t.max_batch_size for b, t in zip(result.sizes, targets))
    assert result.throughput_after >= result.throughput_before


def test_alignment_rejects_infeasible_mean_batch():
    targets = alignment_fixture([10, 20], [300, 2300])
    with pytest.raises(ValueError, match="cannot meet"):
        align_batches(targets, min_batch=1, max_batch=128, min_mean_batch=30)


def test_mean_constraint_uses_real_draw_shares_not_alignment_weights():
    targets = alignment_fixture([108, 14], [300, 2300], weights=[.5, .5])
    result = align_batches(targets, min_batch=1, max_batch=128,
                           min_mean_batch=50, sample_shares=[.9, .1])
    assert mean_emitted_batch([.9, .1], result.sizes) >= 50


@pytest.mark.parametrize("floor", [-1, float("nan"), float("inf")])
def test_alignment_rejects_invalid_mean_target(floor):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        align_batches(alignment_fixture([10], [300]), min_batch=1,
                      max_batch=128, min_mean_batch=floor)


@pytest.mark.parametrize("interval", [(0, .75), (.75, .95), (.95, 1)])
def test_beta_weights_use_stage_probabilities_and_preserve_row_mass(tmp_path, interval):
    root = write_sidecar(tmp_path / "a", [(1, [64, 512]), (2, [300, 1024, 2048])])
    policy = CaptionPolicy(kind="beta")
    records = load_sidecar_lengths(parse_dataset_mix(str(root)), policy=policy,
                                   progress_start=interval[0], progress_end=interval[1])
    pooled = resolution_lengths(records)
    points = interval[0] + (np.arange(8) + .5) / 8 * (interval[1] - interval[0])
    for res, lengths in [(1, [64, 512]), (2, [300, 1024, 2048])]:
        expected = np.mean([caption_probabilities_from_lengths(lengths, policy.beta(p))
                            for p in points], axis=0) / 2
        assert pooled[res].weights == pytest.approx(expected)
        assert pooled[res].weights.sum() == pytest.approx(.5)


def test_aspect_draw_mass_survives_histogram_normalization(tmp_path):
    from scripts.bench.plan_buckets import draft_plan, fit_models
    root = write_sidecar(tmp_path / "a", [(1, [10]), (1, [20]), (2, [30])])
    pooled = resolution_lengths(load_sidecar_lengths(parse_dataset_mix(str(root))))
    memory, time = fit_models(sweep_points(txt_lens=(128, 256, 512)))
    draft = draft_plan(pooled, {1: 256, 2: 256}, buckets=2, cap=2048,
                       memory_table=memory, time_table=time, budget_gb=40,
                       min_batch=1, max_batch=128, weight_mode="count")
    assert sum(b.draw_share for b in draft.drafts if b.resolution_id == 1) == pytest.approx(2/3)
    assert sum(b.draw_share for b in draft.drafts if b.resolution_id == 2) == pytest.approx(1/3)


def test_stationary_policy_keeps_full_run_probabilities_in_later_stage(tmp_path):
    from src.dataset.captions import average_caption_probabilities
    root = write_sidecar(tmp_path / "a", [(1, [64, 512, 2048])])
    policy = CaptionPolicy(kind="beta", schedule="stationary")
    records = load_sidecar_lengths(parse_dataset_mix(str(root)), policy=policy,
                                   progress_start=.95, progress_end=1)
    assert records[0].weights[1] == pytest.approx(
        average_caption_probabilities([64, 512, 2048], policy))


@pytest.mark.parametrize("start,end,grid", [(-.1, 1, 8), (.9, .2, 8), (0, 1, 0)])
def test_invalid_planning_progress_is_rejected(start, end, grid):
    with pytest.raises(ValueError, match="progress"):
        load_sidecar_lengths([], progress_start=start, progress_end=end, progress_grid=grid)


def test_image_tokens_accepts_one_number_or_a_mapping():
    assert parse_image_tokens("256", [1, 2]) == {1: 256, 2: 256}
    assert parse_image_tokens('{"1": 256, "2": 252}', [1, 2]) == {1: 256, 2: 252}
    with pytest.raises(ValueError, match="missing resolution ids"):
        parse_image_tokens('{"1": 256}', [1, 2])
    with pytest.raises(ValueError, match="positive"):
        parse_image_tokens("0", [1])


# ---------------------------------------------------------------------------
# Calibration loading.
# ---------------------------------------------------------------------------


def test_calibration_points_read_the_sweeps_own_fields(tmp_path):
    payload = json.loads(write_sweep(tmp_path / "sweep.json").read_text())
    points = calibration_points(payload)
    assert len(points) == 8
    squared = [point for point in points if point.img_tokens == 256]
    assert all(point.latent_hw == (32, 32) for point in squared)
    assert {(point.txt_len, point.micro_batch) for point in squared} == {
        (128, 8), (128, 16), (512, 8), (512, 16)}


def test_calibration_points_skip_oom_corners_and_read_flat_lists():
    payload = {"results": [
        {"latent_hw": [80, 80], "img_tokens": 1600, "txt_seq": 256, "micro_batch": 8,
         "peak_mem_gb": 21.7, "ms_per_step": 533.0},
        {"latent_hw": [80, 80], "txt_seq": 512, "micro_batch": 32,
         "error": "cuda_oom"},
    ]}
    points = calibration_points(payload)
    assert len(points) == 1
    assert (points[0].img_tokens, points[0].txt_len, points[0].micro_batch) == \
        (1600, 256, 8)


def test_calibration_points_read_the_older_nested_shape():
    payload = {"results": {"80": {"8": {"128": {
        "peak_vram_gb": 21.7, "ms_per_step": 533.0, "img_tokens": 1600}}}}}
    points = calibration_points(payload)
    assert len(points) == 1
    assert points[0].latent_hw == (80, 80)
    assert (points[0].img_tokens, points[0].txt_len, points[0].micro_batch) == \
        (1600, 128, 8)


def test_calibration_points_are_refused_when_there_are_none():
    with pytest.raises(ValueError, match="no calibration points"):
        calibration_points({"results": {"80x80": {"128": {"8": {"error": "cuda_oom"}}}}})


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------


def plan_file(tmp_path, *, extra_args=(), buckets="10", budget="42.24",
              image_tokens='{"1": 256, "2": 1600}') -> str:
    short = write_sidecar(tmp_path / "short", [
        (1, [12, 18, 24, 30, 40, 64, 96, 128]),
        (2, [900, 1200, 1500]),
    ])
    long = write_sidecar(tmp_path / "long", [
        (1, [50, 80, 200, 400, 700, 1500]),
        (2, [300, 1500]),
    ])
    sweep = write_sweep(tmp_path / "sweep.json")
    out = tmp_path / "plan.json"
    code = main(["--dataset", f"{short}:0.7", "--dataset", f"{long}:0.3",
                 "--calibration", str(sweep), "--image-tokens", image_tokens,
                 "--buckets", buckets, "--vram-budget-gb", budget,
                 "--out", str(out), *extra_args])
    assert code == 0
    return str(out)


def test_plan_round_trips_through_the_trainer_loader(tmp_path):
    path = plan_file(tmp_path)
    payload = json.loads(open(path, encoding="utf-8").read())
    # No extra top-level keys: the loader reads every one of them as a resolution.
    assert set(payload) == {"1", "2"}
    plan = load_bucket_plan(path, resolution_ids=[1, 2])
    for resolution_id in (1, 2):
        buckets = plan.buckets_for(resolution_id)
        assert len(buckets) == 10
        bounds = [bucket.max_length for bucket in buckets]
        assert all(right > left for left, right in zip(bounds, bounds[1:]))
        assert bounds[-1] == MAX_SEQUENCE_LENGTH
        assert all(bucket.batch_size >= 2 for bucket in buckets)


def test_every_planned_batch_is_predicted_to_fit_the_budget(tmp_path):
    path = plan_file(tmp_path, budget="20", extra_args=("--no-align",))
    model = fit_memory_model(sweep_points())
    payload = json.loads(open(path, encoding="utf-8").read())
    image_tokens = {1: 256, 2: 1600}
    for resolution, buckets in payload.items():
        for bucket in buckets:
            total = image_tokens[int(resolution)] + bucket["max_length"]
            assert model.predict_gb(bucket["batch_size"], total) <= 20
            # The size is the largest one that fits under the budget.
            assert bucket["batch_size"] == 128 or \
                model.predict_gb(bucket["batch_size"] + 1, total) > 20


def test_no_align_keeps_the_memory_solution(tmp_path):
    aligned = json.loads(open(plan_file(tmp_path / "aligned"), encoding="utf-8").read())
    plain = json.loads(open(plan_file(tmp_path / "plain", extra_args=("--no-align",)),
                            encoding="utf-8").read())
    for resolution in plain:
        for row, other in zip(plain[resolution], aligned[resolution]):
            assert row["max_length"] == other["max_length"]
            assert row["batch_size"] >= other["batch_size"]


def test_report_records_the_fits_the_bounds_and_the_decision(tmp_path):
    path = plan_file(tmp_path)
    report = (tmp_path / "plan.json.report.md").read_text(encoding="utf-8")
    for expected in ("# Bucket plan", "## Inputs", "## Calibration", "## Memory model",
                     "## Time model", "## Buckets", "## Time alignment", "## Limits",
                     "R^2", "peak =", "L^2", "draw share", "predicted GB"):
        assert expected in report
    assert "minimise padded compute" in report
    assert "alignment" in report


def test_a_shape_the_sweep_never_measured_falls_back_to_the_pooled_fit(tmp_path):
    short = write_sidecar(tmp_path / "short", [(1, [12, 24, 48, 96, 200])])
    sweep = write_sweep(tmp_path / "sweep.json")
    out = tmp_path / "plan.json"
    assert main(["--dataset", str(short), "--calibration", str(sweep),
                 "--image-tokens", "640", "--buckets", "3",
                 "--out", str(out)]) == 0
    report = (tmp_path / "plan.json.report.md").read_text(encoding="utf-8")
    assert "pooled fit for memory and the pooled fit for time" in report
    # 640 image tokens were never measured, so the report must say the sizes
    # extrapolate rather than presenting them as measured.
    assert "outside the calibrated range" in report


def test_a_sweep_that_cannot_identify_time_skips_the_alignment(tmp_path, capsys):
    # One micro-batch size at two text lengths cannot separate the linear from the
    # quadratic term of the time model, so the alignment must not run on it.
    short = write_sidecar(tmp_path / "short", [(1, [12, 24, 48, 96, 200])])
    sweep = write_sweep(tmp_path / "sweep.json", shapes={
        (32, 32): {"img_tokens": 256, "txt_lens": (128, 512), "micros": (8,)}})
    out = tmp_path / "plan.json"
    assert main(["--dataset", str(short), "--calibration", str(sweep),
                 "--image-tokens", "256", "--buckets", "3", "--out", str(out)]) == 0
    assert "alignment pass is skipped" in capsys.readouterr().err
    report = (tmp_path / "plan.json.report.md").read_text(encoding="utf-8")
    assert "Switched off" in report


def test_a_missing_sidecar_is_reported_not_guessed(tmp_path, capsys):
    sweep = write_sweep(tmp_path / "sweep.json")
    out = tmp_path / "plan.json"
    code = main(["--dataset", str(tmp_path / "absent"), "--calibration", str(sweep),
                 "--image-tokens", "256", "--out", str(out)])
    assert code == 2
    assert "no caption-length sidecar" in capsys.readouterr().err
    assert not out.exists()
