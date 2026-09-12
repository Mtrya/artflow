"""Expected caption exposure per selection policy.

The simulation's numbers are only useful if they describe the trainer's own
draw, so these tests pin the three places where it could drift from it:

* the vectorised selector functions, caption by caption, against
  ``caption_probabilities_from_lengths`` / ``caption_probabilities_from_token_counts``
  evaluated one row at a time;
* the stationary arm against the construction the training sampler builds;
* the draw model itself -- a dataset by mix weight, a row uniformly inside it,
  one caption inside the row -- including rows that cannot be drawn.

They also pin the one property the stationary control exists for: its total
exposure equals the reference schedule's when the numerical integration is
refined, which is the check that would catch an averaging or weighting mistake
in that arm.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from scripts.plan_buckets import load_sidecars, parse_mix_spec
from scripts.simulate_exposure import (
    BANDS,
    DatasetView,
    Settings,
    arm_curve,
    band_masses,
    beta_probabilities,
    build_arms,
    legacy_probabilities,
    main,
    progress_grid,
    stationary_exposures,
    stationary_probabilities,
    trapezoid_weights,
)
from src.dataset.captions import (
    caption_probabilities_from_lengths,
    caption_probabilities_from_token_counts,
)
from src.dataset.length_metadata import RowLengthMetadata, sidecar_path
from src.dataset.mix import DatasetEntry

SHORT_THRESHOLD = 256
RESERVE = 0.20


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


def mix(*entries):
    return " ".join(f"{path}:{weight}" for path, weight in entries)


def view_of(rows, weight=1.0, alias="ds"):
    """A prepared dataset view for a hand-written list of (resolution, lengths)."""
    root = Path(alias)
    metadata = RowLengthMetadata(
        resolution_ids=np.array([resolution for resolution, _ in rows], dtype=np.int64),
        caption_offsets=np.array(
            [0] + list(np.cumsum([len(row) for _, row in rows])), dtype=np.int64
        ),
        prompt_lengths=np.array([length for _, row in rows for length in row], dtype=np.int64),
    )
    entry = DatasetEntry(path=root, weight=weight, alias=root.name)
    return DatasetView.from_metadata(entry, metadata, SHORT_THRESHOLD)


def band_of(length):
    if length >= 1280:
        return 4
    if length >= 896:
        return 3
    if length >= 512:
        return 2
    if length >= 256:
        return 1
    return 0


def masses_from_probabilities(lengths, probabilities):
    """Band masses of one row whose draw weight is one."""
    masses = np.zeros(len(BANDS))
    for length, probability in zip(lengths, probabilities):
        masses[band_of(length)] += probability
    return masses


def settings_for_run(total_draws, **overrides):
    values = dict(
        total_draws=total_draws,
        short_reserve=RESERVE,
        short_threshold=SHORT_THRESHOLD,
    )
    values.update(overrides)
    return Settings(**values)


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------


def test_end_to_end_writes_a_json_payload_and_a_report(tmp_path):
    first = write_sidecar(tmp_path / "painting",
                          [(1, [120, 400]), (2, [200, 200, 1400]), (1, [])])
    second = write_sidecar(tmp_path / "photo", [(2, [300, 900]), (1, [1500])])
    out = tmp_path / "exposure.json"

    code = main(["--mix", mix((first, 0.7), (second, 0.3)),
                 "--steps", "10", "--samples-per-step", "4", "--out", str(out)])

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["inputs"]["total_draws"] == 40
    assert payload["inputs"]["size_source"] == "--steps x --samples-per-step"
    assert [dataset["alias"] for dataset in payload["inputs"]["datasets"]] == ["painting", "photo"]
    # Rows without captions are counted, not drawn.
    assert payload["inputs"]["datasets"][0]["rows"] == 3
    assert payload["inputs"]["datasets"][0]["rows_with_captions"] == 2
    assert [arm["key"] for arm in payload["arms"]] == ["A", "B", "C", "D", "E"]

    band_names = [name for name, _, _ in BANDS]
    for arm in payload["arms"]:
        shares = [arm["bands"][name]["fraction"] for name in band_names]
        assert sum(shares) == pytest.approx(1.0, abs=1e-6)
        for name, share in zip(band_names, shares):
            assert arm["bands"][name]["count"] == pytest.approx(share * 40, abs=1e-3)
        assert arm["long_bands"]["ge256"]["fraction"] == pytest.approx(
            sum(shares[1:]), abs=1e-9)
        assert arm["long_bands"]["ge512"]["fraction"] == pytest.approx(
            sum(shares[2:]), abs=1e-9)
        assert len(arm["cumulative_counts"][band_names[0]]) == 11
    # Ten unordered pairs of five arms.
    assert len(payload["pairwise"]) == 10
    assert payload["pairwise"][0]["arms"] == ["A", "B"]

    report = (tmp_path / "exposure.json.report.md").read_text(encoding="utf-8")
    for section in ("# Caption-length exposure by selection policy", "## Inputs",
                    "## Method", "## Expected exposure per band",
                    "## Pairwise differences in long-caption draws",
                    "## Arm D versus arm B", "## Integration error",
                    "## Exposure over progress", "## Run-to-run noise", "## Limitations"):
        assert section in report
    # The report must not present the numbers as an exact integration.
    assert "numerical integration" in report
    assert "exposure-matched" in report


# ---------------------------------------------------------------------------
# Alignment with the selector functions used in training.
# ---------------------------------------------------------------------------


def test_two_caption_row_matches_the_selector_functions():
    """The tool's per-progress shares equal the selectors' own outputs."""
    lengths = [150, 700]
    view = view_of([(1, lengths)])
    settings = settings_for_run(1000)
    arms = {arm.key: arm for arm in build_arms(settings)}
    # One grid point at progress 0.5: stage = beta = 0 for the beta arms.
    points = np.array([0.5, 1.0])

    for beta in (-1.0, -0.5, 0.0, 0.75, 1.0):
        direct = caption_probabilities_from_lengths(lengths, beta, RESERVE, SHORT_THRESHOLD)
        vectorised = beta_probabilities(view, beta, RESERVE, SHORT_THRESHOLD)
        assert vectorised == pytest.approx(direct, rel=1e-12)
        assert masses_from_probabilities(lengths, direct) == pytest.approx(
            masses_from_probabilities(lengths, vectorised), abs=1e-12)

    for stage in (0.0, 0.25, 0.5, 1.0):
        direct = caption_probabilities_from_token_counts(lengths, stage)
        vectorised = legacy_probabilities(view, stage)
        assert vectorised == pytest.approx(direct, rel=1e-9)

    # Arm A at progress 0.5 drives the legacy selector with stage 0.5; arm B
    # drives the beta selector with beta 0.
    legacy_direct = caption_probabilities_from_token_counts(lengths, 0.5)
    legacy_curve = arm_curve(arms["A"], [view], points, settings)
    assert legacy_curve[0] == pytest.approx(
        masses_from_probabilities(lengths, legacy_direct), abs=1e-12)

    beta_direct = caption_probabilities_from_lengths(lengths, 0.0, RESERVE, SHORT_THRESHOLD)
    beta_curve = arm_curve(arms["B"], [view], points, settings)
    assert beta_curve[0] == pytest.approx(
        masses_from_probabilities(lengths, beta_direct), abs=1e-12)
    # The reserve protects the short caption: 0.2 alone plus an even share of
    # the rest, so the row is not drawn evenly between the two captions.
    assert beta_curve[0][band_of(150)] == pytest.approx(0.6)
    assert beta_curve[0][band_of(700)] == pytest.approx(0.4)


def test_vectorised_selectors_match_the_row_functions_on_a_mixed_corpus():
    """Rows with and without a short caption in one dataset are both exact."""
    rows = [(1, [100, 400, 900]), (2, [250]), (1, [1280, 60]), (2, [512, 513, 514, 515]),
            (1, [2000, 300])]
    view = view_of(rows, weight=0.6)

    for beta in (-1.5, -0.3, 0.0, 0.4, 1.5):
        vectorised = beta_probabilities(view, beta, RESERVE, SHORT_THRESHOLD)
        start = 0
        for _, row_lengths in rows:
            direct = caption_probabilities_from_lengths(
                row_lengths, beta, RESERVE, SHORT_THRESHOLD)
            assert vectorised[start:start + len(row_lengths)] == pytest.approx(
                direct, rel=1e-12)
            assert sum(vectorised[start:start + len(row_lengths)]) == pytest.approx(1.0)
            start += len(row_lengths)

    for stage in (0.0, 0.5, 1.0):
        vectorised = legacy_probabilities(view, stage)
        start = 0
        for _, row_lengths in rows:
            direct = caption_probabilities_from_token_counts(row_lengths, stage)
            assert vectorised[start:start + len(row_lengths)] == pytest.approx(
                direct, rel=1e-9)
            start += len(row_lengths)


# ---------------------------------------------------------------------------
# The stationary control.
# ---------------------------------------------------------------------------


def test_stationary_arm_total_matches_the_reference_when_the_grid_is_refined():
    """D exists to match B's total; refining the grids must show that."""
    rows = [(1, [150, 700]), (2, [300, 1200]), (1, [280, 500, 900]), (2, [80]),
            (1, [900, 1200, 1500]), (2, [60, 1000])]
    view = view_of(rows)
    arms = {arm.key: arm for arm in build_arms(settings_for_run(1))}

    def reference_mass(points):
        grid = progress_grid(points)
        weights = trapezoid_weights(grid)
        return weights @ arm_curve(arms["B"], [view], grid, settings_for_run(1))

    def matched_mass(points):
        return stationary_exposures([view], arms["D"].policy, points)[1]

    coarse = float(np.max(np.abs(matched_mass(11) - reference_mass(11))))
    fine = float(np.max(np.abs(matched_mass(2001) - reference_mass(2001))))
    assert fine < 1e-4
    assert fine < coarse / 10.0
    # The totals, not just the bands, agree once both sides are refined.
    assert float(matched_mass(2001).sum()) == pytest.approx(
        float(reference_mass(2001).sum()), abs=1e-6)

    # The arm as implemented is the exposure-matched one: a row without a short
    # caption keeps its own distribution instead of leaving mass for the
    # sampler to hand to its last caption.
    implemented = stationary_exposures([view], arms["D"].policy, 64)[0]
    assert float(np.max(np.abs(implemented - matched_mass(64)))) < 1e-9


def test_stationary_arm_keeps_rows_without_a_short_caption_normalised():
    """A row with no short caption has nothing for the reserve to cover."""
    view = view_of([(1, [300, 900])])
    policy = next(arm for arm in build_arms(settings_for_run(1)) if arm.key == "D").policy

    probabilities, row_totals = stationary_probabilities(view, policy, 64)

    # Both captions sit above the short threshold, so the row keeps the
    # distribution the length preference gives it — the average of a
    # two-caption softmax is symmetric here, so each caption averages 0.5.
    assert row_totals[0] == pytest.approx(1.0)
    assert probabilities == pytest.approx([0.5, 0.5], abs=1e-12)

    as_implemented, exposure_matched = stationary_exposures([view], policy, 64)
    assert as_implemented == pytest.approx([0.0, 0.5, 0.0, 0.5, 0.0], abs=1e-12)
    # Renormalising the row is what the training-side average does now, so the
    # two readings agree.
    assert exposure_matched == pytest.approx(as_implemented, abs=1e-12)


def test_stationary_probabilities_match_the_sampler_construction():
    """The arm must be the one training builds, not a look-alike."""
    from src.dataset.sampler import _stationary_probabilities

    rows = [(1, [120, 400]), (2, [90, 700, 1500]), (1, [200, 250]), (2, [2048, 300])]
    view = view_of(rows)
    policy = next(arm for arm in build_arms(settings_for_run(1)) if arm.key == "D").policy
    metadata = RowLengthMetadata(
        resolution_ids=np.array([resolution for resolution, _ in rows], dtype=np.int64),
        caption_offsets=np.array(
            [0] + list(np.cumsum([len(row) for _, row in rows])), dtype=np.int64
        ),
        prompt_lengths=np.array([length for _, row in rows for length in row], dtype=np.int64),
    )

    ours, _ = stationary_probabilities(view, policy, 64)
    theirs = _stationary_probabilities(metadata, policy)

    assert ours == pytest.approx(np.asarray(theirs), rel=1e-12)


# ---------------------------------------------------------------------------
# The draw model.
# ---------------------------------------------------------------------------


def test_brute_force_draws_agree_with_the_expected_exposure(tmp_path):
    """Draw samples the way the trainer does and compare to the expectation.

    This is the strongest end-to-end check available on a CPU: the rows are
    drawn dataset-first and row-uniform, the captions are drawn by the real
    selector functions (and, for the stationary arm, by the sampler's own
    weighted index over its own averaged probabilities), and the progress clock
    advances one step at a time.  The simulation never runs those draws; it
    integrates their expectation, so agreement here covers the row weighting,
    the mix weights, the schedule mapping and the band edges at once.
    """
    from src.dataset.captions import (
        sample_caption_index_from_lengths,
        sample_caption_index_from_token_counts,
    )
    from src.dataset.sampler import _stationary_probabilities, _weighted_index

    short = write_sidecar(tmp_path / "short", [(1, [120, 400]), (2, [90]), (1, [200, 200, 1400])])
    long = write_sidecar(tmp_path / "long", [(1, [300, 900]), (2, [600, 1500])])
    mix_spec = mix((short, 0.7), (long, 0.3))
    entries = parse_mix_spec(mix_spec, [], [])
    metadatas = load_sidecars(entries)

    steps, per_step = 2000, 8
    settings = settings_for_run(steps * per_step)
    arms = {arm.key: arm for arm in build_arms(settings)}
    views = [DatasetView.from_metadata(entry, metadata, SHORT_THRESHOLD)
             for entry, metadata in zip(entries, metadatas)]
    stationary = [_stationary_probabilities(metadata, arms["D"].policy)
                  for metadata in metadatas]

    rng = random.Random(20240911)
    observed = {key: np.zeros(len(BANDS)) for key in arms}
    weights = [entry.weight for entry in entries]
    for step in range(steps):
        stage = step / steps
        for _ in range(per_step):
            index = rng.choices(range(len(entries)), weights=weights)[0]
            metadata = metadatas[index]
            row = rng.randrange(metadata.num_rows)
            selected = metadata.row_slice(row)
            lengths = metadata.prompt_lengths[selected].tolist()

            def band_at(index):
                return int(np.searchsorted((256, 512, 896, 1280),
                                           metadata.prompt_lengths[selected][index],
                                           side="right"))

            observed["A"][band_at(sample_caption_index_from_token_counts(
                lengths, stage=stage, rng=rng))] += 1
            for key in ("B", "C", "E"):
                policy = arms[key].policy
                observed[key][band_at(sample_caption_index_from_lengths(
                    lengths, policy.beta(stage), RESERVE, SHORT_THRESHOLD, rng=rng))] += 1
            observed["D"][band_at(
                _weighted_index(rng, stationary[index][selected]))] += 1

    grid = progress_grid(11)
    integral = trapezoid_weights(grid)
    for key in "ABCDE":
        expected = (integral @ arm_curve(arms[key], views, grid, settings)) * settings.total_draws
        share = expected / settings.total_draws
        sigma = np.sqrt(settings.total_draws * share * (1.0 - share))
        # Four standard deviations over 25 numbers: a systematic error in the
        # row weighting or the schedule mapping would be far larger than this.
        assert float(np.max(np.abs(observed[key] - expected) / np.maximum(sigma, 1.0))) < 4.0


def test_mix_weights_change_the_exposure(tmp_path):
    short = write_sidecar(tmp_path / "short", [(1, [100, 120]), (2, [80, 130])])
    long = write_sidecar(tmp_path / "long", [(1, [1400, 1500]), (2, [1200, 1300])])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert main(["--mix", mix((short, 0.9), (long, 0.1)), "--steps", "10",
                 "--samples-per-step", "10", "--out", str(first)]) == 0
    assert main(["--mix", mix((short, 0.1), (long, 0.9)), "--steps", "10",
                 "--samples-per-step", "10", "--out", str(second)]) == 0

    swapped = json.loads(first.read_text(encoding="utf-8"))
    original = json.loads(second.read_text(encoding="utf-8"))
    assert swapped["inputs"]["datasets"][0]["alias"] == "short"
    assert original["inputs"]["datasets"][0]["alias"] == "short"
    for arm_index in range(5):
        swapped_long = swapped["arms"][arm_index]["long_bands"]["ge256"]["count"]
        original_long = original["arms"][arm_index]["long_bands"]["ge256"]["count"]
        # One mix is nearly all short captions, the other nearly all long ones.
        assert swapped_long < 0.25 * 100
        assert original_long > 0.75 * 100
        assert swapped_long < original_long


def test_empty_rows_are_not_drawn(tmp_path):
    """Rows without captions cannot produce a sample, so they must not dilute."""
    with_empty = write_sidecar(tmp_path / "with_empty",
                               [(1, [150, 700]), (1, []), (2, [])])
    without_empty = write_sidecar(tmp_path / "without_empty", [(1, [150, 700])])
    padded = tmp_path / "padded.json"
    clean = tmp_path / "clean.json"

    assert main(["--mix", f"{with_empty}:1.0", "--steps", "10", "--samples-per-step", "10",
                 "--out", str(padded)]) == 0
    assert main(["--mix", f"{without_empty}:1.0", "--steps", "10", "--samples-per-step", "10",
                 "--out", str(clean)]) == 0

    padded_payload = json.loads(padded.read_text(encoding="utf-8"))
    clean_payload = json.loads(clean.read_text(encoding="utf-8"))
    assert padded_payload["inputs"]["datasets"][0]["rows"] == 3
    assert padded_payload["inputs"]["datasets"][0]["rows_with_captions"] == 1
    for arm_index in range(5):
        for name, _, _ in BANDS:
            assert padded_payload["arms"][arm_index]["bands"][name]["count"] == pytest.approx(
                clean_payload["arms"][arm_index]["bands"][name]["count"], abs=1e-9)


def test_a_drawn_row_weighs_the_same_whatever_its_caption_count():
    """A row is one sample, so its caption count must not weigh it up."""
    view = view_of([(1, [150, 400]), (1, [150, 400, 400, 400])])
    # No reserve and beta = 0 make every caption inside a row equally likely,
    # so the two-caption row gives its short caption half of its own mass and
    # the four-caption row a quarter. Drawing captions instead of rows would
    # give the short caption 2/5 of the dataset.
    probabilities = beta_probabilities(view, 0.0, 0.0, SHORT_THRESHOLD)
    masses = band_masses(view, probabilities)
    assert masses[band_of(150)] == pytest.approx(0.5 * 0.5 + 0.5 * 0.25)
    assert masses[band_of(400)] == pytest.approx(0.5 * 0.5 + 0.5 * 0.75)


def test_run_size_must_be_given_exactly_once(tmp_path, capsys):
    dataset = write_sidecar(tmp_path / "ds", [(1, [150, 700])])
    out = tmp_path / "out.json"

    assert main(["--mix", f"{dataset}:1.0", "--out", str(out)]) == 2
    assert "so the run size is known" in capsys.readouterr().err

    assert main(["--mix", f"{dataset}:1.0", "--total-draws", "100", "--steps", "10",
                 "--samples-per-step", "10", "--out", str(out)]) == 2
    assert "not both" in capsys.readouterr().err


def test_trapezoid_weights_integrate_a_linear_schedule_exactly():
    grid = progress_grid(11)
    weights = trapezoid_weights(grid)
    assert weights.sum() == pytest.approx(1.0)
    for start, end in ((0.0, 1.0), (1.0, -1.0), (-1.0, 1.5), (2.0, 2.0)):
        schedule = start + (end - start) * grid
        assert float(weights @ schedule) == pytest.approx((start + end) / 2.0)
    with pytest.raises(ValueError):
        trapezoid_weights(np.array([0.5]))
    with pytest.raises(ValueError):
        progress_grid(1)
