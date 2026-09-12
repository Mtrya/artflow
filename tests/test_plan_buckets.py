"""The bucket planner must turn a real mix into a plan the trainer can read.

The plan file is the trainer's only input for bucket bounds and micro-batch
sizes, and the report next to it is the only record of how those numbers were
produced.  These tests pin both: an end-to-end run whose plan the training
loader accepts, the boundary solver on distributions whose answer can be
checked by hand, and the inputs the planner must refuse or honour (mix weights,
phase weights, batch sizes).
"""

import json

import numpy as np
import pytest

from scripts.plan_buckets import (
    PHASES,
    combine_phases,
    load_sidecars,
    main,
    parse_mix_spec,
    phase_histograms,
    phase_points,
    resolve_image_tokens,
    solve_boundaries,
)
from src.dataset.captions import CaptionPolicy
from src.dataset.length_buckets import architecture_cost
from src.dataset.length_metadata import RowLengthMetadata, sidecar_path
from src.train.train import load_bucket_plan
from src.utils.prompt_contract import MAX_SEQUENCE_LENGTH


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


def planned_distributions(mix_spec, phase_weights=None):
    """The planner's own p(l) per resolution id, straight from the mix text."""
    entries = parse_mix_spec(mix_spec, [], [])
    policy = CaptionPolicy(kind="beta")
    metadatas = load_sidecars(entries)
    resolutions = sorted({int(resolution) for metadata in metadatas
                          for resolution in metadata.resolution_ids})
    phases = phase_points(policy, 0.0, 1.0)
    histograms = phase_histograms(entries, metadatas, phases, policy, resolutions)
    weights = phase_weights or [1.0 / len(PHASES)] * len(PHASES)
    return combine_phases(histograms, weights)


def mean_length(probabilities):
    return float(np.dot(np.arange(1, MAX_SEQUENCE_LENGTH + 1), probabilities))


def test_plan_round_trips_through_the_trainer_loader(tmp_path):
    short = write_sidecar(tmp_path / "short",
                          [(1, [12, 24, 36]), (1, [18]), (2, [900, 1200])])
    long = write_sidecar(tmp_path / "long",
                         [(1, [40, 64]), (2, [300, 1500])])
    plan_path = tmp_path / "plan.json"

    code = main(["--mix", mix((short, 0.7), (long, 0.3)), "--buckets", "3",
                 "--image-tokens", '{"1": 256, "2": 640}', "--out", str(plan_path)])

    assert code == 0
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    # No extra top-level keys: the loader reads every one of them as a resolution.
    assert set(payload) == {"1", "2"}
    plan = load_bucket_plan(str(plan_path), resolution_ids=[1, 2])
    for resolution_id in (1, 2):
        buckets = plan.buckets_for(resolution_id)
        assert len(buckets) == 3
        assert [bucket.batch_size for bucket in buckets] == [16, 16, 16]
        assert buckets[-1].max_length == MAX_SEQUENCE_LENGTH
        bounds = [bucket.max_length for bucket in buckets]
        assert all(right > left for left, right in zip(bounds, bounds[1:]))


def test_report_records_every_input_and_the_baselines(tmp_path):
    short = write_sidecar(tmp_path / "short", [(1, [12, 24, 36])])
    long = write_sidecar(tmp_path / "long", [(1, [800, 1600])])
    plan_path = tmp_path / "plan.json"

    assert main(["--mix", mix((short, 0.6), (long, 0.4)), "--buckets", "3",
                 "--image-tokens", "256", "--out", str(plan_path)]) == 0

    report = (tmp_path / "plan.json.report.md").read_text(encoding="utf-8")
    for expected in ("# Bucket plan provenance", "## Inputs", "## Length distribution p(l)",
                     "## Boundaries and padding waste", "## Limitations",
                     "phase weights (early, middle, late)", "saved vs uniform",
                     "equal-mass waste", "share of draws", "short", "long"):
        assert expected in report
    # The two things a future reader must not have to guess: the phase weights are
    # an exposure approximation, and unmeasured batch sizes are placeholders.
    assert "equal steps per phase" in report
    assert "placeholder" in report and "*not* been measured" in report


def test_degenerate_distribution_matches_hand_computed_boundaries():
    costs = architecture_cost(1152, 24, 256).over_lengths(MAX_SEQUENCE_LENGTH)
    point_mass = np.zeros(MAX_SEQUENCE_LENGTH)
    point_mass[7] = 0.5                    # length 8
    point_mass[MAX_SEQUENCE_LENGTH - 1] = 0.5   # the cap itself

    # One bucket can only end at the cap, so the short mass is padded to it.
    assert solve_boundaries(point_mass, 1, costs) == [MAX_SEQUENCE_LENGTH]
    # Two buckets let the short mass sit on its own bound: no padding at all.
    assert solve_boundaries(point_mass, 2, costs) == [8, MAX_SEQUENCE_LENGTH]
    # A third bucket can only hold lengths with no mass, and ties are broken
    # towards the smallest bounds.
    assert solve_boundaries(point_mass, 3, costs) == [1, 8, MAX_SEQUENCE_LENGTH]

    heavier_long = np.zeros(MAX_SEQUENCE_LENGTH)
    heavier_long[63] = 0.3                 # length 64
    heavier_long[127] = 0.7                # length 128
    # Padding to the cap is expensive, so the boundary aligns with the heavier
    # length and the light one is padded up to it rather than to the cap.
    assert solve_boundaries(heavier_long, 2, costs) == [128, MAX_SEQUENCE_LENGTH]


def test_batch_sizes_must_match_the_bucket_count(tmp_path, capsys):
    dataset = write_sidecar(tmp_path / "ds", [(1, [12, 24, 36, 64])])
    plan_path = tmp_path / "plan.json"

    code = main(["--mix", mix((dataset, 1.0)), "--buckets", "3",
                 "--batch-sizes", json.dumps({"1": [8, 4]}), "--out", str(plan_path)])

    assert code == 2
    assert "2 entries but --buckets is 3" in capsys.readouterr().err
    assert not plan_path.exists()


def test_given_batch_sizes_are_used_and_not_called_placeholders(tmp_path, capsys):
    dataset = write_sidecar(tmp_path / "ds", [(1, [12, 24, 36, 64])])
    plan_path = tmp_path / "plan.json"

    code = main(["--mix", mix((dataset, 1.0)), "--buckets", "2",
                 "--batch-sizes", json.dumps({"1": [12, 3]}), "--out", str(plan_path)])

    assert code == 0
    assert [bucket.batch_size for bucket in
            load_bucket_plan(str(plan_path), [1]).buckets_for(1)] == [12, 3]
    assert "placeholder" not in capsys.readouterr().err
    assert "placeholder" not in (tmp_path / "plan.json.report.md").read_text(encoding="utf-8")


def test_a_missing_sidecar_is_named_in_the_error(tmp_path, capsys):
    code = main(["--mix", mix((tmp_path / "absent", 1.0)),
                 "--out", str(tmp_path / "plan.json")])

    assert code == 2
    assert str(sidecar_path(str(tmp_path / "absent"))) in capsys.readouterr().err


def test_mix_weights_change_the_length_distribution(tmp_path):
    short = write_sidecar(tmp_path / "short", [(1, [16, 24]), (1, [32])] * 20)
    long = write_sidecar(tmp_path / "long", [(1, [1536, 2048]), (1, [1280])] * 20)

    short_heavy = planned_distributions(mix((short, 0.9), (long, 0.1)))[1]
    long_heavy = planned_distributions(mix((short, 0.1), (long, 0.9)))[1]

    short_mean = mean_length(short_heavy.probabilities)
    long_mean = mean_length(long_heavy.probabilities)
    # A 90/10 mix is dominated by its heavy side, so the two draws must sit an
    # order of magnitude apart: the weights are not decoration.
    assert short_mean < 250
    assert long_mean > 1000
    assert short_mean * 4 < long_mean

    # The weights must reach the plan, not just the intermediate distribution.
    short_plan = tmp_path / "short_plan.json"
    long_plan = tmp_path / "long_plan.json"
    assert main(["--mix", mix((short, 0.9), (long, 0.1)), "--buckets", "3",
                 "--out", str(short_plan)]) == 0
    assert main(["--mix", mix((short, 0.1), (long, 0.9)), "--buckets", "3",
                 "--out", str(long_plan)]) == 0
    assert json.loads(short_plan.read_text(encoding="utf-8")) != json.loads(
        long_plan.read_text(encoding="utf-8"))


def test_phase_weights_move_the_distribution_along_the_curriculum(tmp_path):
    # One caption at each end of the range, so the phase beta decides which end
    # the draw lands on: early prefers the short caption, late the long one.
    dataset = write_sidecar(tmp_path / "ds", [(1, [16, 2048])] * 10)

    early_heavy = planned_distributions(mix((dataset, 1.0)), [0.8, 0.1, 0.1])[1]
    late_heavy = planned_distributions(mix((dataset, 1.0)), [0.1, 0.1, 0.8])[1]

    early_mean = mean_length(early_heavy.probabilities)
    late_mean = mean_length(late_heavy.probabilities)
    # Early training must draw the short caption far more often than late
    # training draws the long one, otherwise the phase betas are not reaching
    # the distribution.
    assert early_mean < 500
    assert late_mean > 1000
    assert early_mean * 2 < late_mean


def test_image_tokens_are_per_resolution_and_must_cover_the_data():
    assert resolve_image_tokens('{"1": 256, "2": 1024}', [1, 2]) == {1: 256, 2: 1024}
    assert resolve_image_tokens("256", [1, 2]) == {1: 256, 2: 256}
    with pytest.raises(ValueError, match=r"missing resolution ids \[2\]"):
        resolve_image_tokens('{"1": 256}', [1, 2])


def test_mixes_without_data_or_with_non_positive_weights_are_rejected(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"

    assert main(["--out", str(plan_path)]) == 2
    assert "no data given" in capsys.readouterr().err

    dataset = write_sidecar(tmp_path / "ds", [(1, [12, 24])])
    assert main(["--mix", mix((dataset, 0.0)), "--out", str(plan_path)]) == 2
    assert "must be positive" in capsys.readouterr().err


def test_more_buckets_than_available_lengths_are_rejected(tmp_path, capsys):
    dataset = write_sidecar(tmp_path / "ds", [(1, [12, 24])])
    plan_path = tmp_path / "plan.json"

    code = main(["--mix", mix((dataset, 1.0)), "--buckets",
                 str(MAX_SEQUENCE_LENGTH + 1), "--out", str(plan_path)])

    assert code == 2
    assert f"exceeds the {MAX_SEQUENCE_LENGTH} retained lengths" in capsys.readouterr().err
    assert not plan_path.exists()
