"""Caption-selection policy and caption telemetry."""

import numpy as np
import pytest

from src.dataset.captions import (
    CaptionPolicy,
    average_caption_probabilities,
    caption_probabilities_from_lengths,
    sample_caption_index_from_lengths,
)
from src.train.caption_telemetry import CaptionTelemetry


class TestCaptionPolicy:
    def test_beta_orders_captions(self):
        lengths = [150, 700]
        assert caption_probabilities_from_lengths(lengths, 0.0, reserve=0.0) == [0.5, 0.5]
        shorter_first = caption_probabilities_from_lengths(lengths, -1.0, reserve=0.0)
        longer_first = caption_probabilities_from_lengths(lengths, 1.0, reserve=0.0)
        assert shorter_first[0] > shorter_first[1]
        assert longer_first[1] > longer_first[0]

    def test_beta_one_is_length_proportional(self):
        probabilities = caption_probabilities_from_lengths([100, 300], 1.0, reserve=0.0)
        assert probabilities == pytest.approx([0.25, 0.75])

    def test_reserve_protects_short_group(self):
        # Without a reserve the short caption is nearly squeezed out.
        bare = caption_probabilities_from_lengths([100, 2000], 1.0, reserve=0.0)
        assert bare[0] < 0.06
        protected = caption_probabilities_from_lengths([100, 2000], 1.0, reserve=0.2)
        assert protected[0] == pytest.approx(0.2 + 0.8 * bare[0])
        assert sum(protected) == pytest.approx(1.0)

    def test_reserve_is_not_diluted_by_more_long_captions(self):
        two = caption_probabilities_from_lengths([100, 900], 1.0, reserve=0.2)
        three = caption_probabilities_from_lengths([100, 900, 1800], 1.0, reserve=0.2)
        # The short caption keeps at least the reserve share either way.
        assert two[0] >= 0.2
        assert three[0] >= 0.2

    def test_single_caption_is_certain(self):
        assert caption_probabilities_from_lengths([300], 2.0) == [1.0]

    def test_schedules(self):
        linear = CaptionPolicy(kind="beta", beta_start=-1.0, beta_end=1.0, schedule="linear")
        assert [round(linear.beta(u), 2) for u in (0.0, 0.5, 1.0)] == [-1.0, 0.0, 1.0]
        early = CaptionPolicy(kind="beta", beta_start=-1.0, beta_end=1.0,
                              schedule="early", early_at=0.5)
        assert [round(early.beta(u), 2) for u in (0.0, 0.25, 0.5, 1.0)] == [-1.0, 0.0, 1.0, 1.0]

    def test_sampling_matches_probabilities(self):
        import random
        rng = random.Random(0)
        counts = [0, 0]
        for _ in range(20000):
            counts[sample_caption_index_from_lengths([150, 700], 1.0, rng=rng)] += 1
        share = counts[1] / sum(counts)
        assert share == pytest.approx(0.659, abs=0.02)

    def test_average_probabilities_match_elementwise_loop(self):
        policy = CaptionPolicy(kind="beta")
        matrix = np.array([[150.0, 700.0], [200.0, 300.0], [100.0, 900.0]])
        batched = average_caption_probabilities(matrix, policy)
        elementwise = np.stack([average_caption_probabilities(row, policy) for row in matrix])
        assert np.allclose(batched, elementwise)
        assert np.allclose(batched.sum(axis=1), 1.0)

    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError):
            CaptionPolicy(kind="nonsense")
        with pytest.raises(ValueError):
            CaptionPolicy(schedule="nonsense")
        with pytest.raises(ValueError):
            CaptionPolicy(short_reserve=1.5)


class TestCaptionTelemetry:
    def test_counts_selected_and_conditioned_separately(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100, 300, 900], [False, True, False],
                         bucket_hi=1024, resolution_id=0, bucket_idx=1)
        metrics = telemetry.snapshot()
        assert metrics["caption/selected"] == 3
        assert metrics["caption/dropout_rate"] == pytest.approx(1 / 3)
        assert metrics["caption/conditioned"] == 2
        # The dropped caption leaves exposure but stays in the selected shares.
        assert metrics["caption/selected_share/256_511"] == pytest.approx(1 / 3)
        assert metrics["caption/conditioned_share/256_511"] == pytest.approx(0.0)
        assert metrics["caption/conditioned_share/lt256"] == pytest.approx(0.5)

    def test_padding_and_percentiles(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100] * 50 + [900] * 50, [False] * 100,
                         bucket_hi=1024, resolution_id=0, bucket_idx=2)
        metrics = telemetry.snapshot()
        assert metrics["caption/selected_p50"] == 100
        assert metrics["caption/selected_p90"] == 900
        assert metrics["exec/padded_tokens"] == 1024 * 100
        assert metrics["exec/retained_tokens"] == 100 * 50 + 900 * 50
        assert metrics["exec/padding_fraction"] == pytest.approx(
            (1024 * 100 - 50000) / (1024 * 100))
        assert metrics["exec/bucket/res0_len2/samples"] == 100

    def test_snapshot_resets_window(self):
        telemetry = CaptionTelemetry()
        telemetry.record([200], [False], 256, 0, 0)
        first = telemetry.snapshot()
        assert first["caption/selected"] == 1
        assert telemetry.snapshot() == {}

    def test_mismatched_inputs_rejected(self):
        telemetry = CaptionTelemetry()
        with pytest.raises(ValueError):
            telemetry.record([100, 200], [False], 256, 0, 0)
