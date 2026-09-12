"""Caption-selection policy and caption telemetry."""

import numpy as np
import pytest

from src.dataset.captions import (
    CaptionPolicy,
    average_caption_probabilities,
    caption_probabilities_from_lengths,
    sample_caption_index_from_lengths,
)
from src.train.caption_telemetry import CaptionTelemetry, PolicyState


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

    def test_average_probabilities_keep_rows_without_short_captions_intact(self):
        """A row with no short caption has nothing for the reserve to cover.

        Scaling it down the way a row with a short caption is scaled would leave
        its probabilities summing to less than one, and the sampler's fallback
        hands the remainder to whichever caption sits last in the list.
        """
        policy = CaptionPolicy(kind="beta", beta_start=0.0, beta_end=0.0,
                               short_reserve=0.2, short_threshold=256)
        long_only = average_caption_probabilities([300.0, 900.0], policy)
        assert long_only.sum() == pytest.approx(1.0)
        assert long_only == pytest.approx([0.5, 0.5])

        with_short = average_caption_probabilities([100.0, 900.0], policy)
        assert with_short.sum() == pytest.approx(1.0)
        assert with_short == pytest.approx([0.6, 0.4])

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
        telemetry.record([100, 300, 900], [False, True, False], bucket_hi=1024)
        metrics = telemetry.snapshot()
        # The selected mean keeps the dropped caption in it; the dropout rate
        # separates what the model was actually conditioned on.
        assert metrics["caption/selected_mean_tokens"] == pytest.approx(1300 / 3)
        assert metrics["caption/dropout_rate"] == pytest.approx(1 / 3)
        # A window that recorded no loss weights reports no weight moment, so
        # the emitted set is exactly the caption, execution and repetition
        # metrics; the percentiles come off the retained-length histogram,
        # dropped caption included, because that is the distribution the run
        # drew from.
        assert set(metrics) == {
            "caption/selected_mean_tokens",
            "caption/dropout_rate",
            "caption/selected_p50",
            "caption/selected_p90",
            "caption/selected_p99",
            "exec/padding_fraction",
            "exec/samples_per_micro_batch",
            "repetition/unique_rows_cumulative",
        }
        assert metrics["caption/selected_p50"] == 300
        assert metrics["caption/selected_p90"] == 900

    def test_padding_and_percentiles(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100] * 50 + [900] * 50, [False] * 100, bucket_hi=1024)
        metrics = telemetry.snapshot()
        assert metrics["caption/selected_p50"] == 100
        assert metrics["caption/selected_p90"] == 900
        assert metrics["caption/selected_p99"] == 900
        assert metrics["exec/padding_fraction"] == pytest.approx(
            (1024 * 100 - 50000) / (1024 * 100))
        assert metrics["exec/samples_per_micro_batch"] == 100

    def test_snapshot_resets_window(self):
        telemetry = CaptionTelemetry()
        telemetry.record([200], [False], 256)
        first = telemetry.snapshot()
        assert first["caption/selected_mean_tokens"] == 200
        assert telemetry.snapshot() == {}

    def test_mismatched_inputs_rejected(self):
        telemetry = CaptionTelemetry()
        with pytest.raises(ValueError):
            telemetry.record([100, 200], [False], 256)

    def test_unique_rows_cumulative_survives_window_resets(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100] * 10, [False] * 10, 256,
                         row_positions=[(0, row) for row in range(10)])
        first = telemetry.snapshot()
        assert first["repetition/unique_rows_cumulative"] == 10

        telemetry.record([900] * 10, [False] * 10, 1024,
                         row_positions=[(0, row) for row in range(10, 20)])
        second = telemetry.snapshot()
        # The window is only the second batch; the run total keeps both.
        assert second["caption/selected_mean_tokens"] == 900
        assert second["caption/dropout_rate"] == 0.0
        # The one metric that outlives a window: distinct rows drawn so far.
        assert second["repetition/unique_rows_cumulative"] == 20

    def test_read_only_snapshot_does_not_count_a_sample_twice(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100], [False], 256, row_positions=[(0, 7)])
        peek = telemetry.snapshot(reset=False)
        assert peek["caption/selected_mean_tokens"] == 100
        # Nothing has been closed yet, so the run total is still empty.
        assert peek["repetition/unique_rows_cumulative"] == 0.0
        assert telemetry.snapshot(reset=False)["repetition/unique_rows_cumulative"] == 0.0

        closed = telemetry.snapshot()
        assert closed["repetition/unique_rows_cumulative"] == 1.0
        # Drawing the same row again is a repeat, not a second distinct row.
        telemetry.record([900], [False], 1024, row_positions=[(0, 7)])
        after = telemetry.snapshot()
        assert after["repetition/repeat_rate"] == pytest.approx(1.0)
        assert after["repetition/unique_rows_cumulative"] == 1.0

    def test_reduce_is_a_noop_for_a_single_rank(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100, 200], [True, False], 256)
        telemetry.reduce(device=None, world_size=1)
        metrics = telemetry.snapshot()
        assert metrics["caption/selected_mean_tokens"] == pytest.approx(150)
        assert metrics["caption/dropout_rate"] == pytest.approx(0.5)

    def test_cross_rank_merge_weights_by_samples(self):
        rank_a = CaptionTelemetry()
        rank_b = CaptionTelemetry()
        # One micro-batch each but 10 and 30 samples: pooling the counters
        # differs from both an average of the rank rates (0.3) and a
        # per-micro-batch average (0.3).
        rank_a.record([100] * 10, [True] + [False] * 9, 256,
                      row_positions=[(0, row) for row in range(10)])
        rank_b.record([900] * 30, [True] * 15 + [False] * 15, 1024,
                      row_positions=[(1, row) for row in range(30)])

        rank_a.merge_window_counts(rank_b.window_counts())
        metrics = rank_a.snapshot()

        assert metrics["caption/dropout_rate"] == pytest.approx(16 / 40)
        assert metrics["caption/selected_mean_tokens"] == pytest.approx(
            (10 * 100 + 30 * 900) / 40)
        assert metrics["exec/samples_per_micro_batch"] == 20
        padded = 10 * 256 + 30 * 1024
        assert metrics["exec/padding_fraction"] == pytest.approx(
            (padded - (10 * 100 + 30 * 900)) / padded)
        # Percentiles come from the merged histogram, not from the ranks'
        # percentiles, which would average to 500 here.
        assert metrics["caption/selected_p50"] == 900
        assert metrics["caption/selected_p90"] == 900
        # The run total folds the pooled window rather than this rank alone.
        assert metrics["repetition/unique_rows_cumulative"] == 40

    def test_policy_metrics_are_sample_weighted_means(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100] * 8, [False] * 8, 256, policy=PolicyState(
            progress=0.2, curriculum_position=0.1, strength=-0.6, short_reserve=0.2))
        telemetry.record([100] * 2, [False] * 2, 256, policy=PolicyState(
            progress=0.6, curriculum_position=0.5, strength=0.2, short_reserve=0.2))
        first = telemetry.snapshot()

        # Weighted by samples, not by micro-batch: (8 * -0.6 + 2 * 0.2) / 10.
        assert first["policy/strength"] == pytest.approx(-0.6 * 0.8 + 0.2 * 0.2)
        assert first["policy/short_reserve"] == 0.2
        assert set(key for key in first if key.startswith("policy/")) == {
            "policy/strength", "policy/short_reserve"}

        telemetry.record([100] * 5, [False] * 5, 256, policy=PolicyState(
            progress=1.0, curriculum_position=1.0, strength=1.0, short_reserve=0.2))
        second = telemetry.snapshot()
        # A window reports only itself, so the first window's mean is gone.
        assert second["policy/strength"] == pytest.approx(1.0)
        assert second["policy/short_reserve"] == 0.2

    def test_repetition_counts_first_time_and_repeat_rows(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100, 100], [False, False], 256,
                         row_positions=[(0, 7), (0, 8)])
        first = telemetry.snapshot()
        assert first["repetition/repeat_rate"] == 0.0
        assert first["repetition/unique_rows_cumulative"] == 2

        telemetry.record([100, 100, 100], [False] * 3, 256,
                         row_positions=[(0, 7), (0, 9), (1, 7)])
        second = telemetry.snapshot()
        assert second["repetition/repeat_rate"] == pytest.approx(1 / 3)
        # The same row index under another dataset is a different row.
        assert second["repetition/unique_rows_cumulative"] == 4

    def test_repetition_is_optional_and_checked(self):
        telemetry = CaptionTelemetry()
        telemetry.record([100], [False], 256)
        metrics = telemetry.snapshot()
        # Without row positions there is nothing to count repeats against, so
        # the rate is absent; the run total is reported and still empty.
        assert "repetition/repeat_rate" not in metrics
        assert metrics["repetition/unique_rows_cumulative"] == 0.0

        with pytest.raises(ValueError):
            telemetry.record([100], [False], 256, row_positions=[(0, 1), (0, 2)])
