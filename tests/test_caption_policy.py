"""Caption draws checked against probabilities; distributed telemetry against serial recording."""

import random

import numpy as np
import pytest

from src.dataset.captions import CaptionPolicy, average_caption_probabilities, sample_caption_index_from_lengths
from src.pretrain.caption_telemetry import CaptionTelemetry


def test_sampling_matches_probabilities():
    rng = random.Random(0)
    draws = [sample_caption_index_from_lengths([150, 700], 1.0, rng=rng) for _ in range(20000)]
    assert np.mean(draws) == pytest.approx(0.659, abs=0.02)


def test_average_probabilities_match_individual_rows():
    policy = CaptionPolicy(kind='beta')
    matrix = np.array([[150., 700.], [200., 300.], [100., 900.]])
    expected = np.stack([average_caption_probabilities(row, policy) for row in matrix])
    np.testing.assert_allclose(average_caption_probabilities(matrix, policy), expected)


def test_distributed_telemetry_matches_serial_recording():
    rank_a, rank_b, serial = CaptionTelemetry(), CaptionTelemetry(), CaptionTelemetry()
    batches = [([100] * 10, [True] + [False] * 9, 256, [(0, row) for row in range(10)]),
               ([900] * 30, [True] * 15 + [False] * 15, 1024, [(1, row) for row in range(30)])]
    for rank, (lengths, dropped, width, rows) in zip((rank_a, rank_b), batches):
        rank.record(lengths, dropped, width, row_positions=rows)
        serial.record(lengths, dropped, width, row_positions=rows)
    rank_a.merge_window_counts(rank_b.window_counts())
    assert rank_a.snapshot() == pytest.approx(serial.snapshot())
