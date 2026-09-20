import json

import pytest

from scripts.bench.summarize_infra import summarize


def write_rank(directory, rank, times):
    rows = [dict(step=i + 1, rank=rank, seconds=seconds, global_samples=640,
                 shapes=[dict(shape=[10, 16, 32, 32, 128], count=1)], profiled=i == 2,
                 peak_allocated_bytes=100, peak_reserved_bytes=200)
            for i, seconds in enumerate(times)]
    (directory / f"rank-{rank}.jsonl").write_text("\n".join(map(json.dumps, rows)))


def test_summary_aligns_ranks_uses_slowest_and_labels_partial_data(tmp_path):
    write_rank(tmp_path, 0, [10, 2, 3, 4])
    write_rank(tmp_path, 1, [11, 1, 4])
    result = summarize(tmp_path, skip=1, window=2)
    assert result["common_steps"] == 3
    assert result["unaligned_records"] == 1
    assert result["all_updates"]["seconds"] == 17
    assert result["after_skip"]["mean_seconds"] == 3
    assert result["all_updates"]["first_seen_shape_updates"] == 1
    assert result["repeated_shape_updates"]["steps"] == 2
    assert result["all_updates"]["profiled_updates"] == 1
    assert result["observed_ranks"] == 2
    assert result["all_in_hero_gate_established"] is False


def test_short_run_does_not_invent_steady_measurement(tmp_path):
    write_rank(tmp_path, 0, [1])
    assert summarize(tmp_path)["after_skip"] is None


def test_invalid_rank_or_duplicate_not_silently_accepted(tmp_path):
    write_rank(tmp_path, 1, [1])
    with pytest.raises(ValueError, match="contiguous"):
        summarize(tmp_path)
    path = tmp_path / "rank-1.jsonl"
    path.rename(tmp_path / "rank-0.jsonl")
    with pytest.raises(ValueError, match="rank mismatch"):
        summarize(tmp_path)
