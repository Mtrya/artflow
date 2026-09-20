import json
import pytest
from unittest.mock import MagicMock, patch

from src.train.infra_metrics import InfraRecorder


def test_replay_digest_tracks_caption_and_row_identity(tmp_path):
    recorder = InfraRecorder(tmp_path, 0, record_identity=True)
    for index, caption in enumerate((1, 1, 2)):
        recorder.begin_update(index)
        recorder.micro((2, 16, 4, 4), 32,
                       sample_identity=dict(rows=[[0, 42], [0, 43]], captions=[caption, 0], batch_id=3))
        recorder.end_update(step=index + 1, seconds=1, global_samples=2, loss=.5,
                            progress=.5, peak_allocated=100, peak_reserved=120)
    recorder.begin_update(3)
    with pytest.raises(ValueError, match="identity"):
        recorder.micro((2, 16, 4, 4), 32)
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path / "infra/rank-0.jsonl").read_text().splitlines()]
    assert rows[0]["sample_identity_sha256"] == rows[1]["sample_identity_sha256"]
    assert rows[1]["sample_identity_sha256"] != rows[2]["sample_identity_sha256"]


def test_recorder_preserves_shapes_and_does_not_profile_by_default(tmp_path):
    with patch("torch.profiler.profile", side_effect=AssertionError("disabled")):
        recorder = InfraRecorder(tmp_path, 2)
        recorder.begin_update(100)
        recorder.micro((8, 16, 112, 112), 256)
        recorder.micro((8, 16, 112, 112), 256)
        recorder.end_update(step=101, seconds=1.5, global_samples=32, loss=.5,
                            progress=.9, peak_allocated=100, peak_reserved=120)
        recorder.close()
    result = json.loads((tmp_path / "infra/rank-2.jsonl").read_text())
    assert result["rank"] == 2 and result["step"] == 101
    assert result["shapes"] == [{"shape": [8, 16, 112, 112, 256], "count": 2}]
    assert not result["profiled"]


def test_trace_covers_only_requested_update_window(tmp_path):
    profiler = MagicMock()
    profiler.key_averages.return_value.table.return_value = "operator summary"
    with patch("torch.profiler.profile", return_value=profiler):
        recorder = InfraRecorder(tmp_path, 0, trace_start=2, trace_steps=2)
        for step in range(6):
            recorder.begin_update(step)
            recorder.end_update(step=step+1, seconds=1, global_samples=8, loss=.5,
                                progress=.5, peak_allocated=100, peak_reserved=120)
        recorder.close()
    profiler.start.assert_called_once()
    profiler.stop.assert_called_once()
    profiler.export_chrome_trace.assert_called_once()
    rows = [json.loads(line) for line in (tmp_path / "infra/rank-0.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if r["profiled"]] == [3, 4]


def test_checkpoint_cost_is_separate_from_update_timings(tmp_path):
    recorder = InfraRecorder(tmp_path, 1)
    recorder.checkpoint(step=30, seconds=12.5)
    recorder.close()
    assert (tmp_path / "infra/rank-1.jsonl").read_text() == ""
    assert json.loads((tmp_path / "infra/checkpoint-rank-1.jsonl").read_text()) == {
        "step": 30, "rank": 1, "seconds": 12.5,
    }
