import json
import subprocess
import sys
import pytest

from scripts.bench import collect_infra_ledger


def test_runtime_uses_rank_container_stop_not_late_controller_cleanup():
    def event(time, reason, message, instance=None):
        return dict(time=f"2026-09-14 {time}", reason=reason,
                    message=message, instance=instance)

    events = [
        event("10:00:00", "Started", "Started container pytorch", "rank=0"),
        event("10:00:20", "Started", "Started container pytorch", "rank=1"),
        event("10:00:25", "Killing", "Stopping container exit-code-server", "rank=1"),
        event("10:01:00", "Killing", "Stopping container pytorch", "rank=0"),
        event("10:02:00", "Killing", "Stopping container pytorch", "rank=1"),
        event("18:00:00", "JobTerminated", "Deleting PodGroup"),
    ]
    assert collect_infra_ledger.observed_runtime_hours(
        events, gpus_per_instance=8, timezone="Asia/Shanghai") == pytest.approx((60 + 100) * 8 / 3600)
    assert collect_infra_ledger.observed_runtime_hours(
        [events[0], events[-1]], gpus_per_instance=8, timezone="Asia/Shanghai") == 0


def test_live_job_has_an_open_bound_not_a_query_error(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        action = cmd[3]
        if action == "list":
            data = {"items": [{"name": "infra-live", "project": "test"}]}
        elif action == "status":
            data = {"resource": {"gpu": 1, "nodes": 1},
                    "created_at": 1735689600000, "finished_at": None}
        else:
            assert action == "events"
            data = {"items": []}
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"success": True, "data": data}), "")

    path = tmp_path / "ledger.json"
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["ledger", "--workspace", "test", "--project", "test",
                                     "--prefix", "infra-", "--out", str(path)])
    collect_infra_ledger.main()
    result = json.loads(path.read_text())
    assert result["errors"] == 0
    assert result["open_allocations"] == 1
    assert result["jobs"][0]["accounting"] == "creation_to_observation_upper_bound"
    assert result["gpu_hours_bound"] > 0
