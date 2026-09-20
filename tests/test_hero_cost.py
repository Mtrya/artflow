"""Accounting arithmetic, not evidence of measured hero throughput."""

from copy import deepcopy

import pytest

from scripts.bench.cost_hero_run import STAGES, cost, maximize, stage_counts
from src.evaluation.prompt_grid import grid_due


def synthetic_costs():
    return dict(world_size=8, evidence="Synthetic unit-test costs, not GPU measurements",
                recovery_gpu_hours=0, allocated_idle_gpu_hours=0, final_kid_seconds=0,
                stages={stage: dict(step_seconds=1, startup_seconds=0,
                                    checkpoint_seconds=0, grid_seconds=0,
                                    loss_probe_seconds=0) for stage in STAGES})


def test_cli_requires_explicit_hardware_budget(monkeypatch, capsys):
    from scripts.bench.cost_hero_run import main

    monkeypatch.setattr("sys.argv", ["cost_hero_run", "not-read-without-budget.json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--budget-gpu-hours" in capsys.readouterr().err


def test_frozen_400k_event_counts():
    result = cost(400000, synthetic_costs())
    for stage, updates, checkpoints, grids, probes in (
        ("256p", 300000, 150, 30, 601),
        ("640p", 80000, 40, 10, 161),
        ("896p", 20000, 10, 4, 41),
    ):
        assert result["stages"][stage]["counts"] == dict(
            updates=updates, checkpoints=checkpoints, grids=grids, loss_probes=probes)
    assert result["gpu_hours"] == pytest.approx(400000 * 8 / 3600)
    assert result["readiness_established"] is False


@pytest.mark.parametrize("total", [5, 39, 40, 10001, 40001, 400019])
def test_count_matches_actual_grid_trigger_and_integer_endpoints(total):
    ends = (0, total * 75 // 100, total * 95 // 100, total)
    for i, (start, end) in enumerate(zip(ends, ends[1:])):
        extras = [end] if i == 0 else [start, start + 2000, end]
        expected = sum(grid_due(step, 10000, extras) for step in range(start + 1, end + 1))
        expected += int(i > 0)  # New-resolution baseline, not an update.
        assert stage_counts(start, end, incoming=i > 0)["grids"] == expected


def test_overheads_are_counted_as_whole_allocation_seconds():
    data = synthetic_costs()
    data.update(recovery_gpu_hours=10, allocated_idle_gpu_hours=2, final_kid_seconds=3600)
    for row in data["stages"].values():
        row.update(startup_seconds=60, checkpoint_seconds=2, grid_seconds=3, loss_probe_seconds=4)
    expected_seconds = 400000 + 3 * 60 + 200 * 2 + 44 * 3 + 803 * 4 + 3600
    assert cost(400000, data)["gpu_hours"] == pytest.approx(expected_seconds * 8 / 3600 + 12)


def test_report_ratios_include_overhead_without_selecting_production_horizon():
    data = synthetic_costs()
    data["recovery_gpu_hours"] = 10
    result = cost(400000, data)
    assert result["optimizer_steps_per_gpu_hour"] * result["gpu_hours"] == pytest.approx(400000)
    assert result["optimizer_steps_per_allocated_wall_hour"] == pytest.approx(
        8 * result["optimizer_steps_per_gpu_hour"])
    assert result["gpu_hours_per_1000_steps"] * 400 == pytest.approx(result["gpu_hours"])
    assert result["production_steps_selected"] is False


def test_maximum_matches_exhaustive_search_across_rounding_and_grid_collisions():
    data = synthetic_costs()
    data["stages"]["256p"]["step_seconds"] = 0.4
    data["stages"]["896p"].update(step_seconds=3, grid_seconds=50)
    budget = 0.25
    expected = max(t for t in range(5, 282) if cost(t, data)["gpu_hours"] <= budget)
    assert maximize(data, budget)["total_steps"] == expected
    assert maximize(data, 0.000001) is None


def test_refuses_lower_rank_extrapolation_and_missing_overhead():
    data = synthetic_costs()
    data["world_size"] = 4
    with pytest.raises(ValueError, match="eight-rank"):
        cost(400000, data)
    data = deepcopy(synthetic_costs())
    del data["stages"]["640p"]["grid_seconds"]
    with pytest.raises(KeyError):
        cost(400000, data)


@pytest.mark.parametrize("total", [0, 1, 2, 3, 4, 5.5, True])
def test_empty_or_noninteger_schedule_rejected(total):
    with pytest.raises(ValueError, match="three nonempty"):
        cost(total, synthetic_costs())
