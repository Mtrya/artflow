import pytest

from scripts.bench.watch_infra_resources import available_groups


def test_fragmented_or_reclaimable_gpus_are_not_a_free_node():
    payload = {"success": True, "data": {"items": [
        {"compute_group": "fragmented", "available_gpus": 16, "free_nodes": 0, "gpus_per_node": 8},
        {"compute_group": "busy", "available_gpus": 0, "high_priority_available_gpus": 8,
         "free_nodes": 1, "gpus_per_node": 8},
        {"compute_group": "ready", "available_gpus": 8, "free_nodes": 1, "gpus_per_node": 8},
        {"compute_group": "unapproved", "available_gpus": 8, "free_nodes": 1, "gpus_per_node": 8},
    ]}}
    assert available_groups(payload, ["fragmented", "busy", "ready"]) == ["ready"]


def test_failed_query_never_establishes_availability():
    with pytest.raises(ValueError):
        available_groups({"success": False}, ["ready"])
