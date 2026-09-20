from pathlib import Path

from scripts.bench.infra_closeout import commands


def test_256_closeout_isolates_three_changes_and_repeats_reference():
    cases = commands(Path("/approved"), "probe", "256p", 2)
    assert [c["variant"] for c in cases] == [
        "cache_cpu", "nocache_cpu", "nocache_gpu", "autotune_gpu", "nocache_gpu_repeat"]
    policies = []
    for case in cases:
        cmd = case["command"]
        policies.append(("--no-cache-clear" in cmd, "--gpu-health-snapshot" in cmd,
                         "--autotune-blocks" in cmd))
        assert case["steps"] > 250
        assert "--no-cpu-wall-profile" in cmd
        assert "--foreach-updates" not in cmd
    assert policies == [(False, False, False), (True, False, False),
                        (True, True, False), (True, True, True), (True, True, False)]
    reference = cases[0]["command"]
    assert reference[reference.index("--cache-clear-interval") + 1] == "100"


def test_high_resolution_keeps_math_flags_fixed_for_autotune_aba():
    for stage in ("640p", "896p"):
        cases = commands(Path("/approved"), "probe", stage, 4)
        assert [c["variant"] for c in cases] == ["nocache_gpu", "autotune_gpu", "nocache_gpu_repeat"]
        assert all("--gpu-health-snapshot" in c["command"] for c in cases)
        assert all(c["command"][c["command"].index("--ranks") + 1] == "4" for c in cases)
