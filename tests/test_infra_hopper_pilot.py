from pathlib import Path
from scripts.bench.infra_hopper_pilot import pilot_plan


def test_hopper_pilot_is_bounded_reference_measurement():
    phases = pilot_plan(Path("/approved"), "pilot")
    assert [p["name"] for p in phases] == ["config-256p", "config-640p", "config-896p",
        "cuda-primitives", "rates-256p-mid", "rates-640p-mid", "rates-896p-mid"]
    for phase in phases[4:]:
        cmd = phase["command"]
        assert cmd[cmd.index("--ranks") + 1] == "8"
        assert cmd[cmd.index("--steps") + 1] == "128"
        assert cmd[cmd.index("--stage-timeout") + 1] == "1800"
        assert cmd[cmd.index("--trace-start") + 1] == "64"
        assert "--autotune-blocks" in cmd
        assert phase["timeout_seconds"] == 1850
