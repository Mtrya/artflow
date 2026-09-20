from pathlib import Path

from scripts.bench.infra_acceptance import plan
import pytest


def test_selected_hardware_configs_and_accumulation_reach_every_phase():
    selected = {'256p': 1, '640p': 2, '896p': 3}
    phases = plan(Path('/approved'), 'h200', set(selected),
                  stage_config_dir=Path('/pinned/h200'), accumulations=selected)
    assert not any(p['name'].startswith('config-') for p in phases)
    for stage, accumulation in selected.items():
        rates = [p for p in phases if p['name'].startswith(f'rates-{stage}-')]
        for phase in rates:
            cmd = phase['command']
            assert cmd[cmd.index('--accumulation') + 1] == str(accumulation)
            assert cmd[cmd.index('--stage-config-dir') + 1] == '/pinned/h200'
        prep = next(p['command'] for p in phases if p['name'] == f'prepare-{stage}')
        assert prep[prep.index('--accumulation') + 1] == str(accumulation)
        assert f'/pinned/h200/hero-{stage}.toml' in prep
        chain = phases[-1]['command']
        assert f'--{stage}-accumulation={accumulation}' in chain
    with pytest.raises(ValueError, match='all three'):
        plan(Path('/approved'), 'h200', set(), accumulations={'896p': 3})


def test_completion_plan_only_repeats_unfinished_validation():
    full = plan(Path("/approved"), "completion", {"256p", "640p", "896p"})
    remaining = plan(Path("/approved"), "completion", {"256p", "640p", "896p"},
                     completion_only=True)
    names = [p["name"] for p in remaining]
    assert names == ["config-256p", "config-640p", "config-896p", "prepare-896p",
                     "memory-896p", "verify-memory-896p", "cuda-primitives",
                     "stage-chain-and-evaluation"]
    assert remaining == [p for p in full if p["name"] in names]


def test_final_plan_covers_physical_ranks_cadence_and_distribution():
    phases = plan(Path("/approved"), "acceptance", {"256p", "896p"})
    rates = [p for p in phases if p["name"].startswith("rates-")]
    assert len(rates) == 9
    assert [p["name"] for p in phases[:3]] == ["config-256p", "config-640p", "config-896p"]
    assert [p["name"] for p in rates[:3]] == [
        "rates-256p-mid", "rates-640p-mid", "rates-896p-mid"]
    for phase in rates:
        cmd = phase["command"]
        stage = cmd[cmd.index("--stage") + 1]
        assert cmd[cmd.index("--ranks") + 1] == "8"
        assert ("--autotune-blocks" in cmd) == (stage != "640p")
        assert "--no-cache-clear" in cmd and "--gpu-health-snapshot" in cmd
        assert "--no-cpu-wall-profile" in cmd
        assert cmd[cmd.index("--stage-config-dir") + 1] == "/approved/runs/acceptance/configs"
        if phase["name"].endswith("-mid"):
            assert cmd[cmd.index("--steps") + 1] == "256"
            assert cmd[cmd.index("--trace-steps") + 1] == "3"
        else:
            assert "--trace-start" not in cmd


def test_memory_and_stage_chain_use_same_selected_flags():
    phases = {p["name"]: p["command"] for p in plan(Path("/approved"), "acceptance", {"640p"})}
    for stage, accumulation in (("256p", "1"), ("640p", "5"), ("896p", "7")):
        prep = phases[f"prepare-{stage}"]
        assert prep[prep.index("--accumulation") + 1] == accumulation
        assert prep[prep.index("--health-interval") + 1] == "1"
        assert prep[prep.index("--cycles") + 1] == "2"
        train = phases[f"memory-{stage}"]
        assert "--nproc_per_node=8" in train
        assert ("--train-arg=--compile_autotune" in train) == (stage == "640p")
    chain = phases["stage-chain-and-evaluation"]
    assert "--with-evaluation" in chain
    assert "--640p-train-arg=--compile_autotune" in chain
    assert "--256p-train-arg=--compile_autotune" not in chain
