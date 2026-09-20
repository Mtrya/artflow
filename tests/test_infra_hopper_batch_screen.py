from pathlib import Path

from scripts.bench.infra_hopper_batch_screen import plan


def test_bounded_candidate_screen_preserves_stage_policy_and_precision_flags():
    phases = plan(Path('/approved'), 'probe')
    assert [p['name'] for p in phases] == ['640p-reference','640p-candidate',
                                         '896p-candidate','896p-reference']
    assert [p['optional'] for p in phases] == [False,False,False,True]
    for p, accum in zip(phases,[5,2,3,7]):
        cmd = p['command']
        assert cmd[cmd.index('--accumulation')+1] == str(accum)
        assert cmd[cmd.index('--ranks')+1] == '8'
        assert cmd[cmd.index('--trace-start')+1] == '-1'
        assert ('--bucket-plan' in cmd) == p['name'].endswith('candidate')
        assert '--autotune-blocks' in cmd
        assert '--native-flash-varlen' in cmd


def test_candidate_only_has_explicit_longer_bounds_and_no_reference_claim():
    phases = plan(Path('/approved'), 'probe', candidate_only=True)
    assert [p['name'] for p in phases] == ['640p-candidate', '896p-candidate']
    for phase in phases:
        assert phase['timeout_seconds'] == 1170
        assert phase['optional'] is False
        cmd = phase['command']
        assert cmd[cmd.index('--stage-timeout')+1] == '1140'
        assert cmd[cmd.index('--steps')+1] == '96'
