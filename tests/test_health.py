import torch
from torch import nn

from src.train.health import (
    ema_rel_distance,
    qk_gain_stats,
    snapshot_weights,
    update_weight_ratios,
)


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))


def test_update_weight_ratio_zero_when_unchanged():
    model = _toy_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    snaps = snapshot_weights([opt])
    ratios = update_weight_ratios(snaps)
    assert ratios == [0.0]


def test_update_weight_ratio_matches_manual_delta():
    model = _toy_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    snaps = snapshot_weights([opt])
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    ratio = update_weight_ratios(snaps)[0]
    numel = sum(p.numel() for p in model.parameters())
    weight_f = sum(float(old.pow(2).sum()) for _, old in snaps[0])
    expected = (numel * 0.01**2) ** 0.5 / (weight_f**0.5 + 1e-12)
    assert abs(ratio - expected) < 1e-6


def test_qk_gain_stats_reads_rmsnorm_gains():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_norm = nn.RMSNorm(4)
            self.k_norm = nn.RMSNorm(4)

    model = Attn()
    with torch.no_grad():
        model.q_norm.weight.fill_(2.0)
        model.k_norm.weight.fill_(3.0)
    gains = qk_gain_stats(model)
    assert gains is not None
    assert gains[0] == 3.0
    assert gains[1] == 2.5


def test_qk_gain_stats_none_without_qk_norms():
    assert qk_gain_stats(_toy_model()) is None


def test_ema_rel_distance_zero_for_identical_models():
    a = _toy_model()
    b = _toy_model()
    assert ema_rel_distance(a, b) == 0.0


def test_ema_rel_distance_positive_after_drift():
    ema = _toy_model()
    live = _toy_model()
    with torch.no_grad():
        for p in live.parameters():
            p.add_(0.5)
    assert ema_rel_distance(ema, live) > 0.0


def test_same_device_snapshot_is_independent_fp32_storage():
    model = _toy_model().to(torch.bfloat16)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    snapshots = snapshot_weights([opt], device=next(model.parameters()).device)
    for param, old in snapshots[0]:
        assert old.dtype == torch.float32 and not old.requires_grad
        assert old.data_ptr() != param.data_ptr()
        assert torch.equal(old, param.float())
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.25)
    assert update_weight_ratios(snapshots)[0] > 0


def test_actual_training_health_block_releases_snapshots():
    import ast
    import inspect
    import weakref
    from src.train import train

    node = next(n for n in ast.walk(ast.parse(inspect.getsource(train.main)))
                if isinstance(n, ast.If) and ast.unparse(n.test) == "health_snapshot is not None")
    model = _toy_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    snapshots = snapshot_weights([opt])
    refs = [weakref.ref(pair[1]) for pair in snapshots[0]]
    ns = dict(health_snapshot=snapshots, update_weight_ratios=update_weight_ratios)
    del snapshots
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<training health block>", "exec"), ns)
    assert ns["health_snapshot"] is None
    assert all(ref() is None for ref in refs)
    assert ns["health_metrics"]["health/update_weight_ratio_muon"] == 0
