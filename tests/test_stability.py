"""Stability observations checked against unchanged state and measured parameter changes."""

import pytest
import torch

from src.models.inko import Inko
from src.pretrain.stability import StabilityMonitor, update_metrics


@pytest.fixture
def observation(tmp_path):
    torch.manual_seed(82)
    model = Inko(hidden_size=32, num_heads=4, double_stream_depth=1,
                    single_stream_depth=1, mlp_ratio=2, )
    # Nonzero final output and modulation expose conditioning changes that
    # adaLN-zero deliberately hides at initialization.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "modulation" in name or name.startswith("final_layer"):
                param.normal_(0, 0.05)
            param.grad = torch.full_like(param, 0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    monitor = StabilityMonitor(model, [opt], tmp_path)
    monitor.ensure_panel(torch.randn(4, 16, 8, 8), torch.randn(4, 16, 8, 8),
                         torch.randn(4, 5, 1024), torch.randn(4, 1024), torch.ones(4, 5))
    return model, opt, monitor


def test_observation_is_passive_and_skipped_update_is_zero(observation):
    model, opt, monitor = observation
    model.train()
    model.c_mlp.eval()  # Preserve mixed modes as well as the top-level flag.
    modes = [m.training for m in model.modules()]
    weights = {n: p.clone() for n, p in model.named_parameters()}
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    monitor.before_update()
    metrics = monitor.after_update(step=1, applied=False)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [m.training for m in model.modules()] == modes
    assert opt.state_dict()["state"] == {}
    for name, param in model.named_parameters():
        assert torch.equal(param, weights[name])
        assert torch.equal(param.grad, grads[name])
    assert metrics["stability/conditioning/update_rms"] == 0
    assert metrics["stability/response/prediction_change_rms"] == 0


def test_conditioning_counterfactual_removes_conditioner_only_change(observation):
    model, _, monitor = observation
    monitor.before_update()
    with torch.no_grad():
        model.c_mlp[-1].weight.add_(0.05)
        model.txt_pooled_proj.weight.add_(0.01)
    metrics = monitor.after_update(step=2, applied=True)
    assert metrics["stability/conditioning/update_rms"] > 0
    assert metrics["stability/conditioning/c2_shared_shift_rms"] > 0
    assert metrics["stability/response/conditioning_prediction_change_rms"] > 0
    for t in ("t010", "t050", "t090"):
        assert metrics[f"stability/response/{t}/loss_before"] == pytest.approx(
            metrics[f"stability/response/{t}/loss_held_condition"], abs=1e-7)


@pytest.mark.parametrize("decay", [0.0, 0.01])
def test_radial_decomposition_explains_actual_norm_change(decay):
    old = torch.tensor([[1.0, -2], [0, 3]])
    new = (1 - decay) * old + torch.tensor([[0.3, 0.2], [-0.1, 0.5]])
    metrics = update_metrics(old, new, decay)
    observed = float((torch.linalg.vector_norm(new) / torch.linalg.vector_norm(old)).square() - 1)
    assert metrics["norm_sq_change"] == pytest.approx(observed, abs=1e-6)
    assert metrics["radial"] + metrics["energy"] + metrics["decay"] == pytest.approx(
        observed, abs=1e-6)
