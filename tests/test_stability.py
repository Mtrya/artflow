"""The observer must explain real updates without changing the training state."""
import json

import pytest
import torch

from src.models.artflow import ArtFlow
from src.pretrain.config import load_config
from src.pretrain.stability import StabilityMonitor, feature_metrics, update_metrics


@pytest.fixture
def observation(tmp_path):
    torch.manual_seed(82)
    model = ArtFlow(hidden_size=32, num_heads=4, double_stream_depth=1,
                    single_stream_depth=1, txt_in_features=16, in_channels=4,
                    mlp_ratio=2, conditioning_scheme="fused",
                    single_stream_modulation="layer", branch_norm=True)
    # Nonzero final output and modulation expose conditioning changes that
    # adaLN-zero deliberately hides at initialization.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "modulation" in name or name.startswith("final_layer"):
                param.normal_(0, 0.05)
            param.grad = torch.full_like(param, 0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    monitor = StabilityMonitor(model, [opt], tmp_path)
    monitor.ensure_panel(torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8),
                         torch.randn(4, 5, 16), torch.randn(4, 16), torch.ones(4, 5))
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
    assert monitor.before is None
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


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


def test_fixed_panel_survives_resume_and_ignores_new_training_batch(observation):
    model, opt, monitor = observation
    resumed = StabilityMonitor(model, [opt], monitor.run_dir)
    assert resumed.panel_id == monitor.panel_id
    resumed.ensure_panel(None, None, None, None, None)
    for a, b in zip(monitor._inputs()[:2], resumed._inputs()[:2]):
        assert torch.equal(a, b)
    resumed.before_update()
    resumed.after_update(step=15, applied=False)
    line = json.loads((monitor.run_dir / "stability.jsonl").read_text())
    assert line["panel_id"] == monitor.panel_id and line["step"] == 15
    assert line["scope"] == "rank0_fixed_panel"


def test_feature_variation_separates_time_and_caption():
    time_only = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unsqueeze(1)
    stats = feature_metrics(time_only, rows=2)
    assert stats["caption_ratio"] == 0 and stats["time_ratio"] > 0
    caption_only = torch.tensor([1.0, 2.0] * 3).unsqueeze(1)
    stats = feature_metrics(caption_only, rows=2)
    assert stats["time_ratio"] == 0 and stats["caption_ratio"] > 0


@pytest.mark.parametrize("decay", [0.0, 0.01])
def test_radial_decomposition_explains_actual_norm_change(decay):
    old = torch.tensor([[1.0, -2], [0, 3]])
    new = (1 - decay) * old + torch.tensor([[0.3, 0.2], [-0.1, 0.5]])
    metrics = update_metrics(old, new, decay)
    assert metrics["norm_sq_change"] == pytest.approx(
        metrics["radial"] + metrics["energy"] + metrics["decay"], abs=1e-6)
    zero = update_metrics(torch.zeros(2), torch.zeros(2), 0)
    assert all(value == 0 for value in zero.values())


def test_probe_restores_hooks_modes_and_compile_wrapper_on_error(observation):
    model, _, monitor = observation
    block = model.blocks[0]
    original = block.forward
    calls = []

    def failing(*args, **kwargs):
        calls.append("eager")
        raise RuntimeError("probe failure")

    def compiled(*args, **kwargs):
        pytest.fail("observer must bypass the training compiler")

    compiled._torchdynamo_orig_callable = failing
    block.forward = compiled
    with pytest.raises(RuntimeError, match="probe failure"):
        monitor.before_update()
    assert block.forward is compiled and calls == ["eager"]
    assert model.training
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    block.forward = original


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_stability_cadence_validation(tmp_path, value):
    config = tmp_path / "invalid.toml"
    config.write_text("[telemetry]\nstability_interval = " + str(value).lower())
    with pytest.raises(ValueError, match="stability_interval"):
        load_config([config])


def test_first_resumed_update_uses_new_configured_lr():
    from src.pretrain.train import build_linear_cosine_scheduler, restore_scheduler_base_lrs

    param = torch.nn.Parameter(torch.ones(1))
    old_opt = torch.optim.SGD([param], lr=0.02)
    old_sch = build_linear_cosine_scheduler(old_opt, num_warmup_steps=2,
                                           num_training_steps=100, min_learning_rate=0.001,
                                           base_learning_rate=0.02)
    for _ in range(5):
        old_opt.step()
        old_sch.step()
    opt = torch.optim.SGD([param], lr=0.003)
    sch = build_linear_cosine_scheduler(opt, num_warmup_steps=2, num_training_steps=100,
                                       min_learning_rate=0.00015, base_learning_rate=0.003)
    opt.load_state_dict(old_opt.state_dict())
    sch.load_state_dict(old_sch.state_dict())
    restore_scheduler_base_lrs(sch, [0.003])
    expected = 0.003 * sch.lr_lambdas[0](5)
    assert sch.last_epoch == 5
    param.grad = torch.ones(1)
    opt.step()
    assert param.item() == pytest.approx(1 - expected)
    assert sch.get_last_lr() == [expected]
