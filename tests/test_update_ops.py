from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest
import torch

from src.pretrain.update_ops import divide_gradients, update_ema, clear_local_cuda_cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_foreach_updates_match_reference(dtype):
    torch.manual_seed(7)
    live = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.Linear(5, 3)).to(dtype)
    live.register_buffer("counter", torch.tensor(11))
    ema = deepcopy(live)
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(.2)
    fast = deepcopy(ema)
    for _ in range(5):
        update_ema(ema, live, .9999)
        update_ema(fast, live, .9999, foreach=True)
    for expected, actual in zip(ema.state_dict().values(), fast.state_dict().values()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    params = list(live.parameters())
    for p in params[:-1]:
        p.grad = torch.randn_like(p)
    reference = deepcopy(live)
    for a, b in zip(params, reference.parameters()):
        b.grad = None if a.grad is None else a.grad.clone()
    divide_gradients(live.parameters(), 37.25, foreach=True)
    divide_gradients(reference.parameters(), 37.25)
    for a, b in zip(live.parameters(), reference.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_cache_clear_only_visits_requested_rank_device():
    with patch("torch.cuda.device", return_value=MagicMock()) as device, \
         patch("torch.cuda.empty_cache") as empty, \
         patch("torch.cuda.device_count", side_effect=AssertionError("must not enumerate peers")):
        clear_local_cuda_cache(torch.device("cuda:3"))
    device.assert_called_once_with(torch.device("cuda:3"))
    empty.assert_called_once_with()


def test_muon_and_adamw_preserve_updates_with_retained_gradient_views():
    from src.models.artflow import ArtFlow
    from src.pretrain.muon import build_param_groups

    torch.manual_seed(31)
    reference = ArtFlow(hidden_size=32, num_heads=4, double_stream_depth=1,
                        single_stream_depth=1, mlp_ratio=2)
    candidate = deepcopy(reference)
    optimizers = [build_param_groups(model, muon_lr=.02, adam_lr=1e-4, muon_wd=.0015, adam_wd=.01, adam_eps=1e-8, adam_betas=(.9,.95), muon_momentum=.95) for model in (reference, candidate)]
    storage = torch.empty(sum(p.numel() for p in candidate.parameters()))
    offset = 0
    for parameter in candidate.parameters():
        parameter.grad = storage[offset:offset + parameter.numel()].view_as(parameter)
        offset += parameter.numel()
    for _ in range(3):
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            expected.grad = torch.randn_like(expected)
            actual.grad.copy_(expected.grad)
        for group in optimizers:
            for optimizer in group:
                optimizer.step()
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for left, right in zip(*optimizers):
            for expected, actual in zip(left.state.values(), right.state.values()):
                for key in expected:
                    if torch.is_tensor(expected[key]):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    else:
                        assert actual[key] == expected[key]
            left.zero_grad(set_to_none=True)
            right.zero_grad(set_to_none=False)
        assert not storage.any()
        assert all(p.grad is None for p in reference.parameters())
        assert all(p.grad._base is storage for p in candidate.parameters())


def test_ema_decay_warmup_schedule():
    from src.pretrain.update_ops import ema_decay_at

    assert ema_decay_at(1, 0.9999) == 0.9999
    assert ema_decay_at(1, 0.9999, warmup=True) == 2.0 / 11.0
    # Monotone ramp that saturates exactly at the configured decay.
    values = [ema_decay_at(t, 0.9999, warmup=True) for t in (1, 100, 1000, 10000)]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert ema_decay_at(10**9, 0.9999, warmup=True) == 0.9999
    # Saturation near t ~= 90k: still ramping at 50k, saturated by 200k.
    assert ema_decay_at(50000, 0.9999, warmup=True) < 0.9999
    assert ema_decay_at(200000, 0.9999, warmup=True) == 0.9999
