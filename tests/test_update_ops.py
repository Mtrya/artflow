"""EMA checked against PyTorch averaging; gradient scaling against a scaled loss."""

from copy import deepcopy

import pytest
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.pretrain.update_ops import divide_gradients, update_ema


@pytest.mark.parametrize('foreach', [False, True])
def test_ema_matches_pytorch_averaged_model(foreach):
    torch.manual_seed(7)
    model = torch.nn.Linear(7, 5)
    ema = deepcopy(model)
    reference = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    reference.update_parameters(model)
    for _ in range(5):
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        update_ema(ema, model, 0.9, foreach=foreach)
        reference.update_parameters(model)
    x = torch.randn(3, 7)
    torch.testing.assert_close(ema(x), reference(x))


@pytest.mark.parametrize('foreach', [False, True])
def test_gradient_division_matches_differentiating_scaled_loss(foreach):
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2)
    reference = deepcopy(model)
    x = torch.randn(5, 3)
    model(x).square().sum().backward()
    (reference(x).square().sum() / 37.25).backward()
    divide_gradients(model.parameters(), 37.25, foreach=foreach)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
