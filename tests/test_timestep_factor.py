"""Embedding bandwidth changes features without changing flow timesteps."""

import math
import pytest
import torch
from src.models.dit_blocks import TimestepEmbeddings


def test_native_time_features_and_input_preservation():
    times = torch.tensor([0.0, 0.001, 0.5, 1.0])
    original = times.clone()
    features = TimestepEmbeddings(64)(times)
    phase = times[:, None] * 1000 * torch.exp(-math.log(10000) * torch.arange(32) / 32)
    torch.testing.assert_close(features, torch.cat([phase.sin(), phase.cos()], dim=-1))
    assert torch.equal(times, original)
    torch.testing.assert_close(features.square().sum(-1), torch.full((4,), 32.0))


@pytest.mark.parametrize("factor", [1, 100, None])
@pytest.mark.parametrize("strict", [True, False])
def test_incompatible_time_features_cannot_load(factor, strict):
    state = {} if factor is None else {"factor": torch.tensor(factor)}
    with pytest.raises(RuntimeError, match="factor 1000"):
        TimestepEmbeddings(64).load_state_dict(state, strict=strict)
