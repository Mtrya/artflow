"""Time-feature conventions survive loading without changing flow time."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.artflow import ArtFlow
from src.models.dit_blocks import TimestepEmbeddings
from src.pretrain.config import flatten, load_config


def test_legacy_embedding_and_state_are_unchanged():
    t = torch.linspace(0, 1, 17)
    frequencies = torch.exp(-math.log(10000) * torch.arange(32).float() / 32)
    phases = t[:, None] * frequencies[None, :]
    expected = torch.cat([phases.sin(), phases.cos()], dim=-1)
    embedder = TimestepEmbeddings(64)
    assert torch.equal(embedder(t), expected)
    assert not embedder.state_dict()
    embedder.load_state_dict({})


def test_factor_only_changes_features_and_preserves_input():
    t = torch.tensor([0.01, 0.2, 0.9], dtype=torch.bfloat16)
    original = t.clone()
    scaled = TimestepEmbeddings(64, time_factor=1000)
    assert torch.equal(scaled(t), TimestepEmbeddings(64)(t.float() * 1000))
    assert torch.equal(t, original)
    assert scaled.to(torch.bfloat16).factor.dtype == torch.int64


@pytest.mark.parametrize("source,target", [(1, 1000), (1000, 1), (1000, 100)])
@pytest.mark.parametrize("strict", [True, False])
def test_mismatched_checkpoint_is_rejected(source, target, strict):
    saved = TimestepEmbeddings(64, time_factor=source).state_dict()
    with pytest.raises(RuntimeError, match="time_factor mismatch"):
        TimestepEmbeddings(64, time_factor=target).load_state_dict(saved, strict=strict)


def test_model_config_and_weights_recover_factor(tmp_path):
    shape = dict(hidden_size=64, num_heads=4, double_stream_depth=1,
                 single_stream_depth=1, in_channels=4, txt_in_features=32,
                 conditioning_scheme="fused", branch_norm=True)
    model = ArtFlow(**shape, timestep_factor=1000).eval()
    # The normal zero output initialization would hide a wrong time factor.
    with torch.no_grad():
        torch.nn.init.normal_(model.final_layer[1].weight, std=0.02)
        for block in model.blocks:
            for name, module in block.named_children():
                if name.startswith("modulation"):
                    torch.nn.init.normal_(module[-1].weight, std=0.02)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, weights_only=True)
    inferred = ArtFlow._infer_config_from_state_dict(state)
    assert inferred["timestep_factor"] == model.get_config()["timestep_factor"] == 1000
    rebuilt = ArtFlow(**shape, timestep_factor=inferred["timestep_factor"]).eval()
    rebuilt.load_state_dict(state)
    inputs = (torch.randn(2, 4, 4, 4), torch.tensor([0.2, 0.8]),
              torch.randn(2, 3, 32), torch.randn(2, 32))
    with torch.no_grad():
        assert torch.equal(model(*inputs), rebuilt(*inputs))


def test_config_passes_factor_to_training(tmp_path):
    path = tmp_path / "factor.toml"
    path.write_text("[model]\ntimestep_factor = 1000\n")
    assert flatten(load_config([str(path)]))["timestep_factor"] == 1000


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_factor_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        TimestepEmbeddings(64, time_factor=value)
