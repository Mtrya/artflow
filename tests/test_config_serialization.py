"""Checkpoints carry architecture metadata instead of guessing it from shapes."""

import json
import pytest
import torch
from src.models.artflow import ArtFlow


@pytest.mark.parametrize("double,single", [(0, 2), (1, 2), (2, 0)])
def test_model_file_roundtrip_preserves_all_weights_and_capacity(
    tmp_path, double, single
):
    model = ArtFlow(
        hidden_size=32,
        num_heads=4,
        double_stream_depth=double,
        single_stream_depth=single,
        mlp_ratio=2.25,
    )
    config = model.get_config()
    assert config["architecture"] == "artflow-v2"
    assert set(config) == {
        "architecture",
        "hidden_size",
        "num_heads",
        "double_stream_depth",
        "single_stream_depth",
        "mlp_ratio",
    }
    path = tmp_path / "ema_weights.pt"
    torch.save(model.state_dict(), path)
    (tmp_path / "transformer_config.json").write_text(json.dumps(config))
    rebuilt = ArtFlow.from_single_file(str(path))
    assert rebuilt.get_config() == config
    for key, value in model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key])


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError, match="unsupported architecture"):
        ArtFlow(
            hidden_size=32,
            num_heads=4,
            double_stream_depth=1,
            single_stream_depth=1,
            mlp_ratio=2,
            architecture="old-experiment",
        )


def test_missing_metadata_is_not_guessed(tmp_path):
    with pytest.raises(FileNotFoundError, match="transformer_config"):
        ArtFlow.from_single_file(str(tmp_path / "weights.pt"))
