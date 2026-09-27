"""Native model behavior with nonzero gates, padding, and variable capacity."""

import json

import pytest
import torch
from src.models.artflow import ArtFlow


@pytest.mark.parametrize("double,single", [(0, 2), (1, 2), (2, 0)])
def test_padding_does_not_affect_output_or_receive_gradients(double, single):
    torch.manual_seed(32)
    model = ArtFlow(
        hidden_size=32,
        num_heads=4,
        double_stream_depth=double,
        single_stream_depth=single,
        mlp_ratio=2,
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "modulation" in name or "final_layer.1" in name:
                param.normal_(0, 0.03)
    x, t = torch.randn(2, 16, 8, 8), torch.tensor([0.2, 0.8])
    txt, pooled = torch.randn(2, 7, 1024, requires_grad=True), torch.randn(2, 1024)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    output = model(x, t, txt, pooled, mask)
    changed = txt.detach().clone()
    changed[0, 4:] = 100 * torch.randn_like(changed[0, 4:])
    torch.testing.assert_close(model(x, t, changed, pooled, mask), output)
    output.square().mean().backward()
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert torch.count_nonzero(txt.grad[0, 4:]) == 0
    assert txt.grad[0, :4].abs().sum() > 0


@pytest.mark.parametrize("double,single", [(0, 2), (1, 2), (2, 0)])
def test_model_file_roundtrip_preserves_predictions(tmp_path, double, single):
    torch.manual_seed(9)
    model = ArtFlow(hidden_size=32, num_heads=4, double_stream_depth=double,
                    single_stream_depth=single, mlp_ratio=2.25)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)
    inputs = (torch.randn(2, 16, 8, 8), torch.tensor([0.2, 0.8]),
              torch.randn(2, 7, 1024), torch.randn(2, 1024))
    expected = model(*inputs)
    path = tmp_path / "ema_weights.pt"
    torch.save(model.state_dict(), path)
    (tmp_path / "transformer_config.json").write_text(json.dumps(model.get_config()))
    restored = ArtFlow.from_single_file(str(path))
    torch.testing.assert_close(restored(*inputs), expected)
