import copy

import pytest
import torch

from src.models.artflow import ArtFlow
from src.models.dit_blocks import set_real_rope
from src.train.train import parse_args


@pytest.mark.parametrize("real_rope", [False, True])
@pytest.mark.parametrize("fast_attn", [False, True])
def test_hoisted_double_rope_preserves_each_blocks_table_and_all_gradients(real_rope, fast_attn):
    torch.manual_seed(42)
    reference = ArtFlow(in_channels=4, txt_in_features=16, hidden_size=32,
                        num_heads=4, double_stream_depth=2, single_stream_depth=1,
                        mlp_ratio=2)
    # Nonzero modulation/output exercise attention rather than AdaLN-zero identity.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "modulation" in name or "final_layer" in name:
                parameter.normal_(std=0.1)
        # Reusing the first block's table for both blocks must fail this check.
        reference.blocks[1].attn.rope.pos_freqs.mul_(0.8)
    set_real_rope(reference, real_rope)
    candidate = copy.deepcopy(reference)
    latent = torch.randn(2, 4, 4, 6)
    text = torch.randn(2, 5, 16)
    timestep = torch.tensor([0.2, 0.7])
    mask = torch.tensor([[True, True, False, False, False], [False] * 5])
    outputs, input_gradients = [], []
    for model, hoist in ((reference, False), (candidate, True)):
        x, txt = latent.clone().requires_grad_(), text.clone().requires_grad_()
        output = model(x, timestep, txt, txt_mask=mask, fast_attn=fast_attn,
                       hoist_double_rope=hoist)
        output.square().mean().backward()
        outputs.append(output)
        input_gradients.append((x.grad, txt.grad))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for expected, actual in zip(input_gradients[0], input_gradients[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert expected.grad is not None and actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert reference.blocks[0].attn.qkv_img.weight.grad.abs().sum() > 0


def test_hoisting_is_an_explicit_default_off_training_flag():
    assert not parse_args().parse_args(["--config", "configs/base.toml"]).hoist_double_rope
    assert parse_args().parse_args(["--config", "configs/base.toml",
                                    "--hoist_double_rope"]).hoist_double_rope
