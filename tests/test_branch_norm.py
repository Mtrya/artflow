"""Branch normalization checked by rescaling branch outputs with fixed learned gates."""

import torch

from src.models.dit_blocks import DoubleStreamDiTBlock, SingleStreamDiTBlock


def _scale_branch_outputs(linears, factor):
    """Scale each branch's output by exactly `factor`.

    The projections named by the caller are the last linearities of their
    branches; weight *and* bias have to move, because an nn.Linear's default
    bias would otherwise stop the branch output from scaling cleanly.
    """
    with torch.no_grad():
        for linear in linears:
            linear.weight.mul_(factor)
            linear.bias.mul_(factor)


def test_single_stream_branch_norm_is_scale_invariant():
    torch.manual_seed(0)
    block = SingleStreamDiTBlock(dim=64, num_heads=4, c_dim=32, rope_axes_dim=[8, 8])
    # Wake the gates up: adaLN-zero would hide the branch entirely.
    block.modulation[-1].bias.data.fill_(0.1)

    img, txt, c = torch.randn(2, 16, 64), torch.randn(2, 12, 64), torch.randn(2, 32)
    with torch.no_grad():
        before = block(img, txt, c, (4, 4), 12)
        _scale_branch_outputs([block.attn.proj, block.mlp.down_proj], 50.0)
        after = block(img, txt, c, (4, 4), 12)

    for a, b in zip(before, after):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-3), (a - b).abs().max()


def test_double_stream_branch_norm_is_scale_invariant():
    torch.manual_seed(0)
    block = DoubleStreamDiTBlock(dim=64, num_heads=4, c_dim=32, rope_axes_dim=[8, 8])
    for modulation in (block.modulation_img, block.modulation_txt):
        modulation[-1].bias.data.fill_(0.1)

    img, txt, c = torch.randn(2, 16, 64), torch.randn(2, 12, 64), torch.randn(2, 32)
    with torch.no_grad():
        before = block(img, txt, c, (4, 4), 12)
        _scale_branch_outputs(
            [
                block.attn.proj_img,
                block.attn.proj_txt,
                block.mlp_img.down_proj,
                block.mlp_txt.down_proj,
            ],
            50.0,
        )
        after = block(img, txt, c, (4, 4), 12)

    for a, b in zip(before, after):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-3), (a - b).abs().max()
