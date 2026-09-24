"""Check initialization and branch-output scaling with fixed learned gates.

These local invariants do not establish long-run stability or bound gates,
norm gains, and arbitrary changes to internal branch weights.
"""

import torch

from src.models.artflow import ArtFlow
from src.models.dit_blocks import DoubleStreamDiTBlock, SingleStreamDiTBlock

TINY = dict(
    hidden_size=64,
    num_heads=4,
    double_stream_depth=2,
    single_stream_depth=2,
    mlp_ratio=2.0,
    conditioning_scheme="fused",
    txt_in_features=32,
    patch_size=2,
    in_channels=4,
)


def _build(seed=0, **flags):
    torch.manual_seed(seed)
    return ArtFlow(**TINY, **flags)


def _inputs(model, batch=2):
    torch.manual_seed(1)
    return [
        torch.randn(batch, model.in_channels, 16, 16),
        torch.rand(batch),
        torch.randn(batch, 12, model.txt_embedder.in_features),
        torch.randn(batch, model.txt_embedder.in_features),
    ]


def _forward(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(*inputs)


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


def test_flags_are_init_transparent():
    """AdaLN-zero: with every gate at 0 the branch never enters the residual
    stream and the conditioning vector only feeds zero-initialised modulations,
    so flipping either flag cannot change the output."""
    base = _build()
    inputs = _inputs(base)
    reference = _forward(base, inputs)

    for flags in ({"branch_norm": True}, {"cond_norm": True},
                  {"branch_norm": True, "cond_norm": True}):
        model = _build(**flags)
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected, unexpected
        # The only new tensors are the norms' unit scales.
        assert missing and all("norm" in key for key in missing), missing
        out = _forward(model, inputs)
        assert torch.equal(out, reference), f"{flags} changed the init output"


def test_single_stream_branch_norm_is_scale_invariant():
    torch.manual_seed(0)
    block = SingleStreamDiTBlock(
        dim=64, num_heads=4, c_dim=32, rope_axes_dim=[8, 8], branch_norm=True
    )
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
    block = DoubleStreamDiTBlock(
        dim=64, num_heads=4, c_dim=32, rope_axes_dim=[8, 8], branch_norm=True
    )
    for modulation in (block.modulation_img, block.modulation_txt):
        modulation[-1].bias.data.fill_(0.1)

    img, txt, c = torch.randn(2, 16, 64), torch.randn(2, 12, 64), torch.randn(2, 32)
    with torch.no_grad():
        before = block(img, txt, c, (4, 4), 12)
        _scale_branch_outputs(
            [block.attn.proj_img, block.attn.proj_txt,
             block.mlp_img.down_proj, block.mlp_txt.down_proj],
            50.0,
        )
        after = block(img, txt, c, (4, 4), 12)

    for a, b in zip(before, after):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-3), (a - b).abs().max()


def test_without_branch_norm_the_same_scaling_leaks_through():
    """Control: the same scaling moves the output by several times its own
    magnitude when the branch is not normalised."""
    torch.manual_seed(0)
    block = SingleStreamDiTBlock(
        dim=64, num_heads=4, c_dim=32, rope_axes_dim=[8, 8], branch_norm=False
    )
    block.modulation[-1].bias.data.fill_(0.1)

    img, txt, c = torch.randn(2, 16, 64), torch.randn(2, 12, 64), torch.randn(2, 32)
    with torch.no_grad():
        before = block(img, txt, c, (4, 4), 12)
        _scale_branch_outputs([block.attn.proj, block.mlp.down_proj], 50.0)
        after = block(img, txt, c, (4, 4), 12)

    delta = (after[0] - before[0]).abs().max()
    assert delta > 3 * before[0].abs().max(), delta


def test_cond_norm_removes_the_conditioning_input_offset():
    """The conditioning MLP's input carries a sample-independent DC offset --
    the sinusoid's slow dimensions sit at cos ~ 1 -- and that offset is what
    saturates c_mlp's SiLU. LayerNorm has to remove it."""
    stats = {}

    for name, model in (("raw", _build()), ("normed", _build(cond_norm=True))):
        captured = []
        handle = model.c_mlp.register_forward_pre_hook(
            lambda _module, args, sink=captured: sink.append(args[0].detach())
        )
        inputs = _inputs(model)
        for value in (0.001, 0.999):
            inputs[1] = torch.full_like(inputs[1], value)
            with torch.no_grad():
                model(*inputs)
        handle.remove()
        assert len(captured) == 2
        stats[name] = [
            (tensor.mean().item(), tensor.pow(2).mean().sqrt().item())
            for tensor in captured
        ]

    # A DC offset that no sample can change is present in the raw input ...
    offset = max(abs(mean) for mean, _ in stats["raw"])
    assert offset > 0.1, stats
    # ... and it is gone once the conditioning input is normalised.
    assert max(abs(mean) for mean, _ in stats["normed"]) < 1e-2, stats
    assert all(abs(rms - 1.0) < 0.05 for _, rms in stats["normed"]), stats


def test_flags_round_trip_through_get_config():
    flagged = ArtFlow(**{**TINY, "double_stream_depth": 0}, branch_norm=True,
                      cond_norm=True)
    flags = flagged.get_config()
    assert flags["branch_norm"] is True and flags["cond_norm"] is True
    plain = ArtFlow(**{**TINY, "double_stream_depth": 0}).get_config()
    assert plain["branch_norm"] is False and plain["cond_norm"] is False

    # The flags must also be recoverable from a checkpoint, otherwise a flagged
    # checkpoint could silently load into a model built with them off.
    state = flagged.state_dict()
    assert any(key.endswith(".norm_msa_out.weight") for key in state)
    assert "cond_norm.weight" in state
