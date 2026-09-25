"""Tests for the Muon optimizer (chunked orthogonalization, param routing)."""

import pytest
import torch

from src.models.artflow import ArtFlow
from src.pretrain.muon import Muon, build_param_groups, _chunk_hint


def _tiny_model():
    return ArtFlow(
        hidden_size=64,
        num_heads=4,
        double_stream_depth=1,
        single_stream_depth=1,
        mlp_ratio=2.67,
        )


def test_newtonschulz_orthogonalizes():
    from src.pretrain.muon import _zeropower_via_newtonschulz5

    torch.manual_seed(0)
    g = torch.randn(64, 32)
    out = _zeropower_via_newtonschulz5(g).float()
    # The 5-step quintic in bf16 is approximate: singular values land near 1
    # (measured [0.68, 1.13]) rather than exactly at 1 — this matches Keller
    # Jordan's reference behavior.
    sv = torch.linalg.svdvals(out)
    assert (sv > 0.55).all() and (sv < 1.2).all()
    # Cross-direction correlation stays small relative to unit diagonal
    gram = out.mT @ out
    off_diag = gram - torch.diag(torch.diag(gram))
    assert off_diag.abs().max() < 0.2




def test_chunk_hint_rules():
    assert _chunk_hint("blocks.0.attn.qkv.weight", torch.Size([192, 64])) == 3
    assert _chunk_hint("blocks.0.modulation.1.weight", torch.Size([384, 64])) == 6
    assert _chunk_hint("blocks.0.modulation.1.weight", torch.Size([192, 64])) == 3
    assert _chunk_hint("blocks.0.mlp.up_proj.weight", torch.Size([340, 64])) == 2
    assert _chunk_hint("blocks.0.attn.proj.weight", torch.Size([64, 64])) == 1


@pytest.mark.parametrize("shape,chunks", [
    ((16, 16), 1),
    ((64, 16), 1),
    ((16, 64), 1),
    ((48, 16), 3),
    ((128, 16), 2),
    ((32, 64), 2),
])
def test_original_scaling_matches_pytorch_per_chunk(shape, chunks):
    # Zero NS iterations isolate scaling, chunk partitioning, momentum and
    # decay from fused-vs-unfused bf16 polynomial rounding. Orthogonalization
    # and compiled routing are covered separately.
    params = [torch.nn.Parameter(torch.full(shape, value)) for value in (0.0, 1.0)]
    reference = [
        [torch.nn.Parameter(part.clone()) for part in param.detach().chunk(chunks)]
        for param in params
    ]
    actual_opt = Muon(
        [dict(params=params, chunks=chunks)],
        ns_steps=0, weight_decay=0.01,
    )
    reference_opt = torch.optim.Muon(
        [part for parts in reference for part in parts],
        lr=0.02, ns_steps=0, weight_decay=0.01, adjust_lr_fn="original",
    )
    for step in range(3):
        for param, parts in zip(params, reference):
            grad = torch.zeros_like(param)
            for index, chunk in enumerate(grad.chunk(chunks)):
                chunk[step, step] = (index + 1) * (step + 1)
            param.grad = grad.clone()
            for part, chunk_grad in zip(parts, grad.chunk(chunks)):
                part.grad = chunk_grad.clone()
        actual_opt.step()
        reference_opt.step()
        for param, parts in zip(params, reference):
            torch.testing.assert_close(param, torch.cat(parts), rtol=1e-6, atol=1e-7)


def test_param_groups_cover_each_param_once():
    model = _tiny_model()
    opts = build_param_groups(model, muon_lr=0.02, adam_lr=3e-4, muon_wd=.0015, adam_wd=.01, adam_conditioning_wd=.4, adam_eps=1e-8, adam_betas=(.9,.95), muon_momentum=.95)
    assert len(opts) == 2
    muon, adam = opts
    assert isinstance(muon, Muon) and isinstance(adam, torch.optim.AdamW)

    assigned = []
    for opt in opts:
        for group in opt.param_groups:
            assigned.extend(group["params"])
    total = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in assigned) == total
    assert len({id(p) for p in assigned}) == len(assigned)

    # Routing spot checks
    muon_names = {
        n
        for n, p in model.named_parameters()
        if any(p is q for g in muon.param_groups for q in g["params"])
    }
    assert "blocks.0.attn.qkv.weight" in muon_names or any(
        "qkv" in n for n in muon_names
    )
    assert not any("final_layer" in n for n in muon_names)
    assert not any("norm" in n.lower() for n in muon_names)

    # Chunk hints made it into the groups
    chunk_counts = sorted(g["chunks"] for g in muon.param_groups)
    assert 1 in chunk_counts and 3 in chunk_counts and 6 in chunk_counts


def test_muon_step_updates_and_decays():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(48, 16))
    opt = Muon([dict(params=[p], chunks=3)], lr=0.02, weight_decay=0.1)
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt.step()
    assert not torch.allclose(p, before)
    # weight decay shrinks the norm
    assert p.norm() < before.norm()


def test_muon_two_steps_momentum_state():
    p = torch.nn.Parameter(torch.randn(32, 32))
    opt = Muon([p], lr=0.02)
    for _ in range(2):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "momentum_buffer" in opt.state[p]
