"""Chunked Muon updates cross-checked with torch.optim.Muon."""

import pytest
import torch

from src.pretrain.muon import Muon


@pytest.mark.parametrize("shape,chunks", [
    ((16, 16), 1),
    ((64, 16), 1),
    ((16, 64), 1),
    ((48, 16), 3),
    ((128, 16), 2),
    ((32, 64), 2),
])
@pytest.mark.parametrize("ns_steps", [0, 5])
def test_updates_match_pytorch_per_chunk(shape, chunks, ns_steps):
    torch.manual_seed(7)
    # Independently optimized matrices are the reference for each fused chunk.
    params = [torch.nn.Parameter(torch.full(shape, value)) for value in (0.0, 1.0)]
    reference = [
        [torch.nn.Parameter(part.clone()) for part in param.detach().chunk(chunks)]
        for param in params
    ]
    actual_opt = Muon(
        [dict(params=params, chunks=chunks)],
        ns_steps=ns_steps, weight_decay=0.01,
    )
    reference_opt = torch.optim.Muon(
        [part for parts in reference for part in parts],
        lr=0.02, ns_steps=ns_steps, weight_decay=0.01, adjust_lr_fn="original",
    )
    for step in range(3):
        before = [param.detach().clone() for param in params]
        ref_before = [torch.cat(parts).detach().clone() for parts in reference]
        for param, parts in zip(params, reference):
            if ns_steps:
                grad = torch.randn_like(param)
            else:
                grad = torch.zeros_like(param)
                for index, chunk in enumerate(grad.chunk(chunks)):
                    chunk[step, step] = (index + 1) * (step + 1)
            param.grad = grad.clone()
            for part, chunk_grad in zip(parts, grad.chunk(chunks)):
                part.grad = chunk_grad.clone()
        actual_opt.step()
        reference_opt.step()
        for param, parts, old, ref_old in zip(params, reference, before, ref_before):
            if ns_steps:
                # Five BF16 iterations amplify rounding differences between
                # PyTorch's fused addmm and separate multiply/add operations.
                # Compare updates so initial weights cannot hide a bad result.
                actual = param.detach() - old
                expected = torch.cat(parts).detach() - ref_old
                assert (actual - expected).norm() / expected.norm() < 0.05
            else:
                torch.testing.assert_close(param, torch.cat(parts), rtol=1e-6, atol=1e-7)
