"""Muon batched-Newton-Schulz equivalence and momentum routing tests."""

import torch

from src.pretrain.muon import (
    Muon,
    _zeropower_via_newtonschulz5,
    _zeropower_via_newtonschulz5_batched,
)


def _random_matrices(count, rows, cols, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(rows, cols, generator=generator) * (1.0 + index)
        for index in range(count)
    ]


def test_batched_ns_matches_per_matrix():
    matrices = _random_matrices(6, 32, 16, seed=1)
    reference = [_zeropower_via_newtonschulz5(m, steps=5) for m in matrices]
    batched = _zeropower_via_newtonschulz5_batched(torch.stack(matrices), steps=5)
    assert len(batched) == len(reference)
    for ref, fast in zip(reference, batched):
        assert ref.shape == fast.shape
        assert torch.allclose(ref.float(), fast.float(), atol=2e-2, rtol=2e-2)


def test_batched_ns_matches_per_matrix_transposed():
    matrices = _random_matrices(4, 16, 48, seed=2)
    reference = [_zeropower_via_newtonschulz5(m, steps=5) for m in matrices]
    batched = _zeropower_via_newtonschulz5_batched(torch.stack(matrices), steps=5)
    for ref, fast in zip(reference, batched):
        assert torch.allclose(ref.float(), fast.float(), atol=2e-2, rtol=2e-2)


def _step_with(seed=3):
    torch.manual_seed(seed)
    params = [
        torch.randn(48, 16),
        torch.randn(48, 16),  # same shape -> batched path groups them
        torch.randn(16, 16),
        torch.randn(64, 16),
    ]
    grads = [torch.randn_like(p) for p in params]
    params = [p.clone().requires_grad_(True) for p in params]
    for p, g in zip(params, grads):
        p.grad = g.clone()
    optimizer = Muon(
        [{"params": params[:2], "chunks": 3},
         {"params": params[2:], "chunks": 1}],
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
    )
    optimizer.step()
    return [p.detach().clone() for p in params]


def test_muon_step_batched_matches_per_matrix(monkeypatch):
    fast = _step_with()
    def per_matrix(stacked, steps):
        return [_zeropower_via_newtonschulz5(matrix, steps) for matrix in stacked]
    monkeypatch.setattr("src.pretrain.muon._zeropower_via_newtonschulz5_batched", per_matrix)
    reference = _step_with()
    for ref, out in zip(reference, fast):
        torch.testing.assert_close(ref, out, atol=5e-3, rtol=5e-3)


def test_muon_weight_decay_applied_once_per_param():
    """A chunked param must not be decayed once per chunk."""
    torch.manual_seed(4)
    param = torch.randn(48, 16).requires_grad_(True)
    # Zero gradient isolates the decay from the orthogonalized update.
    param.grad = torch.zeros_like(param)
    optimizer = Muon(
        [{"params": [param], "chunks": 3}], lr=0.02, momentum=0.95, weight_decay=0.5
    )
    before = param.detach().clone()
    optimizer.step()
    decayed = before * (1.0 - 0.02 * 0.5)
    torch.testing.assert_close(param, decayed, rtol=0, atol=0)
