"""Rotary positions checked against geometric symmetry and known plane rotations."""

import torch

from src.models.dit_blocks import MSRoPE, apply_rotary_emb


def test_centered_grid_conjugate_symmetry():
    rope = MSRoPE(theta=10000, axes_dim=[4, 4], centered=True)
    frequencies, _ = rope((3, 5), 4, torch.device('cpu'))
    grid = frequencies.reshape(3, 5, -1)
    torch.testing.assert_close(grid, grid.flip((0, 1)).conj())
    torch.testing.assert_close(grid[1, 2], torch.ones_like(grid[1, 2]))


def test_rotary_embedding_matches_quarter_and_half_turns():
    vector = torch.tensor([[[[1., 2., 3., 4.]]]])
    frequencies = torch.tensor([[1j, -1 + 0j]])
    expected = torch.tensor([[[[-2., 1., -3., -4.]]]])
    torch.testing.assert_close(apply_rotary_emb(vector, frequencies), expected)
