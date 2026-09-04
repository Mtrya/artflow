"""Tests for the centered-grid MSRoPE option (2.1 A/B axis)."""

import torch

from src.models.dit_blocks import MSRoPE


def test_centered_grid_conjugate_symmetry():
    rope = MSRoPE(theta=10000, axes_dim=[4, 4], centered=True)
    img_freqs, _ = rope((4, 4), 3, torch.device("cpu"))
    grid = img_freqs.reshape(4, 4, -1)
    # Positions symmetric about the origin must have conjugate frequencies
    assert torch.allclose(grid[0, 0], grid[3, 3].conj(), atol=1e-5)
    assert torch.allclose(grid[1, 2], grid[2, 1].conj(), atol=1e-5)
    # Non-centered positions are at half-integer offsets, never at 0..3
    corner = MSRoPE(theta=10000, axes_dim=[4, 4], centered=False)
    corner_freqs, _ = corner((4, 4), 3, torch.device("cpu"))
    assert not torch.allclose(img_freqs, corner_freqs)


def test_centered_keeps_text_positions():
    corner = MSRoPE(theta=10000, axes_dim=[4, 4], centered=False)
    centered = MSRoPE(theta=10000, axes_dim=[4, 4], centered=True)
    _, t_corner = corner((4, 4), 3, torch.device("cpu"))
    _, t_centered = centered((4, 4), 3, torch.device("cpu"))
    # Text stays pinned to the same fixed diagonal in both modes
    assert torch.allclose(t_corner, t_centered)


def test_centered_asymmetric_shapes():
    rope = MSRoPE(theta=10000, axes_dim=[4, 4], centered=True)
    img_freqs, txt_freqs = rope((3, 5), 4, torch.device("cpu"))
    assert img_freqs.shape[0] == 15
    assert txt_freqs.shape[0] == 4
