"""
Visualization utilities for evaluation.

Functions:
- make_image_grid: Create and optionally save a grid of images
- visualize_denoising: Visualize the denoising process
- format_prompt_caption: Format prompts for display in image captions
"""

import os
from typing import List, Optional

import numpy as np
import torch
import torchvision


def make_image_grid(
    images: torch.Tensor,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    value_range: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Create a grid of images and optionally save it.

    Args:
        images: Tensor of shape [B, C, H, W]
        rows: Number of rows (optional)
        cols: Number of columns (optional)
        save_path: Path to save the grid image
        normalize: Whether to normalize images to [0, 1]
        value_range: Range of values in input images (min, max)

    Returns:
        Grid tensor
    """
    if rows is None and cols is None:
        nrow = int(np.ceil(np.sqrt(images.shape[0])))
    elif cols is not None:
        nrow = cols
    else:
        nrow = int(np.ceil(images.shape[0] / rows))

    grid = torchvision.utils.make_grid(
        images, nrow=nrow, normalize=normalize, value_range=value_range, padding=2
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(grid, save_path)

    return grid


def visualize_denoising(
    intermediate_steps: List[torch.Tensor], save_path: str, num_steps_to_show: int = 10
):
    """
    Visualize the denoising process by selecting a subset of steps.

    Args:
        intermediate_steps: List of tensors [B, C, H, W] from the sampling process
        save_path: Path to save the visualization
        num_steps_to_show: Number of steps to display
    """
    total_steps = len(intermediate_steps)
    if total_steps < num_steps_to_show:
        indices = list(range(total_steps))
    else:
        indices = np.linspace(0, total_steps - 1, num_steps_to_show, dtype=int).tolist()

    selected_steps = [intermediate_steps[i] for i in indices]

    # Take the first sample from the batch for visualization
    first_sample_steps = [step[0] for step in selected_steps]  # List of [C, H, W]

    # Stack them: [Num_steps, C, H, W]
    stacked = torch.stack(first_sample_steps)

    # Make grid: 1 row, Num_steps columns
    make_image_grid(
        stacked,
        rows=1,
        cols=len(selected_steps),
        save_path=save_path,
        normalize=True,
        value_range=(-1, 1),
    )


def format_prompt_caption(prompts: List[str], limit: int = 32) -> str:
    """
    Format a list of prompts for display as an image caption.

    Args:
        prompts: List of prompt strings
        limit: Maximum number of prompts to include

    Returns:
        Formatted caption string
    """
    if not prompts:
        return ""
    trimmed = [p.replace("\n", " ").strip() for p in prompts[:limit]]
    lines = [f"{idx + 1}. {text}" for idx, text in enumerate(trimmed)]
    remaining = len(prompts) - len(trimmed)
    if remaining > 0:
        lines.append(f"... (+{remaining} more)")
    return "\n\n".join(lines)

