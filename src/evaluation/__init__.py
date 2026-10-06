"""Evaluation module for Inko.

Live evaluation is driven by the training loop and diagnostics:
- eval_loss: teacher-forced eval loss probes (length-binned)
- prompt_grid: periodic sample grids on the fixed monitoring panel
- kid_eval: KID against precomputed reference statistics
"""

from .metrics import calculate_kid
from .visualize import make_image_grid, visualize_denoising, format_prompt_caption

__all__ = [
    "calculate_kid",
    "make_image_grid",
    "visualize_denoising",
    "format_prompt_caption",
]
