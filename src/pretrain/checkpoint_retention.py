"""Prune older complete checkpoints only after publishing their replacement."""

from pathlib import Path
import re
import shutil

from .stage_control import validate_checkpoint


def prune_checkpoints(checkpoint, *, keep_last, max_steps, scheduler_count,
                      use_ema, world_size, protected_steps):
    """Keep recent recovery copies and every completed curriculum endpoint.

    Stages share one run directory; endpoints are explicitly protected.
    Incomplete, incompatible, symlinked and future checkpoints are untouched.
    The caller holds the run's writer lock and calls this after every rank has
    saved and the replacement's completion record has been published.
    """
    if type(keep_last) is not int or keep_last < 0:
        raise ValueError("checkpoint_keep_last must be a nonnegative integer")
    if any(type(step) is not int or not 0 <= step <= max_steps for step in protected_steps):
        raise ValueError("protected checkpoint steps must lie within the training schedule")
    protected_steps = set(protected_steps)
    if keep_last == 0:
        return []
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or checkpoint.parent.is_symlink():
        raise ValueError("checkpoint retention requires a real run directory")
    contract = dict(max_steps=max_steps, scheduler_count=scheduler_count, use_ema=use_ema,
                    world_size=world_size)
    current_step = validate_checkpoint(checkpoint, **contract)
    complete = []
    for path in checkpoint.parent.glob("checkpoint_step_*"):
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"checkpoint_step_\d+", path.name):
            continue
        try:
            step = validate_checkpoint(path, **contract)
        except (ValueError, OSError):
            continue
        if step <= current_step:
            complete.append((step, path))
    complete.sort(key=lambda item: item[0], reverse=True)
    # The just-published checkpoint must be in the retained set. Fail closed
    # if its record disappeared or changed during inventory inspection.
    if checkpoint not in [path for _, path in complete[:keep_last]]:
        raise ValueError("replacement checkpoint changed during retention preflight")
    removed = []
    for step, path in reversed(complete[keep_last:]):
        if step in protected_steps:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
