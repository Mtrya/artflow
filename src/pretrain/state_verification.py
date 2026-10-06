"""Opt-in exact post-load verification, before any stochastic training update.

This is deliberately separate from comparing post-update trajectories: native
attention backward can be nondeterministic even on identical inputs. Reading a
checkpoint here adds CPU copies and I/O; leave it out of throughput timing.
"""
from pathlib import Path

import torch


def require_exact_state(expected, actual, *, label):
    tensors = 0

    def visit(a, b, path):
        nonlocal tensors
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"{path}: restored tensor shape/dtype differs")
            a, b = a.detach().cpu(), b.detach().cpu()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                raise ValueError(f"{path}: nonfinite restored/checkpoint tensor")
            if not torch.equal(a, b):
                raise ValueError(f"{path}: restored tensor differs from checkpoint")
            tensors += 1
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError(f"{path}: restored mapping keys differ")
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError(f"{path}: restored sequence differs")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        elif a != b:
            raise ValueError(f"{path}: restored scalar differs from checkpoint")

    visit(expected, actual, label)
    return dict(exact=True, tensors=tensors)


def verify_restored_training_state(checkpoint, model, optimizers, schedulers, ema):
    """Compare live loaded state, not a second deserialization of the same file."""
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    live_states = [("model.safetensors", model)]
    for kind, objects in (("optimizer", optimizers), ("scheduler", schedulers)):
        live_states.extend((f"{kind}.bin" if index == 0 else f"{kind}_{index}.bin", obj)
                           for index, obj in enumerate(objects))
    if ema is not None:
        live_states.append(("ema_weights.pt", ema))
    checks = {}
    for name, obj in live_states:
        saved = (load_file(str(checkpoint / name), device="cpu") if name.endswith(".safetensors")
                 else torch.load(checkpoint / name, map_location="cpu", weights_only=False))
        checks[name] = require_exact_state(saved, obj.state_dict(), label=name)
        del saved
    return dict(exact=True, files=checks,
                scope="Live model, optimizer, scheduler and EMA immediately after full loading; "
                      "RNG is checked separately. Not a post-update numerical equality claim.")
