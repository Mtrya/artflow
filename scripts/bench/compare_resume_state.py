"""Compare actual training state after uninterrupted and restarted updates.

Run on trusted checkpoints produced by this project. This deserializes optimizer
and RNG pickle files. Sampler queues may differ in prefetch representation, so
this comparison deliberately does not certify future row-stream equivalence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from src.train.stage_control import CHECKPOINT_RECORD, validate_checkpoint


def compare_values(reference, candidate):
    stats = dict(exact=True, tensors=0, max_absolute_difference=0.0)

    def visit(a, b):
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray):
                raise ValueError("state value type differs")
            a, b = torch.from_numpy(a), torch.from_numpy(b)
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("state tensor shape/dtype differs")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite state tensor")
            exact = torch.equal(a, b)
            stats["tensors"] += 1
            stats["exact"] &= exact
            if not exact and a.numel():
                difference = (a.double() - b.double()).abs().max().item()
                stats["max_absolute_difference"] = max(stats["max_absolute_difference"], difference)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError("state mapping keys differ")
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("state sequence differs")
            for x, y in zip(a, b):
                visit(x, y)
        else:
            stats["exact"] &= bool(a == b)

    visit(reference, candidate)
    return stats


def compare_checkpoints(reference, candidate):
    record = json.loads((reference / CHECKPOINT_RECORD).read_text())
    for root in (reference, candidate):
        validate_checkpoint(root, max_steps=record["max_steps"],
                            expected_step=record["global_step"], require_record=True,
                            scheduler_count=record["scheduler_count"],
                            use_ema=record["use_ema"], world_size=record["world_size"])
    names = ["model.safetensors" if (reference / "model.safetensors").is_file()
             else "pytorch_model.bin"]
    names += [f"{kind}.bin" if i == 0 else f"{kind}_{i}.bin"
              for kind in ("optimizer", "scheduler") for i in range(record["scheduler_count"])]
    if record["use_ema"]:
        names += ["ema_weights.pt"]
    names += [f"random_states_{rank}.pkl" for rank in range(record["world_size"])]
    results = {}
    for name in names:
        def read(root):
            if name.endswith(".safetensors"):
                return load_file(root / name, device="cpu")
            return torch.load(root / name, map_location="cpu", weights_only=False)
        results[name] = compare_values(read(reference), read(candidate))
    return dict(exact=all(result["exact"] for result in results.values()), files=results,
                step=record["global_step"], world_size=record["world_size"],
                caveat="Compares model, optimizers, schedulers, EMA and RNG. "
                       "Does not certify sampler replay or other rank counts/execution paths.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare_checkpoints(args.reference, args.candidate)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result), flush=True)
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
