"""Paired full-Muon-update check for selectively compiled square NS batches.

Use the frozen model's real parameter/chunk routing and compare parameters,
mutated gradients and momentum over multiple updates. Both models remain live,
so memory figures are component diagnostics, not full-training headroom.
No model forward, auxiliary AdamW, EMA, DDP or data pipeline is timed.
"""

import argparse
import copy
from dataclasses import asdict
import json
import statistics
import time

import torch

from src.models.artflow import ArtFlow
from src.train.config import load_config
from src.train.muon import Muon, build_param_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.iterations, args.warmup) < 1:
        parser.error("iterations and warmup must be positive")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = load_config(args.config)
    baseline_model = ArtFlow(**asdict(config.model)).cuda()
    models = [baseline_model, copy.deepcopy(baseline_model)]
    optimizers = []
    for model, enabled in zip(models, (False, True)):
        opts = build_param_groups(
            model, muon_lr=config.optim.muon_lr,
            muon_wd=config.optim.muon_wd,
            muon_momentum=config.optim.muon_momentum,
            compile_square_ns=enabled,
        )
        optimizers.append(next(opt for opt in opts if isinstance(opt, Muon)))
    parameters = [[p for group in opt.param_groups for p in group["params"]]
                  for opt in optimizers]
    print(json.dumps(dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                          seed=args.seed, component_only=True,
                          parameters=sum(p.numel() for p in models[0].parameters()),
                          muon_parameters=sum(p.numel() for p in parameters[0]),
                          parameter_dtype="float32", compilation="square NS batches only")),
          flush=True)
    names = ("reference", "compiled_square")
    timings = {name: [] for name in names}
    for index in range(args.warmup + args.iterations):
        for p, q in zip(*parameters):
            gradient = torch.randn_like(p)
            p.grad = gradient
            q.grad = gradient.clone()
        # Reverse order each iteration to reduce clock/order bias.
        for variant in ((0, 1) if index % 2 == 0 else (1, 0)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            optimizers[variant].step()
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            if index >= args.warmup:
                timings[names[variant]].append(seconds)
            print(json.dumps(dict(step=index, variant=names[variant], seconds=seconds,
                                  warmup=index < args.warmup,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated())),
                  flush=True)
        mismatches = []
        for parameter_index, (p, q) in enumerate(zip(*parameters)):
            states = [opt.state[value]["momentum_buffer"]
                      for opt, value in zip(optimizers, (p, q))]
            for kind, a, b in (("parameter", p, q), ("gradient", p.grad, q.grad),
                               ("momentum", *states)):
                if not torch.equal(a, b) or not bool(torch.isfinite(b).all()):
                    mismatches.append(dict(parameter=parameter_index, kind=kind,
                                           shape=list(p.shape),
                                           max_abs=float((a - b).abs().max())))
        print(json.dumps(dict(step=index, exact_finite_update=not mismatches,
                              mismatches=mismatches)), flush=True)
        if mismatches:
            raise RuntimeError("selective Muon compilation changed the update/state")
    print(json.dumps(dict(component_only=True, exact_finite_updates=True,
                          median_seconds={name: statistics.median(values)
                                          for name, values in timings.items()},
                          samples_seconds=timings)), flush=True)


if __name__ == "__main__":
    main()
