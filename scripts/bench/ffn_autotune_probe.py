"""Bounded GEMM autotuning test on actual hero FFN token counts.

FP32 parameters/gradients and BF16 autocast match training. Compare unchanged
weights and dimensions, without CUDA graphs, at two recorded 640p shapes.
This synthetic FFN-only probe reads no datasets or checkpoints and provides
neither full-pipeline speed nor production memory evidence.
"""

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
import torch._inductor.config as inductor_config

from scripts.bench.attention_backend_probe import errors
from scripts.bench.attention_block_probe import require_finite
from src.models.dit_blocks import GatedFeedForward
from src.train.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[26128, 21816])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args(argv)
    if min(*args.tokens, args.iterations, args.rounds) < 1:
        parser.error("token counts, iterations and rounds must be positive")
    return args


def build_ffn(config):
    if config.ffn_type != "gated":
        raise ValueError("this diagnostic targets the hero gated FFN")
    return GatedFeedForward(config.hidden_size, int(config.hidden_size * config.mlp_ratio))


def snapshot(live, x, y):
    values = dict(output=y, input_grad=x.grad)
    values.update({f"parameter_grad_{key}": p.grad for key, p in live.named_parameters()})
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def main():
    args = parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite an existing probe result")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = load_config(args.config).model
    base = build_ffn(config)
    report = dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  component_only=True, parameter_dtype="float32", activation_dtype="bfloat16",
                  hidden_size=config.hidden_size, ffn_width=int(config.hidden_size * config.mlp_ratio),
                  compile_dynamic=False, cuda_graphs=False,
                  gemm_backends=inductor_config.max_autotune_gemm_backends, shapes=[])

    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    for tokens in args.tokens:
        x = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        upstream = torch.randn_like(x)

        def step(live, forward):
            live.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = forward(x)
            output.backward(upstream)
            return output

        eager = copy.deepcopy(base).cuda().train()
        y = step(eager, eager)
        torch.cuda.synchronize()
        oracle = snapshot(eager, x, y)
        require_finite(oracle, "eager")
        del y, eager
        row = dict(tokens=tokens, variants={})
        report["shapes"].append(row)
        variants = {}
        reference = None
        for name, mode in (("default", "default"), ("autotuned", "max-autotune-no-cudagraphs")):
            live = copy.deepcopy(base).cuda().train()
            if any(p.dtype != torch.float32 for p in live.parameters()):
                raise ValueError("parameter precision differs from the trainer")
            forward = (
                torch.compile(live, dynamic=False,
                              options={"max_autotune": False, "triton.cudagraphs": False})
                if name == "default" else torch.compile(live, mode=mode, dynamic=False)
            )
            torch.cuda.synchronize()
            start = time.monotonic()
            y = step(live, forward)
            torch.cuda.synchronize()
            cold = time.monotonic() - start
            values = snapshot(live, x, y)
            require_finite(values, name)
            agreement = {key: errors(value, oracle[key]) for key, value in values.items()}
            paired = None if reference is None else {
                key: errors(value, reference[key]) for key, value in values.items()}
            if reference is None:
                reference = values
            row["variants"][name] = dict(mode=mode, cold_seconds=cold,
                                         eager_agreement=agreement, default_agreement=paired,
                                         rounds=[])
            variants[name] = (live, forward)
            save()
            print(json.dumps(dict(tokens=tokens, variant=name, cold_seconds=cold,
                                  eager_agreement=agreement, default_agreement=paired)), flush=True)
            del y, values
        for round_index in range(args.rounds):
            names = list(variants) if round_index % 2 == 0 else list(reversed(variants))
            for name in names:
                live, forward = variants[name]
                for _ in range(3):
                    step(live, forward)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.iterations):
                    start = time.monotonic()
                    step(live, forward)
                    torch.cuda.synchronize()
                    times.append(time.monotonic() - start)
                result = dict(round=round_index, median_seconds=statistics.median(times),
                              mean_seconds=statistics.mean(times), samples_seconds=times,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
                row["variants"][name]["rounds"].append(result)
                save()
                print(json.dumps(dict(tokens=tokens, variant=name, **result)), flush=True)
        del variants, live, forward, x, upstream, oracle, reference
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
