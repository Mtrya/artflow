"""Bounded double-stream RoPE cache/compile comparison at recorded batch shapes.

Uses synthetic masks/inputs, FP32 parameters, BF16 activations, and no optimizer.
Reports Dynamo graph counts and cold/repeat timings, not hero throughput. Run
with a process timeout and isolated Inductor cache; use a different node from
concurrent full-pipeline timing workloads. This probe reads no training shards.
"""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch._dynamo.backends.registry import lookup_backend

from scripts.bench.attention_backend_probe import errors
from scripts.bench.attention_block_probe import snapshot, require_finite
from src.models.artflow import ArtFlow
from src.models.dit_blocks import DoubleStreamDiTBlock, set_real_rope, set_sdpa_backends
from src.models.varlen_attention import mask_metadata, validate_runtime
from src.train.config import load_config


def shapes_from_records(path, steps):
    records = {row["step"]: row for row in map(json.loads, path.read_text().splitlines())}
    shapes = []
    for step in steps:
        row = records[step]
        if len(row["shapes"]) != 1 or row["shapes"][0]["count"] != 1:
            raise ValueError("requires one-micro recorded updates")
        shape = tuple(row["shapes"][0]["shape"])
        if len(shape) != 5 or min(shape) < 1:
            raise ValueError("invalid recorded latent/text shape")
        shapes.append(shape)
    return shapes


def model_patch_size(config):
    # Patch geometry lives on ArtFlow, not ModelConfig. Meta construction uses
    # the trainer's exact constructor contract without allocating a full model.
    with torch.device("meta"):
        model = ArtFlow(**asdict(config))
    return model.patch_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--steps", default="1,2,3,4,41,44,54,57,76,98,111,115")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-flash-varlen", action="store_true")
    parser.add_argument("--real-rope", action="store_true")
    args = parser.parse_args()
    steps = [int(step) for step in args.steps.split(",")]
    if not 2 <= len(steps) <= 16:
        parser.error("use 2–16 explicitly selected recorded updates")
    if args.out.exists():
        parser.error("output already exists")
    shapes = shapes_from_records(args.records, steps)
    cfg = load_config(args.config).model
    patch_size = model_patch_size(cfg)
    if any(h % patch_size or w % patch_size for _, _, h, w, _ in shapes):
        parser.error("latent shapes must be divisible by the model patch size")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_sdpa_backends(["EFFICIENT_ATTENTION"])
    if args.native_flash_varlen:
        validate_runtime()
    head_dim = cfg.hidden_size // cfg.num_heads
    base = DoubleStreamDiTBlock(
        cfg.hidden_size, cfg.num_heads, cfg.hidden_size, mlp_ratio=cfg.mlp_ratio,
        qkv_bias=cfg.qkv_bias, rope_axes_dim=[head_dim // 2, head_dim // 2],
        modulation_share=cfg.double_stream_modulation, ffn_type=cfg.ffn_type,
        rope_centered=cfg.rope_centered_grid)
    # Constructor modulation is nonzero: do not replace it with AdaLN-zero identity.
    set_real_rope(base, args.real_rope)
    report = dict(torch_version=torch.__version__, component_only=True, steps=steps,
                  shapes=shapes, native_flash_varlen=args.native_flash_varlen,
                  real_rope=args.real_rope, variants={})
    oracles = []
    for name in ("eager", "compiled", "hoisted"):
        torch._dynamo.reset()
        torch._dynamo.config.recompile_limit = 512
        live = copy.deepcopy(base).cuda().train()
        graph_count = 0

        def backend(graph, inputs):
            nonlocal graph_count
            graph_count += 1
            return lookup_backend("inductor")(graph, inputs)

        forward = live if name == "eager" else torch.compile(live, dynamic=True, backend=backend)
        results = []
        for index, (batch, _, height, width, text) in enumerate(shapes):
            hw = (height // patch_size, width // patch_size)
            image_tokens = hw[0] * hw[1]
            gen = torch.Generator(device="cuda").manual_seed(1000 + index)
            inputs = tuple(torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                                       generator=gen, requires_grad=True) for shape in (
                (batch, image_tokens, cfg.hidden_size), (batch, text, cfg.hidden_size),
                (batch, cfg.hidden_size)))
            upstream = tuple(torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=gen)
                             for x in inputs[:2])
            lengths = torch.linspace(0, text, batch, device="cuda").long()
            mask = torch.arange(text, device="cuda")[None, :] < lengths[:, None]
            metadata = None
            if args.native_flash_varlen:
                metadata = mask_metadata(torch.cat([
                    torch.ones(batch, image_tokens, device="cuda", dtype=torch.bool), mask], 1))
            live.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                freqs = live.attn.rope(hw, text, inputs[0].device) if name == "hoisted" else None
                output = forward(*inputs, hw, text, mask, attn_metadata=metadata, rope_freqs=freqs)
            torch.autograd.backward(output, upstream)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            values = snapshot(live, inputs, output)
            require_finite(values, f"{name} at recorded update {steps[index]}")
            result = dict(recorded_step=steps[index], seconds=seconds, graphs=graph_count)
            if name == "eager":
                oracles.append(values)
            else:
                agreement = {key: errors(value, oracles[index][key]) for key, value in values.items()}
                worst = max(agreement, key=lambda key: agreement[key]["relative_l2"])
                result.update(max_relative_l2=agreement[worst]["relative_l2"], worst_tensor=worst,
                              max_absolute_difference=max(v["max_abs"] for v in agreement.values()))
            results.append(result)
            report["variants"][name] = results
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(variant=name, **result)), flush=True)
            del output, inputs, upstream, values
        del forward, live
    print("All selected forward/input-gradient/parameter-gradient states finite.", flush=True)


if __name__ == "__main__":
    main()
