"""Compare compiled hero blocks after the measured 640p SDPA kernel speedup.

FP32 parameters/gradients, BF16 autocast, dynamic block compilation and cuDNN
flags match the trainer. Nonzero diagnostic modulation exercises all gradients;
this is not a proposed initialization change or an end-to-end throughput test.
Compare identical weights/inputs and alternate backend order in timed rounds.
"""

import argparse
import copy
import json
import statistics
import time

import torch

from scripts.bench.attention_backend_probe import errors
from src.models.dit_blocks import DoubleStreamDiTBlock, SingleStreamDiTBlock, set_real_rope, set_sdpa_backends
from src.train.config import load_config
from src.models.varlen_attention import mask_metadata, validate_runtime


def comparison_names(include_native, include_real_rope, compare_autotune=False):
    if include_real_rope and not include_native:
        raise ValueError("--include-real-rope requires --include-native")
    if compare_autotune:
        if not (include_native and include_real_rope):
            raise ValueError("--compare-autotune requires native attention and real RoPE")
        return ["NATIVE_FLASH_VARLEN_REAL_ROPE", "NATIVE_FLASH_VARLEN_REAL_ROPE_AUTOTUNE"]
    names = ["EFFICIENT_ATTENTION", "CUDNN_ATTENTION"]
    if include_native:
        names.append("NATIVE_FLASH_VARLEN")
    if include_real_rope:
        names.append("NATIVE_FLASH_VARLEN_REAL_ROPE")
    return names


def compile_mode(name):
    return "max-autotune-no-cudagraphs" if name.endswith("_AUTOTUNE") else "default"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-hw", type=int, nargs=2, default=[40, 40])
    parser.add_argument("--text", type=int, default=49)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--include-real-rope", action="store_true",
                        help="Compare benchmark-only real RoPE against the native-attention block")
    parser.add_argument("--compare-autotune", action="store_true",
                        help="Only compare default/autotuned dynamic blocks with native attention and real RoPE")
    parser.add_argument("--block-kind", choices=["single", "double"], default="single")
    parser.add_argument("--eager-reference", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable only to isolate compiler first-call/warmup failures")
    args = parser.parse_args()
    if min(args.batch, args.text, args.iterations, args.rounds, *args.image_hw) < 1:
        parser.error("shapes, iterations and rounds must be positive")
    try:
        names = comparison_names(args.include_native, args.include_real_rope, args.compare_autotune)
    except ValueError as exc:
        parser.error(str(exc))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.include_native:
        validate_runtime()
    cfg = load_config(args.config).model
    dim = cfg.hidden_size
    head_dim = dim // cfg.num_heads
    block_type = SingleStreamDiTBlock if args.block_kind == "single" else DoubleStreamDiTBlock
    modulation = cfg.single_stream_modulation if args.block_kind == "single" else cfg.double_stream_modulation
    base = block_type(
        dim, cfg.num_heads, dim, mlp_ratio=cfg.mlp_ratio,
        qkv_bias=cfg.qkv_bias, rope_axes_dim=[head_dim // 2, head_dim // 2],
        modulation_share=modulation, ffn_type=cfg.ffn_type,
        rope_centered=cfg.rope_centered_grid)
    # Do not call AdaLN-zero initialization: its identity block would hide
    # attention/FFN gradient discrepancies in this component comparison.
    image_tokens = args.image_hw[0] * args.image_hw[1]
    sequence = image_tokens + args.text
    inputs = tuple(torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for shape in
                   ((args.batch, image_tokens, dim), (args.batch, args.text, dim),
                    (args.batch, dim)))
    upstream = tuple(torch.randn_like(x) for x in inputs[:2])
    lengths = torch.linspace(0, args.text, args.batch, device="cuda").long()
    # Match the eight-element-aligned additive bias used by compiled attention.
    storage = torch.zeros(args.batch, 1, 1, (sequence + 7) // 8 * 8,
                          device="cuda", dtype=torch.bfloat16)
    bias = storage[..., :sequence]
    positions = torch.arange(sequence, device="cuda")
    bias.masked_fill_(positions[None, None, None, :] >=
                      (image_tokens + lengths)[:, None, None, None], -float("inf"))
    freqs = base.attn.rope.prepare_freqs(tuple(args.image_hw), args.text, inputs[0].device)
    torch.cuda.synchronize()
    started = time.monotonic()
    native_metadata = mask_metadata(bias[:, 0, 0] == 0) if args.include_native else None
    torch.cuda.synchronize()
    metadata_seconds = time.monotonic() - started
    print(json.dumps(dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                          cudnn_version=torch.backends.cudnn.version(), seed=args.seed,
                          batch=args.batch, image_hw=args.image_hw, text=args.text, block_kind=args.block_kind,
                          parameter_dtype="float32", activation_dtype="bfloat16",
                          compile_dynamic=True, cudnn_deterministic=True,
                          cudnn_benchmark=False, nonzero_diagnostic_modulation=True,
                          eager_reference=args.eager_reference,
                          compare_autotune=args.compare_autotune,
                          metadata_seconds=metadata_seconds,
                          component_only=True)), flush=True)
    variants = {}
    reference = None
    compiled_reference = None
    for name in names:
        select_backend(name)
        metadata = native_metadata if name.startswith("NATIVE_FLASH_VARLEN") else None
        live = copy.deepcopy(base).cuda().train()
        set_real_rope(live, "_REAL_ROPE" in name)
        # Establish an eager oracle before compiler execution. A broken compiled
        # reference must not make all subsequent comparisons look acceptable.
        if reference is None and args.eager_reference:
            eager_output = step(live, live, inputs, upstream, args, freqs, bias, None)
            torch.cuda.synchronize()
            reference = snapshot(live, inputs, eager_output)
            require_finite(reference, "eager reference")
            print(json.dumps(dict(backend=name, eager_reference_finite=True)), flush=True)
            del eager_output
        compiled = torch.compile(live, mode=compile_mode(name), dynamic=True)
        variants[name] = (live, compiled, metadata)
        torch.cuda.synchronize()
        started = time.monotonic()
        try:
            output = step(live, compiled, inputs, upstream, args, freqs, bias, metadata)
            torch.cuda.synchronize()
            cold = time.monotonic() - started
            values = snapshot(live, inputs, output)
            require_finite(values, f"compiled {name}")
            if reference is None:
                reference = {key: value.clone() for key, value in values.items()}
            agreement = {key: errors(value, reference[key])
                         for key, value in values.items()}
            paired = None
            if args.compare_autotune:
                if compiled_reference is None:
                    compiled_reference = values
                else:
                    paired = {key: errors(value, compiled_reference[key])
                              for key, value in values.items()}
            print(json.dumps(dict(backend=name, compile_mode=compile_mode(name),
                                  cold_seconds=cold, agreement=agreement,
                                  default_compiled_agreement=paired)), flush=True)
            del values, output
        except (RuntimeError, NotImplementedError) as exc:
            print(json.dumps(dict(backend=name, error=str(exc))), flush=True)
            if name == "EFFICIENT_ATTENTION":
                raise
            del variants[name]
    if len(variants) != len(names):
        raise RuntimeError("all selected compiled backends must work before paired timing")
    for round_index in range(args.rounds):
        order = list(variants) if round_index % 2 == 0 else list(reversed(variants))
        for name in order:
            select_backend(name)
            live, compiled, metadata = variants[name]
            # Warm for at least a second, not a handful of millisecond kernels:
            # the initial SDPA probe showed clock/warmup drift in its first case.
            until = time.monotonic() + 1.
            while time.monotonic() < until:
                step(live, compiled, inputs, upstream, args, freqs, bias, metadata)
                torch.cuda.synchronize()
            live.zero_grad(set_to_none=True)
            for value in inputs:
                value.grad = None
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            samples = []
            for _ in range(args.iterations):
                started = time.monotonic()
                step(live, compiled, inputs, upstream, args, freqs, bias, metadata)
                torch.cuda.synchronize()
                samples.append(1000 * (time.monotonic() - started))
            print(json.dumps(dict(backend=name, round=round_index,
                                  median_ms=statistics.median(samples), samples_ms=samples,
                                  incremental_peak_bytes=torch.cuda.max_memory_allocated()-baseline)),
                  flush=True)


def select_backend(name):
    # Native metadata selects the native path before the SDPA branch.
    set_sdpa_backends(["EFFICIENT_ATTENTION" if name.startswith("NATIVE_FLASH_VARLEN") else name])


def snapshot(live, inputs, output):
    values = dict(zip(("output_img", "output_txt"), output))
    values.update({f"input_grad_{i}": x.grad for i, x in enumerate(inputs)})
    values.update({f"parameter_grad_{key}": p.grad for key, p in live.named_parameters()})
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def require_finite(values, label):
    invalid = [key for key, value in values.items() if not torch.isfinite(value).all()]
    if invalid:
        raise RuntimeError(f"{label} contains nonfinite values: {invalid}")


def step(live, compiled, inputs, upstream, args, freqs, bias, metadata):
    live.zero_grad(set_to_none=True)
    for value in inputs:
        value.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.block_kind == "single":
            output = compiled(*inputs, tuple(args.image_hw), args.text,
                              rope_freqs=freqs, attn_bias=bias, attn_metadata=metadata)
        else:
            text_mask = bias[:, 0, 0, -args.text:] == 0
            output = compiled(*inputs, tuple(args.image_hw), args.text,
                              txt_attention_mask=text_mask, attn_metadata=metadata)
    torch.autograd.backward(output, upstream)
    return output


if __name__ == "__main__":
    main()
