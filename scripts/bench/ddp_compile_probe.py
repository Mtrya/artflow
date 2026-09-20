"""Bounded full-size DiT backward/DDP diagnostic without dataset/encoder startup.

Tests lazy per-block compilation under Accelerate/DDP and captures communication
overlap at explicit synthetic shapes. FP32 parameters/gradients, BF16 autocast,
and non-final-micro no_sync match training. No optimizer/EMA/data work is timed:
these are not end-to-end hero rates or full training correctness evidence.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
import torch.distributed as dist

from src.models.artflow import ArtFlow
from src.models.dit_blocks import MSRoPE
from src.train.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--micro", type=int, default=64)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--text", type=int, nargs="+", default=[128, 192])
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--disable-ddp-compile-split", action="store_true")
    parser.add_argument("--native-flash-varlen", action="store_true")
    parser.add_argument("--gradient-bucket-views", action="store_true",
                        help="Alias gradients into DDP buckets and zero in place to retain those views")
    parser.add_argument("--no-broadcast-buffers", action="store_true",
                        help="Probe one initial RoPE buffer sync instead of repeated forward syncs")
    parser.add_argument("--save-final-gradients", action="store_true")
    parser.add_argument("--reference-gradients", type=Path,
                        help="Compare every final gradient exactly with a prior matched probe directory")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args(argv)
    if min(args.micro, args.latent, args.accumulation, args.steps, *args.text) < 1:
        parser.error("shapes, accumulation and steps must be positive")
    if args.warmup < len(args.text) or args.steps <= args.warmup:
        parser.error("warmup must cover all text shapes and leave measured steps")
    if args.trace and args.steps < args.warmup + 2:
        parser.error("trace requires two post-warmup steps")
    return args


def make_inputs(model, args, device):
    """Use the instantiated model's input contract, not shape-config fields.

    ModelConfig intentionally omits the fixed latent/text feature dimensions.
    Keep this preparation CPU-testable before allocating distributed GPUs.
    """
    inputs = []
    for length in args.text:
        x = torch.randn(args.micro, model.in_channels, args.latent, args.latent,
                        device=device, dtype=torch.bfloat16)
        target = torch.randn_like(x)
        text = torch.randn(args.micro, length, model.txt_embedder.in_features,
                           device=device, dtype=torch.bfloat16)
        pooled = torch.randn(args.micro, model.txt_embedder.in_features,
                             device=device, dtype=torch.bfloat16)
        keep = torch.linspace(0, length, args.micro, device=device).long()
        mask = (torch.arange(length, device=device)[None, :] < keep[:, None]).long()
        inputs.append((x, target, text, pooled, mask, torch.rand(args.micro, device=device)))
    return inputs


def static_rope_buffers(model, *, max_position):
    """Reject mutable/unknown buffers or a workload that expands the RoPE table."""
    buffers = dict(model.named_buffers())
    expected = {f"{name}.pos_freqs": module.pos_freqs
                for name, module in model.named_modules() if isinstance(module, MSRoPE)}
    if not buffers or buffers.keys() != expected.keys():
        raise ValueError("buffer-sync probe only supports the model's RoPE tables")
    for name, value in buffers.items():
        if value is not expected[name] or value.requires_grad or value.shape[0] < max_position:
            raise ValueError(f"mutable or expanding buffer in probe: {name}")
    return buffers


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=1, mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(
            gradient_as_bucket_view=args.gradient_bucket_views,
            broadcast_buffers=not args.no_broadcast_buffers)],
    )
    if accelerator.num_processes < 2:
        raise ValueError("this probe requires at least two ranks")
    rank = accelerator.process_index
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = load_config(args.config)
    model = ArtFlow(**asdict(config.model))
    # Nonzero diagnostic output weights exercise upstream gradients on step 1;
    # not a proposed change to the model's production initialization.
    torch.nn.init.normal_(model.final_layer[1].weight, std=.01)
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = 512
    if args.disable_ddp_compile_split:
        dynamo.config.optimize_ddp = False
    for block in model.blocks:
        block.forward = torch.compile(block.forward, mode="default", dynamic=True)
    raw = model
    model = accelerator.prepare(model)
    buffer_snapshot = None
    if args.no_broadcast_buffers:
        buffers = static_rope_buffers(raw, max_position=args.latent // raw.patch_size + max(args.text))
        # Do not rely on version-specific DDP initial-buffer-sync behavior when
        # broadcast_buffers=False. Synchronize these tables explicitly once.
        for value in buffers.values():
            dist.broadcast(value, src=0)
        buffer_snapshot = {name: value.detach().cpu().clone() for name, value in buffers.items()}
    model.train()
    torch.manual_seed(1234 + rank)
    device = accelerator.device
    inputs = make_inputs(raw, args, device)
    metadata = dict(rank=rank, world_size=accelerator.num_processes,
                    parameters=sum(p.numel() for p in raw.parameters()),
                    gradient_bytes=sum(p.numel() * p.element_size() for p in raw.parameters()),
                    torch_version=torch.__version__, micro=args.micro, latent=args.latent,
                    text=args.text, accumulation=args.accumulation,
                    disable_ddp_compile_split=args.disable_ddp_compile_split,
                    native_flash_varlen=args.native_flash_varlen,
                    gradient_bucket_views=args.gradient_bucket_views,
                    broadcast_buffers=not args.no_broadcast_buffers,
                    zero_grad_set_to_none=not args.gradient_bucket_views,
                    synthetic_nonzero_output_initialization=True, component_only=True)
    print(json.dumps(metadata), flush=True)
    if args.reference_gradients:
        reference_metadata = json.loads(
            (args.reference_gradients / f"rank-{rank}.metadata.json").read_text())
        for key in ("rank", "world_size", "parameters", "gradient_bytes", "torch_version",
                    "micro", "latent", "text", "accumulation", "native_flash_varlen",
                    "synthetic_nonzero_output_initialization"):
            if reference_metadata.get(key) != metadata[key]:
                raise ValueError(f"gradient reference mismatch for {key}")
    (args.out / f"rank-{rank}.metadata.json").write_text(json.dumps(metadata, indent=2))
    profiler = None
    with (args.out / f"rank-{rank}.jsonl").open("x", buffering=1) as log:
        for step in range(args.steps):
            if args.trace and step == args.warmup:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True)
                profiler.start()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            # Retain gradient aliases after their first DDP iteration. Include
            # the extra zeroing work in the timed candidate policy.
            model.zero_grad(set_to_none=not args.gradient_bucket_views)
            for micro in range(args.accumulation):
                x, target, text, pooled, mask, t = inputs[(step + micro) % len(inputs)]
                sync = nullcontext() if micro == args.accumulation - 1 else accelerator.no_sync(model)
                with sync, accelerator.autocast():
                    output = model(x, t, txt=text, txt_pooled=pooled, txt_mask=mask,
                                   fast_attn=True, native_flash_varlen=args.native_flash_varlen)
                    loss = (output.float() - target.float()).square().mean() / args.accumulation
                    accelerator.backward(loss)
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            row = dict(step=step, seconds=seconds, loss=float(loss.detach()),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       profiled=profiler is not None, warmup=step < args.warmup)
            log.write(json.dumps(row) + "\n")
            print(json.dumps(dict(rank=rank, **row)), flush=True)
            if profiler is not None and step == args.warmup + 1:
                profiler.stop()
                profiler.export_chrome_trace(str(args.out / f"rank-{rank}.trace.json"))
                profiler = None
    if buffer_snapshot is not None:
        current = dict(raw.named_buffers())
        unchanged = (current.keys() == buffer_snapshot.keys() and all(
            torch.equal(current[name].detach().cpu(), value)
            for name, value in buffer_snapshot.items()))
        failed = torch.tensor(int(not unchanged), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("buffer-sync probe changed a supposedly static buffer")
        print(json.dumps(dict(rank=rank, rope_buffers_unchanged=True,
                              buffer_bytes=sum(v.numel() * v.element_size()
                                               for v in buffer_snapshot.values()))), flush=True)
    # These untimed aggregate checks detect gross nonfinite/divergent reductions;
    # they do not prove elementwise agreement with an independent reference.
    stats = torch.stack([torch.stack([p.grad.double().sum(), p.grad.double().square().sum()])
                         for p in raw.parameters() if p.grad is not None]).sum(0)
    gathered = [torch.empty_like(stats) for _ in range(accelerator.num_processes)]
    dist.all_gather(gathered, stats)
    finite = all(bool(torch.isfinite(value).all()) for value in gathered)
    agreement = all(torch.equal(gathered[0], value) for value in gathered[1:])
    print(json.dumps(dict(rank=rank, finite_gradient_statistics=finite,
                          rank_gradient_statistics_equal=agreement)), flush=True)
    if not finite or not agreement:
        raise RuntimeError("nonfinite or inconsistent reduced gradient statistics")
    if args.save_final_gradients or args.reference_gradients:
        gradients = {name: p.grad.detach().cpu() for name, p in raw.named_parameters()
                     if p.grad is not None}
        if args.reference_gradients:
            reference = torch.load(args.reference_gradients / f"rank-{rank}.gradients.pt",
                                   map_location="cpu", weights_only=True)
            if gradients.keys() != reference.keys():
                raise RuntimeError("gradient reference parameter set differs")
            mismatches = {name: float((value - reference[name]).abs().max())
                          for name, value in gradients.items()
                          if not torch.equal(value, reference[name])}
            print(json.dumps(dict(rank=rank, exact_reference_gradients=not mismatches,
                                  gradient_mismatches=mismatches)), flush=True)
            if mismatches:
                raise RuntimeError("DDP execution policy changed final gradients")
        if args.save_final_gradients:
            torch.save(gradients, args.out / f"rank-{rank}.gradients.pt")
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
