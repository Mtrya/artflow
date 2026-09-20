"""Paired actual DDP communication hooks at identical FP32 bucket boundaries.

Includes compression, averaging, collective and decompression in each timing.
No DiT compute overlaps these collectives: component savings are not a full-step
speedup. Synthetic numerical comparisons expose lossiness, not training quality
or optimizer equivalence. This script never registers a production model hook.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    allreduce_hook, bf16_compress_hook,
)


class BufferBucket:
    """Minimal bucket interface consumed by PyTorch's eager default hooks."""

    def __init__(self, tensor):
        self.tensor = tensor

    def buffer(self):
        return self.tensor


def apply_hooks(tensor, hook, bucket_elements):
    futures = [hook(None, BufferBucket(chunk)) for chunk in tensor.split(bucket_elements)]
    for future in futures:
        future.wait()


def error_metrics(actual, reference):
    delta = actual.double() - reference.double()
    norm = torch.linalg.vector_norm(reference.double())
    return dict(finite=bool(torch.isfinite(actual).all()),
                exact=bool(torch.equal(actual, reference)),
                max_abs=float(delta.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(delta) / norm.clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=532706812)
    parser.add_argument("--bucket-mib", type=int, default=25,
                        help="FP32 bucket size, preserved when compressing the wire payload")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if min(args.elements, args.bucket_mib, args.rounds, args.iterations) < 1:
        parser.error("sizes, rounds and iterations must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world < 2:
            raise ValueError("need at least two ranks")
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[local_rank])
        bucket_elements = args.bucket_mib * 2**20 // 4
        hooks = {"fp32": allreduce_hook, "bf16": bf16_compress_hook}

        # Small, untimed numerical cases. Independent rank gradients and a
        # cancellation case are deliberately less forgiving than rank constants.
        torch.manual_seed(9182 + rank)
        noise = torch.randn(65536, device="cuda", dtype=torch.float32)
        cases = {"independent": noise,
                 "cancellation": noise * 1e-4 + (1.0 if rank % 2 == 0 else -1.0)}
        checks = {}
        for name, source in cases.items():
            reference, compressed = source.clone(), source.clone()
            apply_hooks(reference, hooks["fp32"], 8192)
            apply_hooks(compressed, hooks["bf16"], 8192)
            torch.cuda.synchronize()
            checks[name] = error_metrics(compressed, reference)
            if not checks[name]["finite"]:
                raise RuntimeError("nonfinite compressed synthetic reduction")
        del cases, source, reference, compressed, noise

        # Zeros prevent repeated averaging from changing/overflowing the timed
        # payload. They do not remove any communication or conversion kernels.
        buffer = torch.zeros(args.elements, device="cuda", dtype=torch.float32)
        samples = {name: [] for name in hooks}
        rounds = []
        for round_index in range(args.rounds):
            order = ("fp32", "bf16") if round_index % 2 == 0 else ("bf16", "fp32")
            for name in order:
                hook = hooks[name]
                for _ in range(3):
                    apply_hooks(buffer, hook, bucket_elements)
                torch.cuda.synchronize()
                dist.barrier(device_ids=[local_rank])
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    apply_hooks(buffer, hook, bucket_elements)
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - start)
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                times = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
                elapsed = times.cpu().tolist()
                samples[name].extend(elapsed)
                rounds.append(dict(round=round_index, hook=name, samples_seconds=elapsed,
                                   median_seconds=statistics.median(elapsed),
                                   peak_allocated_bytes=peak_allocated,
                                   peak_reserved_bytes=peak_reserved))
        if args.trace:
            for name, hook in hooks.items():
                dist.barrier(device_ids=[local_rank])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    apply_hooks(buffer, hook, bucket_elements)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.out / f"rank-{rank}-{name}.trace.json"))
        result = dict(rank=rank, world_size=world, torch_version=torch.__version__,
                      nccl_version=torch.cuda.nccl.version(), elements=args.elements,
                      fp32_bucket_elements=bucket_elements,
                      bucket_count=(args.elements + bucket_elements - 1) // bucket_elements,
                      fp32_message_bytes=args.elements * 4, bf16_message_bytes=args.elements * 2,
                      rounds=rounds, numerical_checks=checks,
                      median_seconds={name: statistics.median(values) for name, values in samples.items()},
                      slowest_rank_per_iteration=True, component_only=True,
                      gradient_equivalence_established=False, production_hook_changed=False)
        (args.out / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
