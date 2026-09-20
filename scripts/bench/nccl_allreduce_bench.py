#!/usr/bin/env python3
"""Time a DDP-sized gradient allreduce across the local ranks.

The hero model carries ~533M parameters; the boundary reduction moves
~2.1GB in fp32 per optimizer step. This benchmark times
allreduce on that message size so the measured per-step
collective cost can be compared against the gap between single-rank and
multi-rank step time. Ring-normalized bandwidth is a reporting convention,
not evidence that NCCL actually selected a ring algorithm. BF16 is an optional
diagnostic only; it is not the frozen hero gradient precision.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=532706812)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bucket-mib", type=int, default=25)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--trace-dir", type=Path,
                        help="Separate, untimed single/bucketed collective trace per rank")
    args = parser.parse_args(argv)
    if min(args.elements, args.iterations, args.bucket_mib) < 1:
        parser.error("elements, iterations and bucket size must be positive")
    return args


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 2:
        raise ValueError("communication probe requires at least two ranks")

    try:
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        buf = torch.full((args.elements,), rank + 1, dtype=dtype, device="cuda")
        nbytes = buf.numel() * buf.element_size()
        dist.all_reduce(buf)
        expected = world * (world + 1) // 2
        failed = torch.tensor(int(not bool((buf == expected).all())), device="cuda")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("allreduce correctness check failed")
        # Repeated SUM on nonzero input grows exponentially and overflows.
        # Zeros keep the timed payload finite without a fill in each interval.
        buf.zero_()

        def timed(ops):
            for _ in range(3):
                for op in ops:
                    op()
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_rank])
            times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                for op in ops:
                    op()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            # Use the slowest rank for each iteration, not rank zero alone.
            values = torch.tensor(times, dtype=torch.float64, device="cuda")
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            times = values.cpu().tolist()
            return statistics.median(times), min(times), times

        # Pattern A: one big allreduce (bandwidth).
        med, mn, single_times = timed([lambda: dist.all_reduce(buf)])
        # Pattern B: the same bytes as DDP's default 25MiB buckets (latency).
        bucket = args.bucket_mib * 2**20 // buf.element_size()
        chunks = list(buf.split(bucket))
        med_b, mn_b, bucket_times = timed([lambda c=c: dist.all_reduce(c) for c in chunks])
        # Verify the bucketed path independently, outside timed samples.
        buf.fill_(rank + 1)
        for chunk in chunks:
            dist.all_reduce(chunk)
        failed = torch.tensor(int(not bool((buf == expected).all())), device="cuda")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("bucketed allreduce correctness check failed")
        if args.trace_dir is not None:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            buf.zero_()
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_rank])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function("single_payload"):
                    dist.all_reduce(buf)
                    torch.cuda.synchronize()
                with torch.profiler.record_function("bucketed_payload"):
                    for chunk in chunks:
                        dist.all_reduce(chunk)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.trace_dir / f"rank-{rank}.json"))
        if rank == 0:
            # Ring allreduce per-rank traffic is 2*(N-1)/N * message.
            traffic = 2 * (world - 1) / world * nbytes
            print(json.dumps(dict(dtype=args.dtype, message_bytes=nbytes, world_size=world,
                                  correctness_passed=True, bucket_correctness_passed=True,
                                  slowest_rank_per_iteration=True,
                                  single_median_ms=med*1000, single_min_ms=mn*1000,
                                  ring_normalized_GiB_s=traffic/med/2**30,
                                  buckets=len(chunks), bucket_mib=args.bucket_mib,
                                  bucket_median_ms=med_b*1000, bucket_min_ms=mn_b*1000,
                                  single_samples_seconds=single_times, bucket_samples_seconds=bucket_times,
                                  torch_version=torch.__version__, nccl_version=torch.cuda.nccl.version(),
                                  nccl_algo=os.environ.get("NCCL_ALGO", "automatic"),
                                  nccl_proto=os.environ.get("NCCL_PROTO", "automatic"),
                                  trace_dir=str(args.trace_dir) if args.trace_dir else None,
                                  component_only=True)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
