#!/usr/bin/env python3
"""Time a DDP-sized gradient allreduce across the local ranks.

The hero model carries ~533M parameters; the boundary reduction moves
~1.1GB in bf16 or ~2.1GB in fp32 per optimizer step. This benchmark times
ring allreduce on exactly those message sizes so the measured per-step
collective cost can be compared against the gap between single-rank and
multi-rank step time.
"""
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    world = dist.get_world_size()

    for dtype, label in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        numel = 533_000_000
        buf = torch.ones(numel, dtype=dtype, device="cuda")
        nbytes = buf.numel() * buf.element_size()

        def timed(ops):
            for op in ops[:3]:
                op()
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                for op in ops:
                    op()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            times.sort()
            return times[len(times) // 2], times[0]

        # Pattern A: one big allreduce (bandwidth).
        med, mn = timed([lambda: dist.all_reduce(buf)])
        # Pattern B: the same bytes as DDP's default 25MiB buckets (latency).
        bucket = 25 * 2**20 // buf.element_size()
        chunks = list(buf.split(bucket))
        med_b, mn_b = timed([lambda c=c: dist.all_reduce(c) for c in chunks])
        if rank == 0:
            # Ring allreduce per-rank traffic is 2*(N-1)/N * message.
            traffic = 2 * (world - 1) / world * nbytes
            print(
                f"{label}: message={nbytes/2**30:.2f}GiB world={world} | "
                f"single median={med*1e3:.0f}ms min={mn*1e3:.0f}ms "
                f"bw={traffic/med/2**30:.1f}GiB/s | "
                f"{len(chunks)}x25MiB median={med_b*1e3:.0f}ms min={mn_b*1e3:.0f}ms",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
