"""Opt-in per-rank measurements and bounded profiler traces for infra A/B runs."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


class InfraRecorder:
    def __init__(self, run_dir, rank, *, trace_start=-1, trace_steps=3, record_identity=False):
        if trace_steps < 1:
            raise ValueError("trace_steps must be positive")
        self.directory = Path(run_dir) / "infra"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.file = (self.directory / f"rank-{rank}.jsonl").open("a", buffering=1)
        self.trace_start = trace_start
        self.trace_steps = trace_steps
        self.profiler = None
        self.shapes = Counter()
        self.record_identity = record_identity
        self.identity = None

    def begin_update(self, step):
        self.shapes.clear()
        if self.record_identity:
            self.identity = hashlib.sha256()
        if step == self.trace_start:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=True, with_stack=False)
            self.profiler.start()

    def micro(self, latent_shape, bucket_hi, *, sample_identity=None):
        self.shapes[(*map(int, latent_shape), int(bucket_hi))] += 1
        if self.record_identity:
            if sample_identity is None:
                raise ValueError("recorded replay identity requires row/caption identifiers")
            self.identity.update(json.dumps(sample_identity, sort_keys=True,
                                            separators=(",", ":")).encode() + b"\n")

    def end_update(self, *, step, seconds, global_samples, loss, progress,
                   peak_allocated, peak_reserved, breakdown=None):
        record = dict(step=step, rank=self.rank, seconds=seconds,
                      global_samples=global_samples, loss=loss, progress=progress,
                      peak_allocated_bytes=peak_allocated,
                      peak_reserved_bytes=peak_reserved,
                      shapes=[dict(shape=list(shape), count=count)
                              for shape, count in self.shapes.items()],
                      breakdown_ms=breakdown, profiled=self.profiler is not None)
        if self.record_identity:
            record["sample_identity_sha256"] = self.identity.hexdigest()
        self.file.write(json.dumps(record) + "\n")
        if self.profiler is not None and step >= self.trace_start + self.trace_steps:
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.directory / f"rank-{self.rank}.trace.json"))
            (self.directory / f"rank-{self.rank}.operators.txt").write_text(
                self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
            self.profiler = None

    def close(self):
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        self.file.close()

    def checkpoint(self, *, step, seconds):
        """Full save through the completed-record barrier, separate from updates."""
        with (self.directory / f"checkpoint-rank-{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(step=step, rank=self.rank, seconds=seconds)) + "\n")
