#!/usr/bin/env python3
"""Pull the CUDA-event step breakdown (train/ms_{text,fwd,bwd,opt,ema})
for a run, averaged over its trailing steps."""
import statistics
import os
import sys

import swanlab

NAME = sys.argv[1]
TAIL = int(sys.argv[2]) if len(sys.argv) > 2 else 30

project = os.environ.get("SWANLAB_PROJECT")
if not project:
    raise SystemExit("Set SWANLAB_PROJECT to <entity>/<project>")
api = swanlab.Api()
cands = sorted(
    (r for r in api.runs(project) if r.name == NAME),
    key=lambda x: x.created_at,
)
r = cands[-1]
keys = ["train/ms_text", "train/ms_fwd", "train/ms_bwd", "train/ms_opt", "train/ms_ema"]
blocks = r.metrics(keys).get("list", [])
print(f"run {NAME} (state {r.state})")
for b in blocks:
    pts = sorted((int(e["index"]), float(e["data"])) for e in b.get("metrics", []))
    if not pts:
        continue
    tail = [v for s, v in pts if s > pts[-1][0] - TAIL]
    print(f"{b['key']:16s} mean_last{len(tail)} = {statistics.mean(tail):9.1f} ms")
