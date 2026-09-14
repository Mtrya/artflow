#!/usr/bin/env python3
"""Pull eval/loss (and a few training) curves for the scaling runs from
swanlab and dump them as compact JSON for local analysis.

A run may appear under several swanlab run ids (restarts continue the same
experiment name); all ids for a name are merged by taking, per step, the
record from the latest-created id that covers it.
"""
import json
import os
import sys

import swanlab

OUT = sys.argv[1] if len(sys.argv) > 1 else "ladder_curves.json"
KEYS = ["eval/loss", "train/loss", "train/grad_norm", "train/samples_per_sec"]

project = os.environ.get("SWANLAB_PROJECT")
if not project:
    raise SystemExit("Set SWANLAB_PROJECT to <entity>/<project>")
api = swanlab.Api()
runs = list(api.runs(project))
print(f"total runs: {len(runs)}", file=sys.stderr)

wanted = {}
for r in runs:
    name = r.name
    if not name.startswith(("s4-", "hero-val")):
        continue
    wanted.setdefault(name, []).append(r)

out = {}
for name, rs in sorted(wanted.items()):
    rs.sort(key=lambda r: r.created_at)
    curves = {k: {} for k in KEYS}
    state = rs[-1].state
    for r in rs:
        try:
            blocks = r.metrics(KEYS).get("list", [])
        except Exception as e:  # noqa: BLE001
            print(f"{name} {r.id}: metrics failed: {e}", file=sys.stderr)
            continue
        for block in blocks:
            key = block.get("key")
            if key not in curves:
                continue
            for entry in block.get("metrics", []):
                curves[key][int(entry["index"])] = float(entry["data"])
    out[name] = {
        "state": state,
        "curves": {k: sorted(v.items()) for k, v in curves.items() if v},
    }
    n = {k: len(v) for k, v in out[name]["curves"].items()}
    print(f"{name}: {n}", file=sys.stderr)

with open(OUT, "w") as f:
    json.dump(out, f)
print(f"wrote {OUT} ({len(out)} runs)", file=sys.stderr)
