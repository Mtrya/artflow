#!/usr/bin/env python3
"""Analytic micro-batch sizing and model-guided trim for bucket plans.

Where this sits in planning a run
---------------------------------
A bucket plan is produced in four steps, and this tool is steps 2 and 4:

1. Boundaries: ``scripts/plan_buckets.py`` solves the caption-length bucket
   boundaries from the corpus' retained-length distribution.
2. Analytic sizes (``table``): the largest micro-batch the card holds per
   bucket, from the measured memory model - no GPU needed.
3. Mixed-stream validation: a short real run on the whole mixture.  The
   per-shape model is exact for an isolated bucket, but a training stream
   serves ~50 compiled shapes and each retains some allocator state, so the
   peak of a mixed run is an empirical property of the plan as a whole.  A
   plan is only valid once this run survives it.
4. Trim (``trim``): if validation dies on a bucket, the OOM message and the
   trainer's shape log say exactly how much memory was live and which shape
   was being served, and the linear model turns that into the largest batch
   that fits - one computation, not a retreat to the next power of two.

The memory model
----------------
Fitted on 163 isolated screen runs (256p, 24-layer 1152-wide transformer,
Qwen3 text encoder): peak allocated memory is

    allocated_GB = const + batch x seq_len x slope

with const = 9.5 GB (weights, optimizer state, compiled kernels), slope =
0.00101 GB per sample per token, and seq_len = image tokens + bucket text
ceiling.  Per-bucket fit residuals are under 0.1 GB.  Reserved memory sits
~3-5 GB above allocated.  All three constants are parameters: a different
model shape or resolution recalibrates them with a sparse probe (a handful of
measured points), not a full screen.

Usage:
  python -m scripts.bench.size_micro_batches table \
      --plan plan.json --image-tokens '{"1": 256}' [--surcharge-gb 10]
  python -m scripts.bench.size_micro_batches trim \
      --plan plan.json --out plan.trimmed.json --log run.log
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Calibrated at 256p on the 24-layer / 1152-wide model; see the module
# docstring.  Override on the command line when a probe says otherwise.
DEFAULT_CONST_GB = 9.5
DEFAULT_SLOPE_GB_PER_TOKEN = 0.00101
DEFAULT_RESERVED_OFFSET_GB = 4.7
# Mixed-stream surcharge: memory retained beyond the largest single-shape
# prediction once ~50 compiled shapes are in play.  Calibrated from the 256p
# full-mix validation (39.1 GB measured against a 29.7 GB single-shape
# prediction); known to grow when the plan contains larger shapes, which is
# why validation (step 3 above) is never skipped.
DEFAULT_SURCHARGE_GB = 10.0
# Default ceiling: 95% of a 47.37 GB card, on reserved memory - the number
# that actually OOMs.
DEFAULT_CARD_GB = 47.37
DEFAULT_HEADROOM = 0.95


def read_plan(path: str) -> Dict[str, List[Dict[str, int]]]:
    with open(path) as handle:
        raw = json.load(handle)
    return {
        str(int(resolution_id)): [
            {"max_length": int(bucket["max_length"]),
             "batch_size": int(bucket["batch_size"])}
            for bucket in buckets
        ]
        for resolution_id, buckets in raw.items()
    }


def write_plan(path: str, plan: Mapping[str, Sequence[Mapping[str, int]]]) -> None:
    with open(path, "w") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _image_tokens(spec: str) -> Dict[str, int]:
    """One integer for every resolution, or an inline JSON object per id."""
    try:
        value = int(spec)
    except ValueError:
        raw = json.loads(spec)
        if not isinstance(raw, dict):
            raise ValueError("--image-tokens must be an integer or a JSON object")
        return {str(int(k)): int(v) for k, v in raw.items()}
    if value < 1:
        raise ValueError(f"--image-tokens must be positive, got {value}")
    return {"*": value}


def largest_batch(seq_len: int, *, const_gb: float, slope: float,
                  reserved_offset_gb: float, surcharge_gb: float,
                  ceiling_gb: float) -> int:
    """Largest batch whose predicted reserved peak stays under the ceiling."""
    available = ceiling_gb - const_gb - surcharge_gb - reserved_offset_gb
    if available <= 0:
        raise ValueError(
            f"memory ceiling {ceiling_gb:.2f} GB leaves nothing after the fixed "
            f"terms (const {const_gb} + surcharge {surcharge_gb} + reserved "
            f"offset {reserved_offset_gb})"
        )
    return max(1, math.floor(available / (slope * seq_len)))


def cmd_table(args: argparse.Namespace) -> int:
    plan = read_plan(args.plan)
    tokens = _image_tokens(args.image_tokens)
    ceiling = args.card_gb * args.headroom
    out: Dict[str, List[Dict[str, int]]] = {}
    for resolution_id, buckets in plan.items():
        image = tokens.get(resolution_id, tokens.get("*"))
        if image is None:
            raise ValueError(f"--image-tokens has no entry for resolution {resolution_id}")
        out[resolution_id] = []
        for bucket in buckets:
            seq = image + bucket["max_length"]
            size = largest_batch(
                seq, const_gb=args.const_gb, slope=args.slope,
                reserved_offset_gb=args.reserved_offset_gb,
                surcharge_gb=args.surcharge_gb, ceiling_gb=ceiling)
            out[resolution_id].append(
                {"max_length": bucket["max_length"], "batch_size": size})
            print(f"res{resolution_id} len<={bucket['max_length']:>5} "
                  f"seq={seq:>5} -> batch {size}")
    write_plan(args.out, out)
    print(f"\nwrote {args.out} (analytic sizes; validate with a mixed-stream "
          f"run before training on it)")
    return 0


_OOM_LIVE = re.compile(r"Of the allocated memory ([0-9.]+) GiB is allocated by PyTorch")
_OOM_TRIED = re.compile(r"Tried to allocate ([0-9.]+) GiB")
_OOM_TRIED_MB = re.compile(r"Tried to allocate ([0-9.]+) MiB")
_SHAPE = re.compile(
    r"\[shape\] step=\d+ micro=\d+ res=(\d+) txt_hi=(\d+) txt_len=\d+ "
    r"B=(\d+) latent=\d+x\d+ mem_gb=([0-9.]+)")


def parse_oom_log(path: str) -> Tuple[float, float, Tuple[int, int, int]]:
    """Read a failed run's log: (live GB at OOM, failed allocation GB, shape).

    The shape is the last ``[shape]`` line before the OOM - the micro-batch
    whose work was in flight - as (resolution id, bucket text ceiling, batch).
    The trainer only emits those lines with ARTFLOW_LOG_SHAPES set; without
    them the offending bucket cannot be attributed, so their absence is an
    error, not a guess.
    """
    live = tried = None
    shape: Optional[Tuple[int, int, int]] = None
    with open(path, errors="replace") as handle:
        for line in handle:
            match = _SHAPE.search(line)
            if match:
                shape = (int(match.group(1)), int(match.group(2)),
                         int(match.group(3)))
            if "OutOfMemoryError" in line or live is not None:
                m = _OOM_TRIED.search(line)
                if m:
                    tried = float(m.group(1))
                m = _OOM_TRIED_MB.search(line)
                if m:
                    tried = float(m.group(1)) / 1024.0
                m = _OOM_LIVE.search(line)
                if m:
                    live = float(m.group(1))
    if live is None or tried is None:
        raise ValueError(f"{path}: no CUDA OOM report found in this log")
    if shape is None:
        raise ValueError(
            f"{path}: no [shape] lines before the OOM; rerun the validation "
            f"with ARTFLOW_LOG_SHAPES=1 so the offending bucket is attributed")
    return live, tried, shape


def cmd_trim(args: argparse.Namespace) -> int:
    plan = read_plan(args.plan)
    tokens = _image_tokens(args.image_tokens)
    live_gb, failed_gb, (resolution_id, text_hi, batch) = parse_oom_log(args.log)
    resolution = str(resolution_id)

    buckets = plan.get(resolution)
    if buckets is None:
        raise ValueError(f"plan has no resolution {resolution}")
    index = next((i for i, b in enumerate(buckets) if b["max_length"] == text_hi),
                 None)
    if index is None:
        raise ValueError(
            f"plan resolution {resolution} has no bucket with bound {text_hi}; "
            f"the log's last shape does not belong to this plan")
    if buckets[index]["batch_size"] != batch:
        raise ValueError(
            f"the log's last shape ran at batch {batch} but the plan says "
            f"{buckets[index]['batch_size']} for res{resolution} len<={text_hi}; "
            f"trim the plan the run actually used")

    # The linear model: the failed forward's peak would have been
    # live + failed (+ unknown backward headroom, covered by the margin), and
    # activation memory scales with batch, so the batch that lands the peak at
    # the ceiling follows by proportion.  The margin absorbs what the OOM
    # report cannot see (allocations after the one that failed).
    ceiling = args.card_gb * args.headroom
    image = tokens.get(resolution, tokens.get("*"))
    if image is None:
        raise ValueError(f"--image-tokens has no entry for resolution {resolution}")
    seq = image + text_hi
    peak_est = live_gb + failed_gb
    fitted = math.floor(batch * (ceiling - args.margin_gb) / peak_est)
    fitted = max(1, min(fitted, batch - 1))

    print(f"OOM at res{resolution} len<={text_hi} batch {batch}: "
          f"{live_gb:.2f} GB live, allocation of {failed_gb:.2f} GB failed")
    print(f"peak estimate {peak_est:.2f} GB, ceiling {ceiling:.2f} GB with "
          f"{args.margin_gb:.1f} GB margin -> batch {fitted}")
    buckets[index]["batch_size"] = fitted
    write_plan(args.out, plan)
    print(f"wrote {args.out}; re-run the mixed-stream validation on it")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--plan", required=True, help="bucket plan JSON")
        p.add_argument("--image-tokens", required=True, metavar="N|JSON",
                       help="image tokens per resolution id: one integer for all, "
                            "or an inline JSON object")
        p.add_argument("--card-gb", type=float, default=DEFAULT_CARD_GB)
        p.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM,
                       help="fraction of the card the plan may use (default "
                            f"{DEFAULT_HEADROOM})")
        p.add_argument("--const-gb", type=float, default=DEFAULT_CONST_GB)
        p.add_argument("--slope", type=float, default=DEFAULT_SLOPE_GB_PER_TOKEN,
                       help="GB per sample per sequence token")
        p.add_argument("--reserved-offset-gb", type=float,
                       default=DEFAULT_RESERVED_OFFSET_GB)

    table = sub.add_parser("table", help="fill every bucket's batch size analytically")
    common(table)
    table.add_argument("--surcharge-gb", type=float, default=DEFAULT_SURCHARGE_GB,
                       help="mixed-stream surcharge reserve (default "
                            f"{DEFAULT_SURCHARGE_GB})")
    table.add_argument("--out", required=True)
    table.set_defaults(func=cmd_table)

    trim = sub.add_parser("trim", help="shrink the bucket an OOM log names")
    common(trim)
    trim.add_argument("--log", required=True,
                      help="failed validation run's log (OOM report + [shape] lines)")
    trim.add_argument("--margin-gb", type=float, default=2.0,
                      help="extra safety margin under the ceiling (default 2.0)")
    trim.add_argument("--out", required=True)
    trim.set_defaults(func=cmd_trim)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
