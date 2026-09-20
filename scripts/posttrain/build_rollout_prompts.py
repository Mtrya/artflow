#!/usr/bin/env python
"""Build the RL rollout prompt pool for Stage 6, mirroring the hero data mix.

Each source is a JSONL file of captioned rows; the per-source sample count is
proportional to (hero mix weight x source size), so the rollout distribution
matches what the model was trained on without re-running the bucket planner.
Prompts keep their domain tag for the per-domain canary/hacking monitors.

Accepted row formats:
  - caption_zh / caption_en / caption_long string fields (HF metadata style)
  - "captions": list[str] (precomputed training manifest style)
Rows without any usable caption are dropped BEFORE allocation, so the
requested counts and mix are preserved; a source with zero usable rows is a
hard error.

Example:
    python -m scripts.posttrain.build_rollout_prompts \
        --source d1=data/hf_d1/captions.jsonl --weight d1=2.0 \
        --source d4=data/hf_d4/captions.jsonl --weight d4=1.0 \
        --total 20000 --out data/rollout_prompts.jsonl
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_FIELDS = ("caption_zh", "caption_en", "caption_long")


def allocate(sizes: dict[str, int], weights: dict[str, float],
             total: int) -> dict[str, int]:
    """Per-source counts proportional to weight x size, largest remainder."""
    if total <= 0:
        raise ValueError("total must be positive")
    mass = {name: sizes[name] * weights.get(name, 1.0) for name in sizes}
    denom = sum(mass.values())
    if denom <= 0:
        raise ValueError("no source has positive weight x size")
    raw = {name: total * m / denom for name, m in mass.items()}
    counts = {name: int(v) for name, v in raw.items()}
    remainder = total - sum(counts.values())
    for name in sorted(raw, key=lambda n: raw[n] - counts[n], reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def fit_counts(counts: dict[str, int], sizes: dict[str, int],
               weights: dict[str, float], total: int) -> dict[str, int]:
    """Cap counts at availability and redistribute the shortfall.

    Allocation may request more rows than a small source holds; capping alone
    would silently shrink and skew the pool. The deficit is re-allocated to
    sources with spare capacity, again proportional to weight x spare size.
    """
    counts = {n: min(c, sizes[n]) for n, c in counts.items()}
    for _ in range(len(counts) + 2):
        deficit = total - sum(counts.values())
        if deficit <= 0:
            return counts
        spare = {n: sizes[n] - counts[n] for n in counts if sizes[n] > counts[n]}
        if not spare:
            raise ValueError(
                f"requested {total} prompts but the sources hold only "
                f"{total - deficit} usable rows in total")
        extra = allocate(spare, weights, deficit)
        for n, e in extra.items():
            counts[n] = min(counts[n] + e, sizes[n])
    raise AssertionError("fit_counts did not converge")


def row_captions(row: dict, fields) -> list[tuple[str, str]]:
    """(field, text) pairs usable as prompts; supports both row formats."""
    out = [(f, row[f]) for f in fields if row.get(f)]
    caps = row.get("captions")
    if isinstance(caps, str):
        caps = [caps]
    if caps:
        out.extend(("captions", c) for c in caps if c)
    return out


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_pool(sources: dict[str, Path], weights: dict[str, float],
               total: int, *, fields=DEFAULT_FIELDS,
               seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    eligible = {}
    for name, path in sources.items():
        rows = [(row, row_captions(row, fields)) for row in iter_rows(path)]
        rows = [(row, caps) for row, caps in rows if caps]
        if not rows:
            raise ValueError(
                f"source {name!r} at {path} has no usable captions "
                f"(looked for {tuple(fields)} and 'captions')")
        eligible[name] = rows
    sizes = {name: len(rs) for name, rs in eligible.items()}
    counts = fit_counts(allocate(sizes, weights, total), sizes, weights, total)
    pool = []
    for name, count in counts.items():
        for row, caps in rng.sample(eligible[name], k=count):
            field, text = rng.choice(caps)
            pool.append({"prompt": text, "source": name, "field": field})
    rng.shuffle(pool)
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        help="name=path/to/captions.jsonl")
    parser.add_argument("--weight", action="append", default=[],
                        help="name=mix weight (default 1.0)")
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sources = dict(s.split("=", 1) for s in args.source)
    weights = {k: float(v) for k, v in
               (w.split("=", 1) for w in args.weight)}
    pool = build_pool({k: Path(v) for k, v in sources.items()},
                      weights, args.total, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in pool:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(pool)} prompts to {args.out}")


if __name__ == "__main__":
    main()
