#!/usr/bin/env python
"""Build the RL rollout prompt pool for Stage 6, mirroring the hero data mix.

Each source is a JSONL file of captioned rows; the per-source sample count is
proportional to (hero mix weight x source size), so the rollout distribution
matches what the model was trained on without re-running the bucket planner.
Prompts keep their domain tag for the per-domain canary/hacking monitors.

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
    rows = {name: list(iter_rows(path)) for name, path in sources.items()}
    sizes = {name: len(rs) for name, rs in rows.items()}
    counts = allocate(sizes, weights, total)
    pool = []
    for name, count in counts.items():
        chosen = rng.sample(rows[name], k=min(count, len(rows[name])))
        for row in chosen:
            available = [f for f in fields if row.get(f)]
            if not available:
                continue
            field = rng.choice(available)
            pool.append({"prompt": row[field], "source": name,
                         "field": field})
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
