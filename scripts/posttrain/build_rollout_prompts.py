#!/usr/bin/env python
"""Build tagged rollout prompts from a run config's selected stage.

Source weights are sampling probabilities, matching training. Sample rows
without replacement and choose a usable caption uniformly within each row.
Sources that run out contribute all their usable rows; redistribute their
shortfall among sources with capacity, in proportion to the remaining weights.
The printed counts expose this finite-pool adjustment.

    python -m scripts.posttrain.build_rollout_prompts \
        --config configs/hero.toml --stage 896p \
        --total 20000 --out data/rollout_prompts.jsonl
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

DEFAULT_FIELDS = ("caption_zh", "caption_en", "caption_long")


def allocate(sizes: dict[str, int], weights: dict[str, float],
             total: int) -> dict[str, int]:
    """Per-source counts proportional to source weights, largest remainder."""
    if total <= 0:
        raise ValueError("total must be positive")
    mass = {name: weights.get(name, 1.0) for name in sizes}
    if not mass or any(not math.isfinite(w) or w <= 0 for w in mass.values()):
        raise ValueError("source weights must be finite and positive")
    denom = sum(mass.values())
    if not math.isfinite(denom):
        raise ValueError("total source weight must be finite")
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
    sources with spare capacity, in proportion to their source weights.
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
    out = [(f, row[f]) for f in fields
           if isinstance(row.get(f), str) and row[f].strip()]
    caps = row.get("captions")
    if isinstance(caps, str):
        caps = [caps]
    if caps:
        out.extend(("captions", c) for c in caps if isinstance(c, str) and c.strip())
    return out


def read_rows(path: Path):
    """Load caption columns only; Arrow image/latent columns stay on disk."""
    if path.is_dir():
        from datasets import load_from_disk

        return load_from_disk(str(path)).select_columns(["captions"])
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stage_sources(config_path: str, stage_name: str) -> tuple[dict, dict]:
    from src.pretrain.config import load_config

    stage = load_config(config_path).stage(stage_name)
    sources, weights = {}, {}
    for entry in stage.datasets:
        path = Path(entry.path)
        if path.name in sources:
            raise ValueError(f"duplicate source name: {path.name}")
        sources[path.name] = path
        weights[path.name] = entry.weight
    return sources, weights


def build_pool(sources: dict[str, Path], weights: dict[str, float],
               total: int, *, fields=DEFAULT_FIELDS,
               seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    unknown = weights.keys() - sources.keys()
    if unknown:
        raise ValueError(f"weights for unknown sources: {sorted(unknown)}")
    eligible = {}
    datasets = {}
    for name, path in sources.items():
        datasets[name] = read_rows(path)
        rows = [i for i, row in enumerate(datasets[name]) if row_captions(row, fields)]
        if not rows:
            raise ValueError(
                f"source {name!r} at {path} has no usable captions "
                f"(looked for {tuple(fields)} and 'captions')")
        eligible[name] = rows
    sizes = {name: len(rs) for name, rs in eligible.items()}
    counts = fit_counts(allocate(sizes, weights, total), sizes, weights, total)
    pool = []
    for name, count in counts.items():
        for index in rng.sample(eligible[name], k=count):
            caps = row_captions(datasets[name][index], fields)
            field, text = rng.choice(caps)
            pool.append({"prompt": text, "source": name, "field": field})
    rng.shuffle(pool)
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="complete run TOML")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sources, weights = stage_sources(args.config, args.stage)
    pool = build_pool(sources, weights, args.total, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in pool:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(pool)} prompts to {args.out}")
    print("source counts:", dict(sorted(Counter(r["source"] for r in pool).items())))


if __name__ == "__main__":
    main()
