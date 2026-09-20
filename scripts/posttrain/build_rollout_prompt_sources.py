#!/usr/bin/env python
"""Stage the per-source caption pools that feed the Stage 6 rollout prompt pool.

`build_rollout_prompts.py` gives each source a share proportional to
(hero mix weight x rows in the source file), so the file it is handed has to
hold the rows the hero run trained on.  Those rows are the per-source precompute
manifests under `<root>/data/meta/precompute/`; each manifest sits marginally
above the resolvable pool reported for the 256p stage (rows the resolution
filter dropped), so every source is subsampled to that reported count first.
With sizes pinned to the training pools, the 256p mix multipliers reproduce the
hero mix exactly.

Manifest rows carry `captions` lists, which the builder already reads; only the
image id and those captions are kept, keeping the staged files small enough to
move off the cluster.  Stdlib only, so it runs on the CPU notebook that holds
the manifests as well as locally.

CLI:
    python -m scripts.posttrain.build_rollout_prompt_sources \\
        --manifest-dir "$ARTFLOW_ROOT/data/meta/precompute" \\
        --out-dir data/posttrain/rollout_prompts/sources
"""

import argparse
import json
import random
from pathlib import Path

# 256p training-pool row counts, from
# bucket_plans/hero/batch-targets-0914/hero-256p-k20.report.md (the resolvable
# pool per source, i.e. what the hero run actually trained on).
POOL_ROWS = {
    "d1": 91416,
    "d2-wikiart": 214041,
    "d2-museum": 10065,
    "d3-human": 118393,
    "d3-people": 114345,
    "d3-pexels": 37417,
    "d3-synth-v2": 10893,
    "d4-inat": 2823,
    "d4-megalith": 5504,
    "d4-pd12m": 162307,
    "d4-vintage": 194625,
    "d4-zimage": 45228,
    "d4-relaion": 621773,
}

# Logical hero dataset name -> precompute manifest stem.
MANIFESTS = {
    "d1": "d1",
    "d2-wikiart": "d2_wikiart",
    "d2-museum": "d2_museum",
    "d3-human": "d3_human",
    "d3-people": "d3_people",
    "d3-pexels": "d3_pexels",
    "d3-synth-v2": "d3_synth_v2",
    "d4-inat": "d4_inat",
    "d4-megalith": "d4_megalith",
    "d4-pd12m": "d4_pd12m",
    "d4-vintage": "d4_vintage",
    "d4-zimage": "d4_zimage",
    "d4-relaion": "d4_relaion",
}


def read_rows(path: Path):
    """(image_id, captions) for every manifest row that has a caption."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            caps = [c for c in row.get("captions") or [] if c]
            if not caps:
                continue
            yield {"image_id": row.get("image_id"), "captions": caps}


def subsample(rows, limit: int, rng: random.Random) -> tuple:
    """Reservoir sample of `rows`, capped at `limit`; returns (kept, seen)."""
    kept = []
    seen = 0
    for i, row in enumerate(rows):
        seen = i + 1
        if i < limit:
            kept.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < limit:
                kept[j] = row
    return kept, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None,
                        help="logical dataset names; default all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(MANIFESTS)
    for name in names:
        path = args.manifest_dir / f"{MANIFESTS[name]}.jsonl"
        limit = POOL_ROWS[name]
        capped, available = subsample(read_rows(path), limit, rng)
        out = args.out_dir / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in capped:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if len(capped) < limit:
            note = f" usable rows < pool {limit}: source is short"
        elif available > limit:
            note = f" subsampled from {available}"
        else:
            note = ""
        print(f"{name:14s} {len(capped):8d} rows{note} -> {out}")


if __name__ == "__main__":
    main()
