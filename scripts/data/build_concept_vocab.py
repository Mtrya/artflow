"""Build the concept-benchmark vocabulary artifact.

Pulls the assembled sources (WordNet via NLTK, ImageNet-1k labels, the
pre-extracted Getty AAT preferred labels, curated zh lists), cleans and
merges them through src.dataset.concept_vocab, and writes one versioned
JSONL plus a human-readable report.

Usage:
    .venv/bin/python scripts/data/build_concept_vocab.py \
        --imagenet data/vocab/imagenet1k.json \
        --aat data/vocab/aat_en_pref.jsonl \
        --curated data/vocab/curated/zh_ink_techniques.jsonl \
                  data/vocab/curated/zh_east_asian_entities.jsonl \
        --out data/vocab/concepts_v1.jsonl \
        --report data/vocab/concepts_v1.report.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dataset.concept_vocab import (AXES, Concept, load_aat, load_curated,
                                       load_imagenet, load_wordnet, merge,
                                       report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet", required=True)
    ap.add_argument("--aat", required=True)
    ap.add_argument("--curated", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--min-zipf", type=float, default=2.5)
    args = ap.parse_args()

    concepts: list[Concept] = []
    concepts += load_wordnet(args.min_zipf)
    concepts += load_imagenet(args.imagenet)
    concepts += load_aat(args.aat, args.min_zipf)
    concepts += load_curated(args.curated)
    merged = merge(concepts)

    random.seed(0)
    with open(args.out, "w") as fh:
        for c in merged:
            fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    samples = []
    for axis in AXES:
        pool = [c for c in merged if c.axis == axis]
        take = random.sample(pool, min(8, len(pool)))
        samples.append(f"### {axis} sample\n" + "\n".join(
            f"- {c.en}" + (f" ({c.zh})" if c.zh else "")
            + f"  [{c.sub}, {','.join(c.sources)}, zipf={c.freq_en:.2f}]"
            for c in take))

    Path(args.report).write_text(
        f"# Concept vocabulary {Path(args.out).name}\n\n"
        f"min_zipf={args.min_zipf}\n\n```\n{report(merged)}\n```\n\n"
        + "\n\n".join(samples) + "\n")
    print(report(merged))
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
