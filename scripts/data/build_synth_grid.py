"""Turn expanded variants into the prompt grid the generator walks.

One row per image to render: the prompt (which is also the caption when the
picture matches it), the frame to render at, and the lineage back to the seed
and the variant axis, so a later audit can group rows by what they came from.

Frames are computed from a target area - the picture budget per image - and
snapped to multiples of 32, which is what the Qwen-Image pipeline accepts: it
silently resizes any frame that is not, and then the row's recorded size no
longer describes the file on disk.  The aspect ratio of each row is drawn by
stable hash from the mixing shares below, the same construction the caption
batches use for their length bands, so aspect cannot correlate with the seed.

CLI:
    python -m scripts.data.build_synth_grid --variants data/hpsv2/variants.jsonl \\
        --out data/hpsv2/grid.jsonl --area 1048576
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

MODEL = "qwen-image-2.1"
GRID_SEED = 42

RATIOS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2),
          "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16)}
FRAME_SHARES = (("1:1", 0.24), ("4:3", 0.14), ("3:4", 0.14), ("3:2", 0.16),
                ("2:3", 0.16), ("16:9", 0.08), ("9:16", 0.08))


def stable_hash(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def frames_for(area: int) -> dict:
    """One frame per aspect ratio, each about ``area`` pixels, snapped to /32."""
    frames = {}
    for name, (rx, ry) in RATIOS.items():
        height = math.sqrt(area * ry / rx)
        width = area / height
        frames[name] = (int(round(width / 32) * 32), int(round(height / 32) * 32))
    return frames


def pick_aspect(key: str) -> str:
    draw = stable_hash(key, GRID_SEED + 5)
    cumulative = 0.0
    for name, share in FRAME_SHARES:
        cumulative += share
        if draw < cumulative:
            return name
    return FRAME_SHARES[-1][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="data/hpsv2/variants.jsonl")
    parser.add_argument("--out", default="data/hpsv2/grid.jsonl")
    parser.add_argument("--area", type=int, default=1048576,
                        help="target pixels per image; frames snap to multiples of 16")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    frames = frames_for(args.area)
    rows = []
    seen = set()
    for line in Path(args.variants).open(encoding="utf-8"):
        record = json.loads(line)
        for variant in record["variants"]:
            key = f"{record['seed_id']}-{variant['axis']}"
            if key in seen:
                raise SystemExit(f"duplicate variant key {key}")
            seen.add(key)
            aspect = pick_aspect(key)
            width, height = frames[aspect]
            rows.append({
                "prompt_id": f"syn21-{len(rows):06d}",
                "image_id": f"syn21-{len(rows):06d}",
                "text": variant["text"],
                "language": variant["language"],
                "family": "hpsv2",
                "subject": record["seed_id"],
                "subject_group": variant["axis"],
                "aspect": aspect,
                "width": width,
                "height": height,
                "model": args.model,
                "band": variant["band"],
                "format": variant["format"],
                "prompt_version": variant["prompt_version"],
                "seed_prompt": record["seed_prompt"],
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    aspects = Counter(row["aspect"] for row in rows)
    frames_used = Counter((row["width"], row["height"]) for row in rows)
    print(f"{len(rows)} rows -> {out}")
    print("  aspects: " + "  ".join(f"{k} {v}" for k, v in aspects.most_common()))
    print("  frames: " + "  ".join(f"{w}x{h} {n}" for (w, h), n in frames_used.most_common()))


if __name__ == "__main__":
    main()
