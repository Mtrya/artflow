"""Merge the generated synthetic set into a precompute manifest.

Generation runs sharded, so each shard leaves its own record file; the trainer
needs one manifest.  The caption of a synthetic row is the prompt it was
generated from, and the grid guarantees the two cannot drift apart, so this
step copies the text rather than describing the picture again.

It also checks every claimed image: a row whose file is missing, or whose pixels
are not the size the grid asked for, is reported instead of being written to a
manifest that would fail later.

CLI:
    python -m scripts.data.synth_manifest \\
        --records data/synth/run-a data/synth/run-b \\
        --out data/meta/precompute/d3_synth.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image


def read_records(directories) -> List[Dict]:
    """Records from one directory or several; several when two generators split
    the grid between them."""
    if isinstance(directories, (str, Path)):
        directories = [directories]
    rows = []
    for directory in directories:
        for path in sorted(Path(directory).glob("generated_shard*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def build(rows: List[Dict]) -> tuple:
    stats = Counter()
    manifest = []
    seen = set()
    for row in rows:
        if row["prompt_id"] in seen:
            stats["duplicate record"] += 1
            continue
        seen.add(row["prompt_id"])
        path = Path(row["path"])
        if not path.is_file():
            stats["image missing"] += 1
            continue
        with Image.open(path) as image:
            width, height = image.size
        if (width, height) != (row["width"], row["height"]):
            stats["size mismatch"] += 1
            continue
        manifest.append({
            "image_id": row["image_id"],
            "local_path": str(path),
            "captions": [row["text"]],
            "width": width,
            "height": height,
            "bbox": None,
            "source": "d3_synth",
            "title": None,
            "artist": None,
        })
        stats[f"aspect {row['aspect']}"] += 1
        stats[f"family {row['family']}"] += 1
        stats[f"language {row['language']}"] += 1
    manifest.sort(key=lambda entry: entry["image_id"])
    return manifest, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", required=True, nargs="+",
                        help="one or more directories holding the generators' "
                             "generated_shard*.jsonl; pass every directory when "
                             "several generators split the grid")
    parser.add_argument("--out", required=True, help="precompute manifest JSONL")
    args = parser.parse_args()

    rows = read_records(args.records)
    manifest, stats = build(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for entry in manifest:
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{len(rows)} records -> {len(manifest)} manifest rows in {out}")
    for key, count in stats.most_common():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
