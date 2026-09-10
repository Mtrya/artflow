"""Collect the artifact findings of the captioning passes into one lookup.

The captioning passes asked a vision model, for every labelled row, whether the photograph
contains things that are not the artwork (colour chart, ruler, label, studio
desk, glare).  The findings live in two kinds of file: the main label pass
stores them under ``label.artifacts``, and the follow-up pass that re-scanned
only full/mounted views stores them at the top level.  This tool merges both
into one ``image_id -> artifacts`` JSONL so later stages can filter on them
without re-reading every label file.

CLI:
    python -m scripts.caption.build_artifact_index \
        --labels data/labels --out data/caption_enrich/artifact_index.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List


def collect(labels_root: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(labels_root, "*", "*.jsonl"))):
        name = os.path.basename(path)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = record.get("image_id")
            if not image_id:
                continue
            artifacts = record.get("artifacts")
            if artifacts is None:
                label = record.get("label")
                artifacts = label.get("artifacts") if isinstance(label, dict) else None
            if artifacts is None:
                continue
            if artifacts:
                # Keep the union: a row found dirty by either pass is dirty.
                index.setdefault(image_id, [])
                for item in artifacts:
                    if item not in index[image_id]:
                        index[image_id].append(item)
            else:
                index.setdefault(image_id, [])
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = collect(args.labels)
    dirty = {key: value for key, value in index.items() if value}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for image_id, artifacts in index.items():
            handle.write(json.dumps({"image_id": image_id, "artifacts": artifacts},
                                    ensure_ascii=False) + "\n")
    counts = Counter(item for values in dirty.values() for item in values)
    print(f"{len(index)} rows indexed, {len(dirty)} flagged ({len(dirty) / max(len(index), 1):.1%})")
    print("kinds:", dict(counts.most_common()))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
