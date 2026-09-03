"""Dedup for harvested museum records.

Two levels, both metadata/exact-hash based — perceptual dHash was evaluated on the
NPM-TW 1K probe (2026-08-26) and abandoned: dark mounted-scroll paintings share a
coarse gradient signature, so even Hamming ≤1 gives 3/3 false positives.

1. exact: md5 of the image file — catches re-downloaded bytes.
2. cross-canvas: same normalized (title, artist) under different canvas ids —
   the same artwork digitized twice. Reported, not auto-dropped (views may differ
   in quality; the assembly step picks the largest view).

CLI:
    python -m src.dataset.dedup --in data/clean/npm_tw/metadata.jsonl
"""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, List


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(local_path: str, meta_path: str) -> str:
    """local_path may be written for a different workroot layout (e.g. with a
    leading ../); fall back to resolving next to the metadata file, then with
    leading ../ stripped relative to cwd."""
    candidates = [
        local_path,
        os.path.normpath(os.path.join(os.path.dirname(meta_path), os.path.basename(local_path))),
        os.path.join(os.path.dirname(meta_path), "images", os.path.basename(local_path)),
        local_path.lstrip("./").removeprefix("../") if local_path.startswith("../") else local_path,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return local_path


def norm_text(s: str) -> str:
    return re.sub(r"[\s　]+", "", s or "")


def find_exact_dupes(recs: List[Dict], meta_path: str = "") -> Dict[str, List[str]]:
    """md5 -> image_ids with more than one entry."""
    by_md5: Dict[str, List[str]] = defaultdict(list)
    for r in recs:
        by_md5[md5(resolve_path(r["local_path"], meta_path))].append(r["image_id"])
    return {k: v for k, v in by_md5.items() if len(v) > 1}


def canvas_id(rec: Dict) -> str:
    """Canvas label without the trailing view suffix (PAA/PAB/...)."""
    label = rec.get("canvas_label", "") or rec["image_id"]
    return re.sub(r"PA[A-Z]$", "", label)


def find_cross_canvas_dupes(recs: List[Dict]) -> Dict[tuple, List[str]]:
    """(title, artist) -> distinct canvas ids, for groups spanning 2+ canvases."""
    by_ta: Dict[tuple, set] = defaultdict(set)
    for r in recs:
        by_ta[(norm_text(r.get("title", "")), norm_text(r.get("artist", "")))].add(canvas_id(r))
    return {k: sorted(v) for k, v in by_ta.items() if len(v) > 1 and k != ("", "")}


def run(meta_path: str) -> None:
    with open(meta_path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    recs = [r for r in recs if not r.get("rejected")]
    exact = find_exact_dupes(recs, meta_path)
    cross = find_cross_canvas_dupes(recs)
    print(f"{meta_path}: {len(recs)} records")
    print(f"  exact md5 dupes: {len(exact)}")
    for k, v in list(exact.items())[:5]:
        print(f"    {k[:8]}: {v}")
    print(f"  cross-canvas (title, artist) groups: {len(cross)}")
    for (title, artist), canvases in list(cross.items())[:10]:
        print(f"    {title} / {artist}: {len(canvases)} canvases")


def main():
    ap = argparse.ArgumentParser(description="Dedup over cleaned metadata.jsonl")
    ap.add_argument("--in", dest="meta", required=True)
    args = ap.parse_args()
    run(args.meta)


if __name__ == "__main__":
    main()
