"""Check that every generated caption belongs to the image it claims.

A caption is only useful if it describes the picture stored next to it.  The
generation client hashes the exact bytes it sends, and the record keeps that
hash, so each caption can be traced back to one file: this pass recomputes the
hash of each row's thumbnail and compares it with the record.

It also reports the two ways the link can break quietly: a caption whose image
is absent from the thumbnail index, and one image carrying two different
captions.

CLI:
    python -m scripts.caption.verify_provenance \\
        --captions captions.jsonl --index thumbs/index.jsonl --thumb-dir thumbs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def load_index(path: str, thumb_dir: str | None) -> dict:
    index = {}
    for line in Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        thumb_path = record.get("thumb_path")
        if thumb_dir and thumb_path:
            thumb_path = os.path.join(thumb_dir, os.path.basename(thumb_path))
        index[record["image_id"]] = thumb_path
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--thumb-dir", default=None,
                        help="directory holding the thumbnails, when the index "
                             "records another machine's paths")
    parser.add_argument("--sample", type=int, default=0,
                        help="check only this many rows (default: all of them)")
    args = parser.parse_args()

    index = load_index(args.index, args.thumb_dir)
    records = [json.loads(line) for line in Path(args.captions).open(encoding="utf-8")
               if line.strip()]
    checked_source = [r for r in records if r.get("text")]
    if args.sample and args.sample < len(checked_source):
        checked_source = random.Random(0).sample(checked_source, args.sample)

    missing_image = 0
    missing_file = 0
    mismatched = 0
    no_fingerprint = 0
    for record in checked_source:
        image_id = record["image_id"]
        if image_id not in index:
            missing_image += 1
            continue
        path = index[image_id]
        if not path or not os.path.exists(path):
            missing_file += 1
            continue
        expected = record.get("image_fingerprint")
        if not expected:
            no_fingerprint += 1
            continue
        with open(path, "rb") as handle:
            actual = hashlib.sha256(handle.read()).hexdigest()[:32]
        if actual != expected:
            mismatched += 1

    by_image = defaultdict(set)
    for record in records:
        if record.get("text"):
            by_image[record["image_id"]].add(record["text"])
    conflicting = sum(1 for texts in by_image.values() if len(texts) > 1)

    total = len(checked_source)
    print(f"checked {total} captions against {len(index)} indexed images")
    print(f"  image absent from index   {missing_image}")
    print(f"  thumbnail file missing    {missing_file}")
    print(f"  record without fingerprint {no_fingerprint}")
    print(f"  fingerprint mismatch      {mismatched}")
    print(f"  images with 2+ captions   {conflicting}")

    if mismatched or missing_image or missing_file:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
