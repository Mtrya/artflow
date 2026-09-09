"""Cut the training view of selected rows into small JPEGs for captioning.

The captioner runs on a machine that cannot read the shared disk, and the
caption must describe the crop the trainer actually sees.  This tool runs on
the shared disk, applies exactly the crop the precompute used (the normalised
``bbox`` box, when present), downscales to a bounded edge, and writes one JPEG
per selected row plus an index that records what was written.

Sharding: run several instances with ``--shard i --shards n`` on different
machines; each writes its own index file, and the driver merges them.

CLI:
    python -m scripts.caption.make_thumbnails \
        --selection <selection.jsonl> --out <thumb-dir> \
        --max-edge 1024 --quality 88 --shard 0 --shards 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 32 * 1024 * 1024


def crop_to_bbox(image: Image.Image, bbox: Optional[List[float]]) -> Tuple[Image.Image, Optional[List[int]]]:
    """Apply the precompute crop rule: normalised 0-1000 box, 32px minimum."""
    if not bbox:
        return image, None
    width, height = image.size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    left = max(0, min(width, x1 / 1000.0 * width))
    top = max(0, min(height, y1 / 1000.0 * height))
    right = max(0, min(width, x2 / 1000.0 * width))
    bottom = max(0, min(height, y2 / 1000.0 * height))
    if right - left < 32 or bottom - top < 32:
        raise ValueError("bbox too small")
    box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
    return image.crop(box), list(box)


def make_thumbnail(path: str, bbox: Optional[List[float]], out_path: Path,
                   max_edge: int, quality: int) -> Dict:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image_width, image_height = image.size
        image, crop_box = crop_to_bbox(image, bbox)
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=quality, optimize=True)
        thumb_size = image.size
    payload = out_path.read_bytes()
    return {
        "thumb_path": str(out_path),
        "thumb_bytes": len(payload),
        "thumb_sha256": hashlib.sha256(payload).hexdigest()[:32],
        "thumb_width": thumb_size[0],
        "thumb_height": thumb_size[1],
        # Geometry needed to map a box found on the thumbnail back to the
        # original photograph: the crop actually applied, and the size of the
        # image it was applied to.
        "crop_box": crop_box,
        "image_width": image_width,
        "image_height": image_height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="selection manifest JSONL")
    parser.add_argument("--out", required=True, help="directory for thumbnails")
    parser.add_argument("--index", default=None, help="index JSONL (default <out>/index.jsonl)")
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index) if args.index else out_dir / "index.jsonl"

    records = [json.loads(line) for line in Path(args.selection).open(encoding="utf-8") if line.strip()]
    shard_records = records[args.shard::args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(shard_records)} of {len(records)} rows", flush=True)

    written = 0
    skipped = 0
    failed = 0
    with index_path.open("w", encoding="utf-8") as index:
        for position, record in enumerate(shard_records):
            image_id = record["image_id"]
            target = out_dir / f"{image_id}.jpg"
            if target.is_file():
                skipped += 1
                continue
            try:
                info = make_thumbnail(record["local_path"], record.get("bbox"), target,
                                      args.max_edge, args.quality)
            except Exception as exc:  # unreadable or malformed row: record and continue
                failed += 1
                index.write(json.dumps({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"},
                                       ensure_ascii=False) + "\n")
                continue
            index.write(json.dumps({"image_id": image_id, **info}, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                index.flush()
                print(f"  {position + 1}/{len(shard_records)} written={written} "
                      f"skipped={skipped} failed={failed}", flush=True)
    print(f"done shard {args.shard}: written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
