"""Assemble the Chinese-painting training manifest from the published dataset.

The published set (``kaupane/chinese-painting-collection``) is the version of
record for this domain: it carries the final crop box per image and the
captions with the calligraphy transcription folded in, neither of which the
earlier manifest had.  Its parquet shards also embed the source photographs,
but those bytes are the files the harvester wrote to the shared disk - the
publisher read them with ``read_bytes`` and stored them unchanged - so this
builds ``local_path`` against the shared-disk copies rather than pulling a
second 200 GB copy of the images.

Captions are ordered shortest to longest.  A row can carry up to three: the
Chinese and English descriptions of the published row, plus the long caption
from the caption pass.  The trainer picks one caption per row as training
progresses, so the long one has to sit in the same list as the short ones.

Run on the machine that has the shared disk:

    python -m scripts.data.build_d1_manifest \\
        --metadata $W/data/meta/d1_release.parquet \\
        --long-captions $W/data/caption_enrich/d1/captions_final.jsonl \\
        --image-root $W/data/clean \\
        --out $W/data/meta/precompute/d1.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pyarrow.parquet as pq


def read_long_captions(path: str) -> Dict[str, str]:
    """Accepted long captions by image id; rejected rows are left out."""
    captions = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("accepted") and record.get("text"):
                captions[record["image_id"]] = record["text"]
    return captions


def image_path(image_root: str, shard: str, image_id: str) -> Path:
    return Path(image_root) / shard / "images" / f"{image_id}.jpg"


def crop_box(value) -> Optional[List[float]]:
    """The published crop box, as four numbers.

    The metadata table stores it as a JSON string rather than a list, so a row
    that carries a box and a row that does not are both strings as far as this
    build is concerned.  Passing the string through would make every cropped
    row fail the crop at precompute time and be dropped, so it is parsed here.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "null"}:
            return None
        value = json.loads(text)
    box = [float(component) for component in value]
    return box or None


def build(metadata_path: str, image_root: str, long_captions: Dict[str, str]) -> Iterator[Dict]:
    table = pq.read_table(metadata_path)
    for record in table.to_pylist():
        path = image_path(image_root, record["shard"], record["image_id"])
        captions = [text for text in (record.get("caption_zh"), record.get("caption_en"))
                    if text]
        long_caption = long_captions.get(record["image_id"])
        if long_caption:
            captions.append(long_caption)
        yield {
            "image_id": record["image_id"],
            "local_path": str(path),
            "captions": captions,
            "width": None,
            "height": None,
            "bbox": crop_box(record.get("bbox")),
            "source": f"d1_{record['source']}",
            "artist": record.get("artist"),
            "title": record.get("title"),
        }


def write_manifest(metadata_path: str, image_root: str, long_captions: Dict[str, str],
                   out_path: str) -> Counter:
    """Write the manifest and report what went in and what could not."""
    stats = Counter()
    written = 0
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in build(metadata_path, image_root, long_captions):
            if not row["captions"]:
                stats["no captions"] += 1
                continue
            if not Path(row["local_path"]).is_file():
                stats["image missing"] += 1
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            stats[f"{len(row['captions'])} captions"] += 1
            stats["with crop box" if row["bbox"] else "whole frame"] += 1
            stats[f"source:{row['source']}"] += 1
    stats["rows"] = written
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True,
                        help="the published metadata table (metadata/metadata.parquet)")
    parser.add_argument("--long-captions", required=True,
                        help="long-caption output JSONL, keyed by image id")
    parser.add_argument("--image-root", required=True,
                        help="directory holding <shard>/images/<image_id>.jpg")
    parser.add_argument("--out", required=True, help="training manifest JSONL")
    args = parser.parse_args()

    long_captions = read_long_captions(args.long_captions)
    stats = write_manifest(args.metadata, args.image_root, long_captions, args.out)
    print(f"{stats['rows']} rows written to {args.out}")
    for key, count in stats.most_common():
        if key != "rows":
            print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
