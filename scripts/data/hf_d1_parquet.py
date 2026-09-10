"""Publish the Chinese-painting set as parquet with embedded images.

HuggingFace renders images in the dataset viewer when they are stored in a
parquet column of type ``struct<bytes: binary, path: string>``, which is how the
``datasets`` library stores an ``Image`` feature.  WebDataset tar shards cannot
be previewed, so the dataset is republished in that form: one row per image,
carrying the photograph, the tombstone fields, the captions and the final crop
box.

Shards are written at a bounded size so a failed upload costs one shard.

CLI (on the machine that holds the images):
    python -m scripts.data.hf_d1_parquet \
        --metadata $W/data/meta/d1/d1_metadata.jsonl \
        --captions $W/data/caption_enrich/d1/captions.jsonl \
        --out-dir $W/hf_d1_parquet --shard-gb 1.0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ["image_id", "shard", "source", "object_no", "title", "artist",
           "category", "culture", "view_type", "caption_zh", "caption_en",
           "caption_long", "ocr_text", "artifacts", "bbox"]

SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("shard", pa.string()),
    ("source", pa.string()),
    ("object_no", pa.string()),
    ("title", pa.string()),
    ("artist", pa.string()),
    ("category", pa.string()),
    ("culture", pa.string()),
    ("view_type", pa.string()),
    ("caption_zh", pa.string()),
    ("caption_en", pa.string()),
    ("caption_long", pa.string()),
    ("ocr_text", pa.string()),
    ("artifacts", pa.list_(pa.string())),
    ("bbox", pa.list_(pa.float32())),
    ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
])

# The dataset viewer takes column types from the "huggingface" key of the parquet
# schema metadata.  Without it the image column is presented as a struct of bytes
# and the viewer cannot render thumbnails, which is the point of publishing
# parquet at all.
FEATURES = {name: {"dtype": "string", "_type": "Value"} for name in COLUMNS}
FEATURES["artifacts"] = {"feature": {"dtype": "string", "_type": "Value"},
                         "_type": "Sequence"}
FEATURES["bbox"] = {"feature": {"dtype": "float32", "_type": "Value"},
                    "_type": "Sequence"}
FEATURES["image"] = {"_type": "Image"}


def schema_metadata() -> Dict[bytes, bytes]:
    return {b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()}


def load_long_captions(path: Optional[str]) -> Dict[str, str]:
    """Long caption per image, newest last.

    A caption is used unless its text is defective (boilerplate opening,
    repeated sentences).  Length is not a reason to drop it: the training
    bucket plan can accommodate any length.
    """
    from src.dataset.caption_prompts import length_only_reject

    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("text"):
            continue
        if record.get("accepted") or length_only_reject(record.get("reject_reasons")):
            out[record["image_id"]] = record["text"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--captions", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-gb", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--row-group-rows", type=int, default=16,
                        help="rows per parquet row group; the dataset viewer "
                             "refuses to scan more than 300 MB for one page, and "
                             "a row carries a whole photograph")
    args = parser.parse_args()

    long_captions = load_long_captions(args.captions)
    print(f"long captions: {len(long_captions)}", flush=True)
    out_dir = Path(args.out_dir) / "default" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_bytes = int(args.shard_gb * 1024 ** 3)
    rows: List[Dict] = []
    current = 0
    shard = 0
    written = 0
    missing = 0

    def flush() -> None:
        nonlocal rows, current, shard, written
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        table = table.replace_schema_metadata(schema_metadata())
        path = out_dir / f"part-{shard:05d}.parquet"
        pq.write_table(table, path, compression="zstd",
                       row_group_size=args.row_group_rows,
                       write_page_index=True)
        print(f"  wrote {path.name}: {len(rows)} rows "
              f"({path.stat().st_size / 1e9:.2f} GB)", flush=True)
        written += len(rows)
        rows = []
        current = 0
        shard += 1

    with Path(args.metadata).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            local_path = record.pop("local_path", None)
            if not local_path or not os.path.exists(local_path):
                missing += 1
                continue
            payload = Path(local_path).read_bytes()
            row = {key: record.get(key) for key in COLUMNS if key != "image"}
            row["bbox"] = [float(v) for v in (record.get("bbox") or [])] or None
            row["artifacts"] = list(record.get("artifacts") or [])
            row["caption_long"] = long_captions.get(record["image_id"])
            row["image"] = {"bytes": payload, "path": os.path.basename(local_path)}
            rows.append(row)
            current += len(payload)
            if current >= shard_bytes:
                flush()
            if args.limit and written + len(rows) >= args.limit:
                break
    flush()
    print(f"done: {written} rows in {shard} shards, {missing} images missing")


if __name__ == "__main__":
    main()
