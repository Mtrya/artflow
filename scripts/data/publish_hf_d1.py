"""Stage and optionally publish D1 metadata and image parquet on Hugging Face.

Every row carries final crop boxes and OCR-enriched captions from --metadata.
--captions supplies accepted long captions. --images embeds unchanged source
photographs in viewer-compatible parquet with bounded shards and row groups.
Metadata-only publication updates the caption/crop table independently.

    python -m scripts.data.publish_hf_d1 --metadata d1_metadata.jsonl \
        --captions captions_final.jsonl --out-dir staging/d1 --images
    # Add --upload --repo-id OWNER/DATASET to publish the staged release.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict

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


def load_long_captions(path: Path | None) -> dict[str, str]:
    """Keep the latest usable caption per image, including length-only rejects."""
    from src.dataset.caption_prompts import length_only_reject

    captions = {}
    if path is None:
        return captions
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("text") and (row.get("accepted") or
                                    length_only_reject(row.get("reject_reasons"))):
                captions[row["image_id"]] = row["text"]
    return captions


def crop_box(value) -> list[float] | None:
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


def write_release(metadata: Path, captions: Path | None, out: Path, *,
                  images: bool = False, shard_gb: float = 1.0,
                  row_group_rows: int = 16) -> dict[str, int]:
    """Write a fresh staging directory; missing requested images fail the build."""
    if not math.isfinite(shard_gb) or shard_gb <= 0 or row_group_rows <= 0:
        raise ValueError("shard-gb and row-group-rows must be positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"staging directory must be empty: {out}")
    long_captions = load_long_captions(captions)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = out / "default" / "train"
    if images:
        image_dir.mkdir(parents=True)
    metadata_rows, image_rows = [], []
    current_bytes = shard = with_long = 0
    seen = set()

    def flush():
        nonlocal current_bytes, shard
        if not image_rows:
            return
        table = pa.Table.from_pylist(image_rows, schema=SCHEMA)
        table = table.replace_schema_metadata(schema_metadata())
        path = image_dir / f"part-{shard:05d}.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=row_group_rows,
                       write_page_index=True)
        image_rows.clear()
        current_bytes = 0
        shard += 1

    with Path(metadata).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["image_id"]
            if image_id in seen:
                raise ValueError(f"duplicate image_id: {image_id}")
            seen.add(image_id)
            row = {key: record.get(key) for key in COLUMNS}
            row["bbox"] = crop_box(record.get("bbox"))
            row["artifacts"] = list(record.get("artifacts") or [])
            row["caption_long"] = long_captions.get(image_id, record.get("caption_long"))
            with_long += bool(row["caption_long"])
            metadata_rows.append(row)
            if images:
                local_path = record.get("local_path")
                if not local_path or not Path(local_path).is_file():
                    raise FileNotFoundError(f"image {image_id}: {local_path}")
                payload = Path(local_path).read_bytes()
                image_rows.append({**row, "image": {"bytes": payload,
                                                    "path": Path(local_path).name}})
                current_bytes += len(payload)
                if current_bytes >= shard_gb * 1024 ** 3:
                    flush()
    if not metadata_rows:
        raise ValueError("metadata contains no rows")
    flush()
    metadata_dir = out / "metadata"
    metadata_dir.mkdir()
    metadata_schema = pa.schema([SCHEMA.field(name) for name in COLUMNS])
    pq.write_table(pa.Table.from_pylist(metadata_rows, schema=metadata_schema),
                   metadata_dir / "metadata.parquet", compression="zstd")
    return {"rows": len(metadata_rows), "long_captions": with_long, "image_shards": shard}


def upload_release(out: Path, repo_id: str, *, images: bool):
    """Publish one release commit; replace the image shard set when supplied."""
    from huggingface_hub import HfApi

    return HfApi().upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(out),
        allow_patterns=["metadata/*.parquet", "default/train/*.parquet", "README.md"],
        delete_patterns=["default/train/*.parquet"] if images else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True, help="fresh staging directory")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--shard-gb", type=float, default=1.0)
    parser.add_argument("--row-group-rows", type=int, default=16)
    parser.add_argument("--card", type=Path, help="optional dataset README")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", help="existing Hugging Face dataset repository")
    args = parser.parse_args()
    if args.upload and not args.repo_id:
        parser.error("--upload requires --repo-id")
    if args.card and not args.card.is_file():
        parser.error(f"dataset card does not exist: {args.card}")
    stats = write_release(args.metadata, args.captions, args.out_dir,
                          images=args.images, shard_gb=args.shard_gb,
                          row_group_rows=args.row_group_rows)
    if args.card:
        shutil.copyfile(args.card, args.out_dir / "README.md")
    print(f"{stats} -> {args.out_dir}")
    if args.upload:
        upload_release(args.out_dir, args.repo_id, images=args.images)
        print(f"uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
