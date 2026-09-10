"""Publish the current Chinese-painting metadata to HuggingFace.

The image shards stay as they are: the dataset ships the uncropped photograph
plus the box that isolates the artwork, so a consumer can crop or not.  What
changes is the per-image record, which is what this script rebuilds and uploads:

* one ``bbox`` per row — the box from the second cropping round where it exists,
  the first-round box otherwise;
* the transcription of in-image text folded into ``caption_zh`` / ``caption_en``;
* ``caption_long``, the new long caption added by the enrichment pass, where one
  exists.

Run on the machine that holds the metadata and the caption output:

    HF_TOKEN=... python -m scripts.data.publish_hf_d1 \
        --metadata $W/data/meta/d1/d1_metadata.jsonl \
        --captions data/caption_enrich/production/captions.jsonl \
        --out $W/hf_d1_staging/metadata.parquet --upload
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ID = "kaupane/chinese-painting-collection"
PARQUET_PATH = "metadata/metadata.parquet"
README_PATH = "README.md"
CARD = Path(__file__).resolve().parents[2] / "data" / "hf_d1" / "README.md"


def load_long_captions(path: str, accepted_only: bool) -> Dict[str, List[Dict]]:
    """Long captions per image, newest last.

    ``accepted_only`` keeps captions that failed the acceptance check out.  A
    caption that only missed its length window is still used, because the
    training bucket plan can accommodate any length.
    """
    from src.dataset.caption_prompts import length_only_reject

    by_image: Dict[str, List[Dict]] = defaultdict(list)
    if not path or not os.path.exists(path):
        return by_image
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("text"):
            continue
        if accepted_only and not (record.get("accepted")
                                  or length_only_reject(record.get("reject_reasons"))):
            continue
        by_image[record["image_id"]].append(record)
    return by_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--captions", default=None, help="enrichment output JSONL")
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--all-captions", action="store_true",
                        help="include captions that failed the acceptance check")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    long_captions = load_long_captions(args.captions, not args.all_captions)
    print(f"long captions for {len(long_captions)} images")

    rows = []
    with_long = 0
    for line in Path(args.metadata).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        record.pop("local_path", None)
        extra = long_captions.get(record["image_id"], [])
        record["caption_long"] = extra[-1]["text"] if extra else None
        if extra:
            with_long += 1
        rows.append(record)

    columns = list(rows[0])
    table = pa.Table.from_pylist([{k: r.get(k) for k in columns} for r in rows])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    print(f"{len(rows)} rows, {with_long} with a long caption -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_file(path_or_fileobj=str(out_path), path_in_repo=PARQUET_PATH,
                        repo_id=REPO_ID, repo_type="dataset")
        print(f"uploaded {PARQUET_PATH}")
        if CARD.is_file():
            api.upload_file(path_or_fileobj=str(CARD), path_in_repo=README_PATH,
                            repo_id=REPO_ID, repo_type="dataset")
            print(f"uploaded {README_PATH}")
        previews = CARD.parent / "previews"
        if previews.is_dir():
            api.upload_folder(folder_path=str(previews), path_in_repo="previews",
                              repo_id=REPO_ID, repo_type="dataset")
            print(f"uploaded previews ({len(list(previews.glob('*.jpg')))} files)")


if __name__ == "__main__":
    main()
