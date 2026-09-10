"""Finalise the Chinese-painting metadata: one crop box per row.

Takes the per-row record from the labelling pass plus the box found by the
second cropping round, and writes the metadata the dataset is built from:

* the crop box is the second-round box where one was produced, otherwise the
  first-round box, and there is exactly one ``bbox`` field;
* rolled-up scrolls and unusable views are dropped — a rolled scroll is a flat
  dark desk with a cylinder on it, which is not something to train on;
* the transcription of in-image text is folded into the captions, because the
  training manifest only reads the captions.

CLI (runs where the metadata lives):
    python -m scripts.data.apply_final_bbox \
        --metadata $W/data/meta/d1/d1_metadata.jsonl \
        --boxes bbox_v2.jsonl --out $W/data/meta/d1/d1_metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset.ocr_merge import append_ocr_block, clean_ocr_lines

DROP_VIEW_TYPES = {"rolled", "junk"}


def load_boxes(path: str) -> Dict[str, List[float]]:
    boxes: Dict[str, List[float]] = {}
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        box = record.get("bbox_v2")
        if box and len(box) == 4:
            boxes[record["image_id"]] = [float(v) for v in box]
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--boxes", required=True, help="second-round box JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--in-place", action="store_true",
                        help="write to --out even if it is the input path")
    args = parser.parse_args()

    boxes = load_boxes(args.boxes)
    print(f"{len(boxes)} second-round boxes")

    stats = {"rows": 0, "kept": 0, "dropped_view": 0, "box_replaced": 0,
             "ocr_merged": 0, "no_box": 0}
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with Path(args.metadata).open(encoding="utf-8") as source, \
            tmp_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            stats["rows"] += 1
            if record.get("view_type") in DROP_VIEW_TYPES:
                stats["dropped_view"] += 1
                continue

            box = boxes.get(record["image_id"])
            if box is not None:
                record["bbox"] = box
                stats["box_replaced"] += 1
            elif not record.get("bbox"):
                stats["no_box"] += 1

            ocr = clean_ocr_lines(record.get("ocr_text") or "")
            if ocr:
                for key, language in (("caption_zh", "zh"), ("caption_en", "en")):
                    caption = record.get(key)
                    if caption:
                        record[key] = append_ocr_block(caption, ocr, language)
                stats["ocr_merged"] += 1

            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    tmp_path.replace(out_path)
    print(json.dumps(stats, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
