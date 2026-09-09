"""Build review bundles so a caption can be checked against its image.

The reviewer is the execution agent: it must look at the image, not just the
text.  Reading several hundred images one at a time is wasteful, so this tool
packs a batch of images into a labelled contact sheet and writes the matching
captions next to it.  Anything suspicious found on the sheet is escalated by
reading that image on its own at full size.

Two more things happen here:

* **Controls.**  ``--inject-controls K`` replaces K captions with deliberately
  wrong ones (a changed count, attribute or spatial relation).  A reviewer that
  cannot flag these is not a usable acceptance mechanism, so the control ids are
  recorded separately and the flag rate on them is reported.
* **Blinding.**  Model identity is stripped from the sheet captions and stored
  only in the manifest, so the reviewer cannot prefer one model's style.

CLI:
    python -m scripts.caption.review \
        --captions captions.jsonl --index thumbs/index.jsonl --thumb-dir thumbs \
        --out-dir review --per-sheet 6 --inject-controls 6
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

CELL = 512
LABEL_HEIGHT = 28
SHEET_COLUMNS = 2


def load_captions(path: str) -> List[Dict]:
    records = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error") or not record.get("text"):
            continue
        records.append(record)
    return records


def make_control(text: str, rng: random.Random) -> str:
    """Corrupt one checkable fact in a caption."""
    numbers = {
        "two": "four", "three": "seven", "four": "one", "five": "nine",
        "二": "九", "三": "七", "四": "一", "五": "八", "两": "六",
        "left": "right", "right": "left", "左": "右", "右": "左",
        "red": "bright green", "blue": "orange", "红": "翠绿", "蓝": "橙",
    }
    choices = [(key, value) for key, value in numbers.items() if key in text]
    if not choices:
        return text + (" An additional elephant stands at the far left edge."
                       if not any("\u4e00" <= ch <= "\u9fff" for ch in text)
                       else "画面最左侧另有一头大象。")
    key, value = rng.choice(choices)
    return text.replace(key, value, 1)


def build_sheet(records: List[Dict], index: Dict[str, Dict], thumb_dir: Path,
                out_path: Path, start: int) -> None:
    columns = SHEET_COLUMNS
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * CELL, rows * (CELL + LABEL_HEIGHT)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, record in enumerate(records):
        entry = index[record["image_id"]]
        path = thumb_dir / Path(entry["thumb_path"]).name
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image.thumbnail((CELL, CELL), Image.LANCZOS)
        column = position % columns
        row = position // columns
        x = column * CELL
        y = row * (CELL + LABEL_HEIGHT)
        sheet.paste(image, (x + (CELL - image.width) // 2, y + LABEL_HEIGHT))
        draw.rectangle([x, y, x + CELL - 1, y + LABEL_HEIGHT - 1], fill="black")
        draw.text((x + 6, y + 7), f"[{start + position}] {record['image_id']}", fill="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--thumb-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-sheet", type=int, default=6)
    parser.add_argument("--inject-controls", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wrap", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {json.loads(line)["image_id"]: json.loads(line)
             for line in Path(args.index).open(encoding="utf-8") if line.strip()}
    records = load_captions(args.captions)

    rng = random.Random(args.seed)
    control_ids = set()
    if args.inject_controls:
        for record in rng.sample(records, min(args.inject_controls, len(records))):
            record["text"] = make_control(record["text"], rng)
            record["control"] = True
            control_ids.add(record["image_id"])

    manifest = []
    for sheet_number, start in enumerate(range(0, len(records), args.per_sheet)):
        batch = records[start:start + args.per_sheet]
        sheet_path = out_dir / f"sheet_{sheet_number:03d}.jpg"
        build_sheet(batch, index, Path(args.thumb_dir), sheet_path, start)
        lines = [f"# sheet {sheet_number:03d}  ({len(batch)} images)"]
        for offset, record in enumerate(batch):
            lines.append(f"\n[{start + offset}] {record['image_id']}  "
                         f"lang={record['language']} fmt={record['format']} "
                         f"band={record['band']} retained={record['retained_tokens']}")
            lines.append(textwrap.fill(record["text"], width=args.wrap))
        (out_dir / f"sheet_{sheet_number:03d}.txt").write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")
        manifest.append({
            "sheet": str(sheet_path),
            "captions": str(out_dir / f"sheet_{sheet_number:03d}.txt"),
            "image_ids": [record["image_id"] for record in batch],
            "models": sorted({record["model"] for record in batch}),
            "controls": [record["image_id"] for record in batch if record.get("control")],
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (out_dir / "control_ids.txt").write_text("\n".join(sorted(control_ids)) + "\n",
                                             encoding="utf-8")
    print(f"{len(records)} captions in {len(manifest)} sheets under {out_dir}")
    print(f"controls injected: {len(control_ids)}")


if __name__ == "__main__":
    main()
