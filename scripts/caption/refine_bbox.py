"""Second cropping round for rows whose training view still shows artifacts.

Stage 1 cropped each photograph once, from a box the model gave for the whole
page.  For digitised book pages that is not enough: the first box often keeps
the colour chart or the accession strip that sits beside the artwork, because
the model was looking at a page where the strip is a small part of the frame.

This pass shows the model the image it has *already* been cropped to, and asks
for the artwork box within that view.  The two boxes compose into one box in
the coordinates of the original photograph, which is what the precompute needs.
The model also reports which artifacts it can still see, so a row that survives
two rounds can be dropped instead of trained on.

CLI:
    python -m scripts.caption.refine_bbox \
        --refine flagged_selection.jsonl --index thumbs/index.jsonl \
        --out bbox_v2.jsonl --cache-dir ~/.cache/artflow_bbox
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from src.dataset.caption_client import CaptionClient, provider_for

PROMPT_VERSION = "bbox-v2"

PROMPT = """这张图片已经是从原始照片里裁过一轮的结果。请判断:图里"作品本身"的完整边界在哪里?

只输出一个 JSON 对象:
{"bbox": [x1, y1, x2, y2], "artifacts": [], "note": ""}

bbox 坐标归一化到 0~1000 的整数,顺序为 [左, 上, 右, 下],相对的是**当前这张图**。
artifacts 列出仍然出现在当前图中的、不属于作品的摄影或存档辅助物,没有就输出 []:
- "color_chart": 比色卡/灰阶色条(一排彩色或灰度方块,常带数字字母编号)
- "ruler": 刻度尺
- "label": 藏品编号标签条、条码、带馆藏号的纸条
- "desk": 黑色洞洞板/摄影台/桌面等拍摄背景(不是装裱)
- "glare": 明显反光
- "other": 其他异物

要求:
- bbox 要**紧贴作品本身**(含装裱的裱绢、镶边),把所有比色卡、刻度尺、标签条、桌面背景排除在外。
- 宁可多留一圈装裱边,也不能切到画面内容。
- 如果当前图里已经没有辅助物、整张就是作品,bbox 输出 [0, 0, 1000, 1000],artifacts 输出 []。
- 如果作品在当前图里倾斜,用能完整包含它的轴对齐矩形。"""

ARTIFACT_KINDS = {"color_chart", "ruler", "label", "desk", "glare", "other"}


def parse_response(text: str) -> Optional[Dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    box = obj.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        box = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    if (x2 - x1) < 100 or (y2 - y1) < 100:
        return None
    artifacts = [a for a in (obj.get("artifacts") or []) if a in ARTIFACT_KINDS]
    return {"bbox": box, "artifacts": artifacts, "note": str(obj.get("note") or "")[:200]}


def compose(round1_box: Optional[List[int]], round2_box: List[int],
            image_width: int, image_height: int) -> List[float]:
    """Map a box found on the round-1 crop back to the original photograph.

    Both boxes are normalised to 0-1000; ``round1_box`` is the integer crop
    actually applied (``None`` means the crop was the whole image).
    """
    if round1_box:
        left, top, right, bottom = round1_box
    else:
        left, top, right, bottom = 0, 0, image_width, image_height
    span_x = right - left
    span_y = bottom - top
    x1, y1, x2, y2 = round2_box
    fx1 = left + x1 / 1000.0 * span_x
    fy1 = top + y1 / 1000.0 * span_y
    fx2 = left + x2 / 1000.0 * span_x
    fy2 = top + y2 / 1000.0 * span_y
    return [
        round(max(0.0, min(1000.0, fx1 / image_width * 1000.0)), 1),
        round(max(0.0, min(1000.0, fy1 / image_height * 1000.0)), 1),
        round(max(0.0, min(1000.0, fx2 / image_width * 1000.0)), 1),
        round(max(0.0, min(1000.0, fy2 / image_height * 1000.0)), 1),
    ]


async def run(args) -> None:
    refine = [json.loads(line) for line in Path(args.refine).open(encoding="utf-8")
              if line.strip()]
    index = {json.loads(line)["image_id"]: json.loads(line)
             for line in Path(args.index).open(encoding="utf-8") if line.strip()}
    done = set()
    if os.path.exists(args.out):
        for line in Path(args.out).open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["image_id"])
    todo = [row for row in refine if row["image_id"] in index
            and row["image_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return
    print(f"{len(todo)} rows to refine ({len(done)} already done)")

    client = CaptionClient(args.cache_dir, args.model, concurrency=args.concurrency,
                           timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = 0
    stats = {"clean": 0, "still_dirty": 0, "error": 0, "unchanged": 0}
    with out_path.open("a", encoding="utf-8") as sink:
        async with httpx.AsyncClient() as http:
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(row):
                async with semaphore:
                    entry = index[row["image_id"]]
                    image_bytes = Path(args.thumb_dir, Path(entry["thumb_path"]).name).read_bytes()
                    return row, entry, await client.generate(
                        http, image_bytes=image_bytes, prompt=PROMPT,
                        prompt_version=PROMPT_VERSION, max_tokens=args.max_tokens,
                        temperature=0.0)

            tasks = [asyncio.create_task(one(row)) for row in todo]
            for coro in asyncio.as_completed(tasks):
                row, entry, response = await coro
                parsed = parse_response(response.text) if not response.error else None
                record = {
                    "image_id": row["image_id"],
                    "source": row.get("source"),
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "round1_bbox": row.get("bbox"),
                    "crop_box": entry.get("crop_box"),
                    "image_width": entry.get("image_width"),
                    "image_height": entry.get("image_height"),
                    "round2": parsed,
                    "bbox_v2": None,
                    "cost_usd": response.cost_usd,
                    "cached": response.cached,
                    "error": response.error,
                    "usage": response.usage,
                }
                if parsed:
                    record["bbox_v2"] = compose(
                        entry.get("crop_box"), parsed["bbox"],
                        entry["image_width"], entry["image_height"])
                    if parsed["artifacts"]:
                        stats["still_dirty"] += 1
                    else:
                        stats["clean"] += 1
                    if record["bbox_v2"] == row.get("bbox"):
                        stats["unchanged"] += 1
                else:
                    stats["error"] += 1
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                finished += 1
                if finished % 200 == 0:
                    sink.flush()
                    print(f"  {finished}/{len(todo)} {stats}", flush=True)
    print(f"done: {stats}")
    billed = sum(1 for line in Path(args.out).open(encoding="utf-8")
                 if line.strip() and not json.loads(line).get("cached"))
    cost = sum(json.loads(line)["cost_usd"] for line in Path(args.out).open(encoding="utf-8")
               if line.strip())
    print(f"requests billed: {billed}, total cost: ${cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", required=True, help="flagged-rows manifest JSONL")
    parser.add_argument("--index", required=True, help="thumbnail index JSONL")
    parser.add_argument("--thumb-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/artflow_bbox"))
    parser.add_argument("--model", default="google/gemini-3.5-flash-lite")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
