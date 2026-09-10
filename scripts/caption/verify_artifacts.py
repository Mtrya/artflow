"""Check whether a second-round crop still shows anything that is not the artwork.

The cropping round asks the model for the artwork box *inside* the view it is
shown, and separately lists what non-artwork objects are visible in that same
input view.  That list therefore describes the view before the new box is
applied, not the result after it.  The only way to know what the training image
will contain is to apply the box and look at the result, which is what this pass
does.

Input is the round-two box from ``refine_bbox.py``; the crop is applied to the
same 640 px thumbnails, so no extra image transfer is needed.

CLI:
    python -m scripts.caption.verify_artifacts \
        --refine flagged_selection.jsonl --index thumbs/index.jsonl \
        --boxes bbox_v2.jsonl --thumb-dir thumbs640 --out artifacts_final.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from PIL import Image

from src.dataset.caption_client import CaptionClient

PROMPT_VERSION = "artifacts-v1"

PROMPT = """这是一张已经按作品边界裁过的图。只回答一个问题：这张图里还能看到哪些不属于作品本身的摄影或存档辅助物？

- "color_chart": 比色卡、灰阶色条（一排彩色或灰度方块，常带数字字母编号）
- "ruler": 刻度尺
- "label": 藏品编号标签条、条码、带馆藏号的纸条
- "desk": 黑色洞洞板、摄影台、桌面等拍摄背景（画作的装裱、裱边不算）
- "glare": 明显反光
- "other": 其他异物

要求：
- 只判断异物，不要描述画面内容。
- 装裱的裱绢、镶边、天地头属于作品本身，不算异物。
- 画面边缘残留的极小一条色卡或桌面，如果只占图像很小一部分，仍然算异物，如实列出。
- 只输出一个 JSON 对象：{"artifacts": []}，没有异物就输出空数组。"""

ARTIFACT_KINDS = {"color_chart", "ruler", "label", "desk", "glare", "other"}


def parse_artifacts(text: str) -> Optional[List[str]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    values = obj.get("artifacts")
    if not isinstance(values, list):
        return None
    return sorted({v for v in values if v in ARTIFACT_KINDS})


def crop_thumb(image: Image.Image, box: List[int]) -> bytes:
    width, height = image.size
    x1, y1, x2, y2 = box
    region = image.crop((int(x1 / 1000 * width), int(y1 / 1000 * height),
                         int(x2 / 1000 * width), int(y2 / 1000 * height)))
    buffer = io.BytesIO()
    region.convert("RGB").save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


async def run(args) -> None:
    boxes = {}
    for line in Path(args.boxes).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("round2") and record["round2"].get("bbox"):
            boxes[record["image_id"]] = record["round2"]["bbox"]

    done = set()
    if os.path.exists(args.out):
        for line in Path(args.out).open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                if record.get("artifacts") is not None:
                    done.add(record["image_id"])

    todo = []
    for line in Path(args.refine).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["image_id"] in boxes and row["image_id"] not in done:
            todo.append(row)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return
    print(f"{len(todo)} rows to check ({len(done)} already done)")

    client = CaptionClient(args.cache_dir, args.model, concurrency=args.concurrency,
                           timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_dir = Path(args.thumb_dir)
    stats = {"clean": 0, "dirty": 0, "error": 0}
    finished = 0
    with out_path.open("a", encoding="utf-8") as sink:
        async with httpx.AsyncClient() as http:
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(row):
                async with semaphore:
                    path = thumb_dir / f"{row['image_id']}.jpg"
                    with Image.open(path) as handle:
                        payload = crop_thumb(handle, boxes[row["image_id"]])
                    return row, await client.generate(
                        http, image_bytes=payload, prompt=PROMPT,
                        prompt_version=PROMPT_VERSION, max_tokens=args.max_tokens,
                        temperature=0.0)

            tasks = [asyncio.create_task(one(row)) for row in todo]
            for coro in asyncio.as_completed(tasks):
                row, response = await coro
                artifacts = parse_artifacts(response.text) if not response.error else None
                if artifacts is None:
                    stats["error"] += 1
                elif artifacts:
                    stats["dirty"] += 1
                else:
                    stats["clean"] += 1
                sink.write(json.dumps({
                    "image_id": row["image_id"],
                    "source": row.get("source"),
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "artifacts": artifacts,
                    "clean": artifacts == [],
                    "cost_usd": response.cost_usd,
                    "cached": response.cached,
                    "error": response.error,
                    "usage": response.usage,
                }, ensure_ascii=False) + "\n")
                finished += 1
                if finished % 500 == 0:
                    sink.flush()
                    print(f"  {finished}/{len(todo)} {stats}", flush=True)
    print(f"done: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", required=True)
    parser.add_argument("--boxes", required=True, help="round-two box JSONL")
    parser.add_argument("--thumb-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/artflow_artifacts"))
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
