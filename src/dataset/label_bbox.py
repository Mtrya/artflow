"""Second-pass VLM bounding-box labeling for artifact removal.

Reads the first-pass labels jsonl (src.dataset.label_vlm output), selects
images whose label.artifacts is non-empty (view_type rolled/junk excluded —
those are dropped at assembly), and asks the VLM for the bounding box of the
main artwork region: the painting/album-spread/scroll INCLUDING its mounting,
but EXCLUDING color charts, rulers, accession labels, desk/background.

Output: one JSONL row per image with a normalized bbox [y0, x0, y1, x1] on a
0..1000 grid (Gemini convention). Crops are applied later (see
scripts/data/apply_bboxes.py), typically on GPFS after transfer. Raw responses
are cached so reruns are idempotent and pay nothing twice.

CLI:
    python -m src.dataset.label_bbox \
        --labels data/labels/npm_tw_c2/labels_google_gemini-3.5-flash-lite.jsonl \
        --clean data/clean/npm_tw_c2 \
        --out data/labels/npm_tw_c2_bbox \
        --model google/gemini-3.5-flash-lite --concurrency 8
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
from typing import Dict, List, Optional

import httpx
from PIL import Image

BASE_URL = "https://zenmux.ai/api/v1/chat/completions"
MAX_EDGE = 1024  # match label_vlm: downscale before base64

PROMPT = """这张照片里有博物馆藏品(绘画/书法/册页/立轴),但画面边缘可能混入了摄影存档用的辅助物:比色卡(一排彩色方块)、刻度尺、藏品编号标签条、黑色/灰色的桌面或摄影背景。

请只输出一个 JSON 对象(不要输出其他内容):
{"bbox": [y0, x0, y1, x1]}

bbox 是"主体艺术品区域"的边界框:包含完整的作品本身及其装裱(裱绢、镶边、天地杆范围内),但把所有比色卡、刻度尺、标签条、桌面/背景排除在外。坐标归一化到 0~1000 的整数,顺序为 [上, 左, 下, 右]。

要求:
- 框要紧贴艺术品(含装裱),宁可多保留一圈装裱边,也不能切到画面内容。
- 若整张照片已经是纯艺术品、没有任何辅助物或背景,输出 [0, 0, 1000, 1000]。
- 若艺术品在画面中倾斜,用能完整包含它的轴对齐矩形。"""


def encode_image(path: str) -> str:
    Image.MAX_IMAGE_PIXELS = None  # legit museum scans exceed PIL's bomb limit
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_EDGE, MAX_EDGE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def parse_bbox(text: str) -> Optional[List[int]]:
    """Extract {"bbox": [y0,x0,y1,x1]} from a model response; None if unusable."""
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    box = obj.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        box = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    y0, x0, y1, x1 = box
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        return None
    if (y1 - y0) < 100 or (x1 - x0) < 100:  # less than 10% on a side: unusable
        return None
    return box


async def bbox_one(client: httpx.AsyncClient, model: str, image_id: str,
                   img_path: str, cache_dir: str, max_retries: int = 3) -> Dict:
    cache_path = os.path.join(cache_dir, f"{image_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return {"image_id": image_id, "bbox": cached.get("bbox"), "cached": True}
    out = {"image_id": image_id, "bbox": None, "error": None}
    try:
        b64 = encode_image(img_path)
    except Exception as exc:  # unreadable image: cache the failure, keep going
        out["error"] = f"encode: {type(exc).__name__}: {exc}"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"bbox": None, "error": out["error"]}, f, ensure_ascii=False)
        return out
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            resp = await client.post(BASE_URL, json=body)
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(2 ** attempt * 2)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            out["bbox"] = parse_bbox(content)
            if out["bbox"] is None:
                out["error"] = f"unparseable: {content[:150]}"
            break
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2 ** attempt * 2)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"bbox": out["bbox"], "error": out["error"]}, f, ensure_ascii=False)
    return out


async def run(labels_path: str, clean_dir: str, out_dir: str, model: str,
              n: int, concurrency: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "vlm_cache", model.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    todo = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            l = d.get("label")
            if not l:
                continue
            if l.get("view_type") in ("rolled", "junk"):
                continue
            if not (l.get("artifacts") or []):
                continue
            img = os.path.join(clean_dir, "images", f"{d['image_id']}.jpg")
            if os.path.exists(img):
                todo.append((d["image_id"], img))
    todo = todo[:n]
    print(f"bbox pass: {len(todo)} flagged images")
    key = os.environ.get("ZENMUX_API_KEY")
    assert key, "ZENMUX_API_KEY not set"
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        async def task(pair):
            async with sem:
                return await bbox_one(client, model, pair[0], pair[1], cache_dir)
        results = await asyncio.gather(*(task(p) for p in todo))
    dt = time.time() - t0
    ok = [r for r in results if r.get("bbox")]
    out_path = os.path.join(out_dir, f"bboxes_{model.replace('/', '_')}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"model={model} n={len(results)} ok={len(ok)} wall={dt:.0f}s -> {out_path}")
    for r in results:
        if r.get("error"):
            print("  ERR", r["image_id"], r["error"][:120])


def main():
    ap = argparse.ArgumentParser(description="VLM bbox pass for artifact removal")
    ap.add_argument("--labels", required=True, help="first-pass labels jsonl")
    ap.add_argument("--clean", required=True, help="clean dir with images/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(run(args.labels, args.clean, args.out, args.model, args.n,
                    args.concurrency))


if __name__ == "__main__":
    main()
