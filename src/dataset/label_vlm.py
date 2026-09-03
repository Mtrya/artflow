"""VLM captioning probe / batch labeler via ZenMux (OpenAI-compatible).

Reads a cleaned metadata.jsonl (from src.dataset.clean), sends each image to a
ZenMux chat-completions model with a structured labeling prompt, and writes one
JSONL row per image with the parsed label plus raw response. Raw responses are
cached under the output dir so reruns are idempotent and pay nothing twice.

CLI:
    ZENMUX_API_KEY=... python -m src.dataset.label_vlm \
        --in /tmp/npm_clean2/metadata.jsonl --out /tmp/vlm_probe \
        --model google/gemini-3.5-flash-lite --n 40 --concurrency 8
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
from typing import Dict, Optional

import httpx
from PIL import Image

from src.dataset.dedup import resolve_path
from src.dataset.domain_filter import assign_domain

BASE_URL = "https://zenmux.ai/api/v1/chat/completions"
MAX_EDGE = 1024  # downscale before base64 to bound image tokens

PROMPT = """你是一名艺术史数据标注员。请观察这张来自博物馆藏品的图片,只输出一个 JSON 对象(不要输出其他内容):
{
  "view_type": "full" | "detail" | "mounted" | "rolled" | "junk",
  "culture": "chinese" | "japanese" | "korean" | "western" | "other" | "uncertain",
  "artifacts": [],
  "ocr_text": "",
  "caption_zh": "<160~240 字中文描述>",
  "caption_en": "<80~160 词英文描述>"
}

view_type 定义:
- full: 画面主体完整可见的作品本身(可以带少量装裱边)
- detail: 明显只是作品的局部特写
- mounted: 带完整装裱的档案照(整幅立轴/手卷连裱)
- rolled: 收卷、包装或侧面状态,画面不可见(典型形态:扁平淡色桌面上横放或立放一个圆筒状/柱状物)
- junk: 其他无法作为画作训练数据的情况

culture 指作品的文化归属(不是拍摄方式):中国画(含水墨、设色、文人画、院体画等)记 chinese;日本画(浮世绘、大和绘、琳派、南画等)记 japanese;西洋油画/水彩/素描记 western;东亚器物照片(青铜器、玉器、瓷器、家具等,即使带有绘画装饰)记 other;看不出来记 uncertain。

书法与题款的特殊要求:
- 如果作品本身是书法,或画面上有可辨认的书法题字/题款/引首/跋文,必须在 "ocr_text" 字段逐字录出可辨认的汉字:繁体原样照录,保持原行款从右到左、每列一行的顺序用换行分隔,无法辨认的字用 □ 代替。完全没有文字内容的作品输出空字符串 ""。
- 画面上的印章也要处理:能辨认印文的把印文并入 ocr_text(可单独成行并注明【印】);同时在 caption 中描述印章的位置、数量、朱文/白文。
- 含大量文字的此类作品,caption_zh 可放宽到 400 字以内;ocr_text 不计入字数限制。

artifacts 列出画面中不属于作品本身的摄影/存档痕迹(无则输出空数组 []):
- "color_chart": 比色卡/色标(一排彩色方块)
- "ruler": 刻度尺
- "label": 藏品编号标签、条码条
- "desk": 黑色洞洞板/摄影台/工作室桌面等拍摄背景(不是装裱)
- "glare": 明显反光、玻璃反光
- "other": 其他明显异物(在 caption 里说明)

caption 要求:描述题材、构图、笔墨/设色(或媒介技法)、风格、氛围与值得注意的细节;直接以画面内容开头,禁止"这幅图片/这是一幅/该作品"一类套话;不要臆造作者、年代、收藏历史;看不清就说不确定。"""

VIEW_TYPES = {"full", "detail", "mounted", "rolled", "junk"}
CULTURES = {"chinese", "japanese", "korean", "western", "other", "uncertain"}


def encode_image(path: str) -> str:
    Image.MAX_IMAGE_PIXELS = None  # legit museum scans exceed PIL's default bomb limit
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_EDGE, MAX_EDGE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def parse_label(text: str) -> Optional[Dict]:
    """Extract the JSON object from a model response; None if unusable."""
    s = text.find("{")  # also skips a leading ```json fence
    e = text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    if obj.get("view_type") not in VIEW_TYPES:
        return None
    if obj.get("culture") not in CULTURES:
        return None
    if not obj.get("caption_zh") or not obj.get("caption_en"):
        return None
    if not isinstance(obj.get("artifacts"), list):
        obj["artifacts"] = []
    if not isinstance(obj.get("ocr_text"), str):
        obj["ocr_text"] = ""
    return obj


async def label_one(client: httpx.AsyncClient, model: str, rec: Dict, cache_dir: str,
                    max_retries: int = 3) -> Dict:
    image_id = rec["image_id"]
    cache_path = os.path.join(cache_dir, f"{image_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return {"image_id": image_id, "label": cached.get("label"),
                "usage": cached.get("usage"), "cached": True}
    out = {"image_id": image_id, "label": None, "usage": None, "error": None}
    try:
        b64 = encode_image(rec["local_path"])
    except Exception as exc:  # unreadable/oversized image: cache the failure, keep going
        out["error"] = f"encode: {type(exc).__name__}: {exc}"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"label": None, "usage": None, "error": out["error"]},
                      f, ensure_ascii=False)
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
        "max_tokens": 1200,
        "temperature": 0.2,
    }
    out = {"image_id": image_id, "label": None, "usage": None, "error": None}
    for attempt in range(max_retries):
        try:
            resp = await client.post(BASE_URL, json=body)
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(2 ** attempt * 2)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            out["usage"] = data.get("usage")
            out["label"] = parse_label(content)
            if out["label"] is None:
                out["error"] = f"unparseable: {content[:200]}"
            break
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2 ** attempt * 2)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"label": out["label"], "usage": out["usage"], "error": out["error"]},
                  f, ensure_ascii=False)
    return out


async def run(meta_path: str, out_dir: str, model: str, n: int, concurrency: int,
              domains: Optional[set] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "vlm_cache", model.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    with open(meta_path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    recs = [r for r in recs if not r.get("rejected")]
    if domains:
        recs = [r for r in recs
                if assign_domain(r.get("source") or r["image_id"].split("-")[0], r) in domains]
    for r in recs:
        r["local_path"] = resolve_path(r["local_path"], meta_path)
    recs = recs[:n]
    key = os.environ.get("ZENMUX_API_KEY")
    assert key, "ZENMUX_API_KEY not set"
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        async def task(rec):
            async with sem:
                return await label_one(client, model, rec, cache_dir)
        results = await asyncio.gather(*(task(r) for r in recs))
    dt = time.time() - t0
    ok = [r for r in results if r.get("label")]
    uncached = [r for r in results if not r.get("cached")]
    in_tok = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in results)
    out_tok = sum((r["usage"] or {}).get("completion_tokens", 0) for r in results)
    out_path = os.path.join(out_dir, f"labels_{model.replace('/', '_')}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"model={model} n={len(results)} ok={len(ok)} "
          f"in_tok={in_tok} out_tok={out_tok} "
          f"wall={dt:.0f}s ({dt / max(1, len(uncached)):.1f}s/img uncached) -> {out_path}")
    for r in results:
        if r.get("error"):
            print("  ERR", r["image_id"], r["error"][:120])


def main():
    ap = argparse.ArgumentParser(description="VLM labeling probe via ZenMux")
    ap.add_argument("--in", dest="meta", required=True, help="cleaned metadata.jsonl")
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--domains", default=None,
                    help="comma-separated domain tags to keep (default: keep all)")
    args = ap.parse_args()
    domains = set(args.domains.split(",")) if args.domains else None
    asyncio.run(run(args.meta, args.out_dir, args.model, args.n, args.concurrency, domains))


if __name__ == "__main__":
    main()
