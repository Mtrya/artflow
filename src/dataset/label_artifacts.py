"""Cheap artifacts-only VLM pass for batches labeled before the artifacts field.

Old label batches (museum/impressionist sets) have no "artifacts" key. This pass
re-scans only full/mounted views with a short artifacts-only prompt (~1/3 the cost
of a full relabel) and writes one JSONL row per image: {image_id, artifacts, ...}.

Idempotent: raw responses are cached under <out>/vlm_cache/<model>/<image_id>.json;
reruns skip cached ids. Purge cache files with "error" set to retry failures.

CLI:
    ZENMUX_API_KEY=... python -m src.dataset.label_artifacts \
        --clean data/clean/aic_china \
        --labels data/labels/aic_china/labels_google_gemini-3.5-flash-lite.jsonl \
        --out data/labels/aic_china_artifacts \
        --model google/gemini-3.5-flash-lite --concurrency 8
"""

import argparse
import asyncio
import json
import os
import time
from typing import Dict, Optional

import httpx

from src.dataset.label_vlm import BASE_URL, encode_image
from src.dataset.dedup import resolve_path

PROMPT = """这是一张博物馆藏品的档案照片。请只检查画面中是否有不属于作品本身的摄影/存档痕迹,只输出一个 JSON 对象(不要输出其他内容):
{"artifacts": []}

artifacts 可选值(无则输出空数组):
- "color_chart": 比色卡/色标(一排彩色方块)
- "ruler": 刻度尺
- "label": 藏品编号标签、条码条
- "desk": 黑色洞洞板/摄影台/工作室桌面等拍摄背景(不是装裱)
- "glare": 明显反光、玻璃反光
- "other": 其他明显异物"""

ARTIFACTS = {"color_chart", "ruler", "label", "desk", "glare", "other"}
KEEP_VIEWS = {"full", "mounted"}


def parse_artifacts(text: str) -> Optional[list]:
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    arts = obj.get("artifacts")
    if not isinstance(arts, list):
        return None
    return [a for a in arts if a in ARTIFACTS]


async def tag_one(client: httpx.AsyncClient, model: str, image_id: str, path: str,
                  cache_dir: str, max_retries: int = 3) -> Dict:
    cache_path = os.path.join(cache_dir, f"{image_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return {"image_id": image_id, "artifacts": cached.get("artifacts"),
                "usage": cached.get("usage"), "cached": True}
    out = {"image_id": image_id, "artifacts": None, "usage": None, "error": None}
    try:
        b64 = encode_image(path)
    except Exception as exc:
        out["error"] = f"encode: {type(exc).__name__}: {exc}"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"artifacts": None, "usage": None, "error": out["error"]},
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
        "max_tokens": 100,
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
            out["usage"] = data.get("usage")
            out["artifacts"] = parse_artifacts(content)
            if out["artifacts"] is None:
                out["error"] = f"unparseable: {content[:200]}"
            break
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2 ** attempt * 2)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"artifacts": out["artifacts"], "usage": out["usage"],
                   "error": out["error"]}, f, ensure_ascii=False)
    return out


async def run(clean_dir: str, labels_path: str, out_dir: str, model: str,
              concurrency: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "vlm_cache", model.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(clean_dir, "metadata.jsonl")
    paths = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("rejected"):
                paths[r["image_id"]] = resolve_path(r["local_path"], meta_path)

    todo = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lab = r.get("label") or {}
            if lab.get("view_type") in KEEP_VIEWS and r["image_id"] in paths:
                todo.append((r["image_id"], paths[r["image_id"]]))

    key = os.environ.get("ZENMUX_API_KEY")
    assert key, "ZENMUX_API_KEY not set"
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        async def task(iid, p):
            async with sem:
                return await tag_one(client, model, iid, p, cache_dir)
        results = await asyncio.gather(*(task(i, p) for i, p in todo))
    dt = time.time() - t0
    ok = [r for r in results if r.get("artifacts") is not None]
    in_tok = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in results)
    out_tok = sum((r["usage"] or {}).get("completion_tokens", 0) for r in results)
    out_path = os.path.join(out_dir, f"artifacts_{model.replace('/', '_')}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"model={model} n={len(results)} ok={len(ok)} "
          f"in_tok={in_tok} out_tok={out_tok} wall={dt:.0f}s -> {out_path}")
    for r in results:
        if r.get("error"):
            print("  ERR", r["image_id"], r["error"][:120])


def main():
    ap = argparse.ArgumentParser(description="Cheap artifacts-only VLM pass")
    ap.add_argument("--clean", required=True, help="clean dir with metadata.jsonl + images/")
    ap.add_argument("--labels", required=True, help="existing labels jsonl (for view_type filter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(run(args.clean, args.labels, args.out, args.model, args.concurrency))


if __name__ == "__main__":
    main()
