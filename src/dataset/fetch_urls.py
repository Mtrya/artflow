"""Download images for URL-list datasets (kaupane/human-recaption style).

Reads a metadata jsonl with `image_url` (or generic `url` + `id`), applies
quality filters (watermark/mode/aesthetic_score only when those fields exist),
downloads images concurrently, and writes a cleaned metadata.jsonl with
local_path added. Idempotent: reruns skip already-downloaded images.

CLI:
    python -m src.dataset.fetch_urls --in data/meta/human_train.jsonl \
        --out data/raw/human_recaption --min-score 0.75 --concurrency 32
"""

import argparse
import asyncio
import hashlib
import json
import os
from urllib.parse import urlparse

import httpx

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) artflow-dataset-fetch"}
MAX_BYTES = 25 << 20  # skip absurdly large originals
MIN_BYTES = 4096

# Hosts measured 0%-alive in the 2026-08-26 probe (dead/blocked); skip outright.
DEAD_HOSTS = {
    "i.pinimg.com", "pbs.twimg.com", "3.bp.blogspot.com", "4.bp.blogspot.com",
    "cdn01.cdn.justjared.com", "imageresizer.static9.net.au",
}

MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8")


def looks_like_image(head: bytes) -> bool:
    return any(head.startswith(m) for m in MAGIC)


def record_url(rec: dict) -> str:
    return rec.get("image_url") or rec["url"]


def record_id(rec: dict) -> str:
    return rec.get("id") or "hr-" + hashlib.md5(record_url(rec).encode()).hexdigest()[:16]


async def fetch_one(client: httpx.AsyncClient, rec: dict, images_dir: str) -> dict:
    url = record_url(rec)
    image_id = record_id(rec)
    ext = ".jpg"
    path = os.path.join(images_dir, image_id + ext)
    out = dict(rec)
    out.update(image_id=image_id, local_path=path, rejected=False, reject_reason="")
    if os.path.exists(path):
        return out
    if urlparse(url).netloc in DEAD_HOSTS:
        out.update(rejected=True, reject_reason="dead_host")
        return out
    for attempt in range(2):
        try:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise httpx.HTTPError(f"status {resp.status_code}")
                n = 0
                head = b""
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    async for chunk in resp.aiter_bytes(1 << 16):
                        if not head:
                            head = chunk[:8]
                            if not looks_like_image(head):
                                raise httpx.HTTPError("not an image")
                        n += len(chunk)
                        if n > MAX_BYTES:
                            raise httpx.HTTPError("too large")
                        f.write(chunk)
                if n < MIN_BYTES:
                    raise httpx.HTTPError(f"too small ({n}B)")
                os.replace(tmp, path)
            return out
        except Exception as exc:
            for p in (path + ".part",):
                if os.path.exists(p):
                    os.unlink(p)
            if attempt == 1:
                out.update(rejected=True,
                           reject_reason=f"{type(exc).__name__}: {exc}"[:120])
            else:
                await asyncio.sleep(1)
    return out


async def run(meta_path: str, out_dir: str, min_score: float, concurrency: int,
              limit: int) -> None:
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    meta_out = os.path.join(out_dir, "metadata.jsonl")
    done_ids = set()
    if os.path.exists(meta_out):
        with open(meta_out, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["image_id"])
    recs = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("watermark"):
                continue
            if r.get("mode") == "grayscale":
                continue
            if (r.get("aesthetic_score") or 0) < min_score:
                continue
            iid = record_id(r)
            if iid in done_ids:
                continue
            recs.append(r)
            if limit and len(recs) >= limit:
                break
    print(f"to download: {len(recs)} (already done: {len(done_ids)})", flush=True)
    sem = asyncio.Semaphore(concurrency)
    n_ok = n_fail = 0
    async with httpx.AsyncClient(headers=UA, timeout=30, follow_redirects=True,
                                 verify=False, http2=False,
                                 limits=httpx.Limits(max_connections=64,
                                                     max_keepalive_connections=0)) as client:
        async def task(r):
            async with sem:
                return await fetch_one(client, r, images_dir)
        with open(meta_out, "a", encoding="utf-8") as mf:
            for i in range(0, len(recs), 500):
                batch = await asyncio.gather(*(task(r) for r in recs[i:i + 500]))
                for out in batch:
                    if not out["rejected"]:
                        mf.write(json.dumps(out, ensure_ascii=False) + "\n")
                        n_ok += 1
                    else:
                        n_fail += 1
                mf.flush()
                print(f"[progress] {i + len(batch)}/{len(recs)} ok={n_ok} fail={n_fail}",
                      flush=True)
    print(f"done: {n_ok} images ({n_fail} failed) -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.75)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args.meta, args.out, args.min_score, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
