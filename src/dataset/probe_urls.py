"""Probe URL aliveness for kaupane/human-recaption (link-rot estimate).

Samples N rows from human_train.jsonl, fetches each image_url concurrently,
reports status code / content-type / byte-size distribution and host stats.

Usage:
    python -m src.dataset.probe_urls --in human_train.jsonl --n 500 --concurrency 32
"""

import argparse
import asyncio
import json
import random
from collections import Counter
from urllib.parse import urlparse

import httpx

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) artflow-dataset-probe"}


async def probe_one(client: httpx.AsyncClient, url: str) -> dict:
    out = {"url": url, "ok": False, "status": None, "bytes": 0, "ctype": "", "err": None}
    try:
        async with client.stream("GET", url) as resp:
            out["status"] = resp.status_code
            out["ctype"] = resp.headers.get("content-type", "")
            n = 0
            async for chunk in resp.aiter_bytes(1 << 16):
                n += len(chunk)
                if n >= 1 << 20:  # cap at 1MB per image; enough to verify
                    break
            out["bytes"] = n
            out["ok"] = (resp.status_code == 200
                         and "image" in out["ctype"]
                         and n > 4096)  # >4KB, not an error placeholder
    except Exception as exc:
        out["err"] = type(exc).__name__
    return out


async def run(meta_path: str, n: int, concurrency: int) -> None:
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["image_url"])
    print(f"total rows: {len(rows)}")
    random.seed(42)
    sample = random.sample(rows, min(n, len(rows)))
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers=UA, timeout=30, follow_redirects=True,
                                 verify=False) as client:
        async def task(u):
            async with sem:
                return await probe_one(client, u)
        results = await asyncio.gather(*(task(u) for u in sample))
    ok = [r for r in results if r["ok"]]
    print(f"sampled {len(results)}: alive {len(ok)} ({len(ok)/len(results)*100:.1f}%)")
    print("status:", Counter(r["status"] for r in results).most_common(10))
    print("errors:", Counter(r["err"] for r in results if r["err"]).most_common(10))
    print("ctype(not ok):", Counter(r["ctype"] for r in results if not r["ok"]).most_common(8))
    hosts = Counter(urlparse(r["url"]).netloc for r in results)
    dead_hosts = Counter(urlparse(r["url"]).netloc for r in results if not r["ok"])
    print("top hosts (alive/total):")
    for h, tot in hosts.most_common(15):
        print(f"  {h:45s} {tot - dead_hosts.get(h, 0):4d}/{tot}")
    if ok:
        sizes = sorted(r["bytes"] for r in ok)
        print(f"alive bytes p10/p50/p90: {sizes[len(sizes)//10]}, "
              f"{sizes[len(sizes)//2]}, {sizes[9*len(sizes)//10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="meta", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    asyncio.run(run(args.meta, args.n, args.concurrency))


if __name__ == "__main__":
    main()
