"""Fetcher for the Metropolitan Museum of Art Collection API (CC0).

API: https://collectionapi.metmuseum.org/public/collection/v1/
- GET /search?departmentId=<id>&hasImages=true&q=<query> -> objectID list
- GET /objects/<id> -> object detail (title, artist, date, isPublicDomain, primaryImage)

Note: `isPublicDomain` is NOT a valid search filter — filter on the object detail.
Rate limit: 80 req/s (we stay far below).

CLI:
    python -m src.dataset.fetchers.met --out data/raw/met --department 6 --query chinese --limit 200
"""

import argparse
import time
from typing import Dict, Iterator, Optional

from .common import JsonlWriter, make_session, request_with_retry, save_jpeg

API = "https://collectionapi.metmuseum.org/public/collection/v1"
LICENSE_PD = "CC0 (Met Open Access)"
LICENSE_NON_PD = "not public domain (Met)"


def search_objects(session, department: int, query: str) -> list:
    resp = request_with_retry(session, "GET", f"{API}/search",
                              params={"departmentId": department, "hasImages": "true", "q": query})
    if resp is None:
        raise RuntimeError("search failed after retries")
    return resp.json().get("objectIDs") or []


def fetch_object(session, object_id: int) -> Optional[Dict]:
    resp = request_with_retry(session, "GET", f"{API}/objects/{object_id}")
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def run(out_dir: str, department: int, query: str, limit: int,
        public_domain_only: bool = True, delay: float = 0.1) -> None:
    session = make_session()
    writer = JsonlWriter(f"{out_dir}/metadata.jsonl")
    ids = search_objects(session, department, query)
    print(f"search: {len(ids)} object ids (dept={department}, q={query!r})")
    n_img = n_obj = 0
    for oid in ids:
        if n_obj >= limit:
            break
        n_obj += 1
        image_id = f"met-{oid}"
        if image_id in writer.seen:
            continue
        obj = fetch_object(session, oid)
        if not obj:
            continue
        if public_domain_only and not obj.get("isPublicDomain"):
            continue
        url = obj.get("primaryImage")
        if not url:
            continue
        resp = request_with_retry(session, "GET", url)
        if resp is None or len(resp.content) < 1024 or not resp.content.startswith(b"\xff\xd8\xff"):
            print(f"[skip] {image_id}: image fetch failed")
            continue
        path = f"{out_dir}/images/{image_id}.jpg"
        save_jpeg(resp.content, path)
        writer.write({
            "image_id": image_id,
            "source": "met",
            "object_id": str(oid),
            "title": obj.get("title", ""),
            "artist": obj.get("artistDisplayName", ""),
            "date": obj.get("objectDate", ""),
            "culture": obj.get("culture", ""),
            "period": obj.get("period", ""),
            "medium": obj.get("medium", ""),
            "classification": obj.get("classification", ""),
            "department": obj.get("department", ""),
            "tags": [t.get("term") for t in (obj.get("tags") or [])],
            "object_url": obj.get("objectURL", ""),
            "license": LICENSE_PD if obj.get("isPublicDomain") else LICENSE_NON_PD,
            "local_path": path,
        })
        n_img += 1
        time.sleep(delay)
        if n_obj % 50 == 0:
            print(f"[progress] objects={n_obj} images={n_img}")
    print(f"done: {n_img} images from {n_obj} objects -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Met Museum collection images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--department", type=int, required=True, help="6=Asian Art, 11=European Paintings")
    ap.add_argument("--query", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--include-non-public-domain", action="store_true")
    ap.add_argument("--delay", type=float, default=0.1)
    args = ap.parse_args()
    run(args.out, args.department, args.query, args.limit,
        public_domain_only=not args.include_non_public_domain, delay=args.delay)


if __name__ == "__main__":
    main()
