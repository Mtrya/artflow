"""Stage-1 probe fetcher: Art Institute of Chicago (api.artic.edu, IIIF, CC0).

Two slices:
  (a) impressionism-era paintings: is_public_domain=true, classification painting,
      date_start in [1860, 1910], has image_id
  (b) Asian works: is_public_domain=true, has image_id, department "Arts of Asia"
      or place_of_origin in {China, Japan}

Images: https://www.artic.edu/iiif/2/{image_id}/full/!1024,1024/0/default.jpg
License: CC0 (is_public_domain). Writes images/ + metadata.parquet per common.py.

Access quirk: api.artic.edu works direct, but www.artic.edu (IIIF) sits behind
a Cloudflare managed challenge ("Just a moment...") for datacenter/CN IPs.
It passes from a US exit — image downloads go through the local mihomo proxy
(127.0.0.1:7897); select a US node (e.g. Washington) if the challenge returns.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (download_image, get_json, make_session, rate_sleep,
                    save_parquet)

API = "https://api.artic.edu/api/v1/artworks/search"
IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/!1024,1024/0/default.jpg"
FIELDS = ("id,title,artist_title,date_display,date_start,classification_title,"
          "department_title,place_of_origin,medium_display,image_id,is_public_domain")
PAGE_SIZE = 100

PD = {"term": {"is_public_domain": True}}
HAS_IMG = {"exists": {"field": "image_id"}}

QUERY_IMPRESSIONISM = {"bool": {"must": [
    PD, HAS_IMG,
    {"term": {"classification_title.keyword": "painting"}},
    {"range": {"date_start": {"gte": 1860, "lte": 1910}}},
]}}

QUERY_ASIAN = {"bool": {"must": [PD, HAS_IMG], "should": [
    {"match_phrase": {"department_title": "Arts of Asia"}},
    {"terms": {"place_of_origin.keyword": ["China", "Japan"]}},
], "minimum_should_match": 1}}


def fetch_slice(session, query: dict, want: int) -> list[dict]:
    """Page through the search endpoint until `want` records or exhausted.

    NOTE: api.artic.edu computes the page offset from the *current* request's
    `limit` ((page-1)*limit), so `limit` must stay constant across pages —
    shrinking it on the last page returns duplicates. Keep PAGE_SIZE fixed.
    """
    out = []
    page = 1
    while len(out) < want:
        params = {
            "query": query,
            "fields": FIELDS,
            "limit": PAGE_SIZE,
            "page": page,
            "sort": [{"id": "asc"}],  # deterministic pagination
        }
        d = get_json(session, API, params={"params": json.dumps(params)})
        data = d["data"]
        if not data:
            break
        out.extend(data)
        total = d["pagination"]["total"]
        if page * PAGE_SIZE >= total:
            break
        page += 1
        rate_sleep(0.1)
    return out[:want]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/aic")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    session = make_session()
    img_session = make_session(proxy="http://127.0.0.1:7897")  # Cloudflare on www.artic.edu

    n_impr = min(500, args.limit // 2)
    recs = fetch_slice(session, QUERY_IMPRESSIONISM, n_impr)
    print(f"[impressionism] wanted {n_impr}, got {len(recs)}")
    n_asian = args.limit - len(recs)
    recs_a = fetch_slice(session, QUERY_ASIAN, n_asian)
    print(f"[asian] wanted {n_asian}, got {len(recs_a)}")
    recs += recs_a

    rows = []
    n_ok = 0
    t0 = time.time()
    for i, a in enumerate(recs):
        sid = str(a["id"])
        img_id = a.get("image_id")
        url = IIIF.format(image_id=img_id) if img_id else ""
        dest = img_dir / f"{sid}.jpg"
        w, h = (None, None)
        ok = False
        if img_id:
            w, h = download_image(img_session, url, dest)
            ok = w is not None
            if ok:
                n_ok += 1
            else:
                bad = img_dir / f"{sid}.jpg"
                if bad.exists() and bad.stat().st_size < 2000:
                    bad.unlink()
        rows.append({
            "source": "aic",
            "source_id": sid,
            "title": a.get("title"),
            "artist": a.get("artist_title"),
            "date": a.get("date_display"),
            "classification": a.get("classification_title"),
            "department": a.get("department_title"),
            "license": "CC0" if a.get("is_public_domain") else "restricted",
            "image_url": url,
            "local_path": str(dest) if ok else "",
            "width": w,
            "height": h,
            "download_ok": ok,
            # extras kept for slice analysis
            "date_start": a.get("date_start"),
            "place_of_origin": a.get("place_of_origin"),
            "medium": a.get("medium_display"),
        })
        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"{i+1}/{len(recs)} ok={n_ok} rate={(i+1)/dt:.2f} img/s")
        rate_sleep(args.sleep)

    save_parquet(rows, out / "metadata.parquet")
    dt = time.time() - t0
    print(f"done: {len(rows)} rows, {n_ok} images, {dt:.0f}s "
          f"({len(rows)/max(dt,1):.2f} img/s incl. metadata)")


if __name__ == "__main__":
    main()
