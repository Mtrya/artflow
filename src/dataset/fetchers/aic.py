"""Fetcher for the Art Institute of Chicago (AIC) public API (CC0).

API: https://api.artic.edu/api/v1/
- POST /artworks/search with an Elasticsearch DSL body (the top-level `q`
  parameter is ignored both in GET and POST; use a `query_string` clause).
- Images via IIIF 2.0: https://www.artic.edu/iiif/2/<image_id>/full/<w>,/0/default.jpg

License: artwork data is CC0 (only the `description` field is CC-BY, which we
do not fetch); see https://api.artic.edu/docs/. Rate limit: 60 req/min
anonymously, so the default delay is 1s. Search pagination caps at 10,000.

CLI:
    python -m src.dataset.fetchers.aic --out data/raw/aic --limit 10
    python -m src.dataset.fetchers.aic --out data/raw/aic --query impressionism --limit 50
"""

import argparse
import time
from typing import Dict, Iterator, Optional

from .common import JsonlWriter, make_session, request_with_retry, save_jpeg

API = "https://api.artic.edu/api/v1"
IIIF_BASE = "https://www.artic.edu/iiif/2"
LICENSE = "CC0 (AIC public domain)"

FIELDS = ["id", "title", "artist_display", "date_display", "medium_display",
          "classification_title", "style_title", "image_id", "is_public_domain"]


def build_query(year_begin: int, year_end: int, artwork_type: Optional[str],
                query: Optional[str], use_match: bool = False) -> Dict:
    """Build the ES DSL query for the search endpoint.

    `term` is preferred on keyword/text fields; pass use_match=True as a
    fallback when the API rejects the term (unknown/analyzed field).
    """
    filters = [
        {"term": {"is_public_domain": True}},
        {"range": {"date_start": {"gte": year_begin, "lte": year_end}}},
    ]
    if artwork_type:
        field = "artwork_type_title"
        clause = {"term": {f"{field}.keyword": artwork_type}}
        if use_match:
            clause = {"match": {field: artwork_type}}
        filters.append(clause)
    bool_q = {"filter": filters}
    if query:
        bool_q["must"] = [{"query_string": {"query": query, "default_operator": "AND"}}]
    return {"bool": bool_q}


def build_image_url(image_id: str, width: int = 843) -> str:
    """IIIF image URL; 843 is the official recommended cache size, 1686 for PD."""
    return f"{IIIF_BASE}/{image_id}/full/{width},/0/default.jpg"


def item_to_record(item: Dict, width: int = 843) -> Dict:
    """Map one search result item to a metadata record (local_path added later)."""
    oid = str(item["id"])
    record = {
        "image_id": f"aic-{oid}",
        "source": "aic",
        "object_id": oid,
        "title": item.get("title", ""),
        "artist": item.get("artist_display", ""),
        "date": item.get("date_display", ""),
        "medium": item.get("medium_display", ""),
        "classification": item.get("classification_title", ""),
        "is_public_domain": bool(item.get("is_public_domain")),
        "object_url": f"https://www.artic.edu/artworks/{oid}",
        "iiif_url": build_image_url(item["image_id"], width),
        "license": LICENSE,
    }
    style = item.get("style_title") or ""
    if style:
        record["style_title"] = style
        record["period"] = style
    return record


def iter_search_results(session, query: Dict, max_items: Optional[int],
                        delay: float, page_size: int = 10) -> Iterator[Dict]:
    """Yield artwork items by paging the search endpoint (idempotent-safe)."""
    page, yielded = 1, 0
    total_pages = None
    while True:
        body = {"query": query, "fields": FIELDS, "limit": page_size, "page": page}
        resp = request_with_retry(session, "POST", f"{API}/artworks/search", json=body)
        if resp is None:
            raise RuntimeError(f"search page {page} failed after retries")
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"search page {page}: invalid JSON response")
        if "error" in data:
            raise RuntimeError(f"search page {page}: API error: {data['error']}")
        items = data.get("data") or []
        if not items:
            break
        for item in items:
            yield item
            yielded += 1
            if max_items is not None and yielded >= max_items:
                return
        total_pages = (data.get("pagination") or {}).get("total_pages") or 0
        if page >= total_pages:
            break
        page += 1
        time.sleep(delay)


def process_item(session, writer: JsonlWriter, item: Dict, out_dir: str,
                 image_width: int, delay: float) -> bool:
    """Fetch and save one item; returns True if an image was written."""
    if not item.get("image_id"):
        print(f"[skip] aic-{item['id']}: no image_id")
        return False
    image_id = f"aic-{item['id']}"
    if image_id in writer.seen:
        return False
    record = item_to_record(item, image_width)
    resp = request_with_retry(session, "GET", record["iiif_url"])
    if resp is None or len(resp.content) < 1024 or not resp.content.startswith(b"\xff\xd8\xff"):
        print(f"[skip] {image_id}: image fetch failed")
        return False
    path = f"{out_dir}/images/{image_id}.jpg"
    save_jpeg(resp.content, path)
    record["local_path"] = path
    writer.write(record)
    time.sleep(delay)
    return True


def run(out_dir: str, limit: int, year_begin: int, year_end: int,
        artwork_type: Optional[str], query: Optional[str], image_width: int = 843,
        delay: float = 1.0) -> None:
    session = make_session()
    writer = JsonlWriter(f"{out_dir}/metadata.jsonl")
    q = build_query(year_begin, year_end, artwork_type, query)
    try:
        results = iter_search_results(session, q, limit, delay)
    except RuntimeError as e:
        # term rejected on an analyzed/unknown field -> retry with match clauses
        print(f"[warn] {e}; retrying with match on artwork_type...")
        q = build_query(year_begin, year_end, artwork_type, query, use_match=True)
        results = iter_search_results(session, q, limit, delay)
    n_img = 0
    for item in results:
        n_img += process_item(session, writer, item, out_dir, image_width, delay)
    print(f"done: {n_img} images -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Art Institute of Chicago public domain artworks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10, help="max artworks (not images)")
    ap.add_argument("--year-begin", type=int, default=1860)
    ap.add_argument("--year-end", type=int, default=1910)
    ap.add_argument("--artwork-type", default="Painting", help="None to disable the filter")
    ap.add_argument("--query", default=None, help="optional keyword, e.g. impressionism")
    ap.add_argument("--image-width", type=int, default=843, help="IIIF width; 1686 for PD originals")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (60 req/min cap)")
    args = ap.parse_args()
    run(args.out, args.limit, args.year_begin, args.year_end,
        args.artwork_type or None, args.query, args.image_width, args.delay)


if __name__ == "__main__":
    main()
