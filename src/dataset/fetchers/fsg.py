"""Fetcher for the Smithsonian National Museum of Asian Art (Freer|Sackler, FSG).

Metadata: Smithsonian Open Access bulk dumps (EDAN JSONL), no API key required.
- index: https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/fsg/index.txt
- each listed file is JSONL of EDAN records (~4608 records / ~26MB total).

Record shape (EDAN):
- content.descriptiveNonRepeating.title.content        -> title
- content.descriptiveNonRepeating.record_link          -> object page URL
- content.descriptiveNonRepeating.online_media.media[] -> media items, each with
  resources[] (per-size url/width/height) and usage.access (e.g. "CC0")
- content.freetext -> labelled value pairs (date, name/Artist, topic, objectType,
  physicalDescription/Medium, identifier, creditLine, ...)
- content.indexedStructured -> culture[], object_type[], ...

We keep records whose freetext topic contains --topic (default "Chinese"),
optionally narrowed by --object-type (e.g. "Painting"), and only media with
usage.access == "CC0". The image is the largest resource of each media item.

License: record-level usage.access=CC0 -> "CC0 (Smithsonian Open Access)".

CLI:
    python -m src.dataset.fetchers.fsg --out data/raw/fsg --limit 20
    python -m src.dataset.fetchers.fsg --out data/raw/fsg --topic Chinese --object-type Painting --limit 200
"""

import argparse
import json
import time
from typing import Dict, Iterator, List, Optional

from .common import JsonlWriter, make_session, request_with_retry, save_jpeg

INDEX_URL = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/fsg/index.txt"
LICENSE = "CC0 (Smithsonian Open Access)"


def fetch_index(session) -> List[str]:
    """Return the list of JSONL batch URLs from the index file."""
    resp = request_with_retry(session, "GET", INDEX_URL)
    if resp is None:
        raise RuntimeError("FSG index fetch failed after retries")
    urls = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    if not urls:
        raise RuntimeError("FSG index is empty")
    return urls


def freetext_values(freetext: Dict, key: str, label: Optional[str] = None) -> List[str]:
    """Contents of freetext entries under `key`, optionally filtered by label."""
    return [
        item.get("content", "")
        for item in freetext.get(key, [])
        if isinstance(item, dict) and (label is None or item.get("label") == label)
    ]


def parse_record(record: Dict) -> Dict:
    """Normalize one EDAN record into a flat dict (raw strings, no filtering)."""
    content = record.get("content", {})
    dnr = content.get("descriptiveNonRepeating", {})
    ft = content.get("freetext", {})
    ist = content.get("indexedStructured", {})
    media = (dnr.get("online_media", {}).get("media")) or []
    return {
        "record_id": dnr.get("record_ID", ""),
        "title": dnr.get("title", {}).get("content", ""),
        "object_url": dnr.get("record_link", ""),
        "artist": " | ".join(freetext_values(ft, "name", "Artist")),
        "date": " | ".join(freetext_values(ft, "date", "Date")),
        "period": " | ".join(freetext_values(ft, "date", "Period")),
        "culture": " | ".join(ist.get("culture") or []),
        "medium": " | ".join(freetext_values(ft, "physicalDescription", "Medium")),
        "classification": " | ".join(ist.get("object_type") or []),
        "object_type": " | ".join(freetext_values(ft, "objectType")),
        "topics": freetext_values(ft, "topic"),
        "accession": " | ".join(freetext_values(ft, "identifier")),
        "credit_line": " | ".join(freetext_values(ft, "creditLine")),
        "dimensions": " | ".join(freetext_values(ft, "physicalDescription", "Dimensions")),
        "media": media,
    }


def topic_matches(parsed: Dict, topic: str) -> bool:
    """Case-insensitive substring match of --topic against freetext topics."""
    return any(topic.lower() in t.lower() for t in parsed["topics"])


def object_type_matches(parsed: Dict, object_type: str) -> bool:
    """Case-insensitive substring match of --object-type against freetext objectType."""
    return object_type.lower() in parsed["object_type"].lower()


def _is_jpeg_url(url: str) -> bool:
    return url.lower().endswith((".jpg", ".jpeg"))


def pick_largest_resource(media: Dict) -> Optional[Dict]:
    """Largest resource (width*height) with known size; prefers JPEG urls (the
    "High-resolution JPEG" variant) over TIFF, falls back to the first URL."""
    sized = [r for r in media.get("resources") or []
             if r.get("url") and r.get("width") and r.get("height")]
    for bucket in (lambda rs: [r for r in rs if _is_jpeg_url(r["url"])], lambda rs: rs):
        jpg = bucket(sized)
        if jpg:
            return max(jpg, key=lambda r: int(r["width"]) * int(r["height"]))
    for r in media.get("resources") or []:
        if r.get("url"):
            return r
    return None


def cc0_media(media: Dict) -> bool:
    """Only CC0 media are kept (record-level Smithsonian Open Access)."""
    usage = media.get("usage") or {}
    return usage.get("access") == "CC0"


def iter_records(session, batch_urls: List[str], topic: str,
                 object_type: Optional[str]) -> Iterator[Dict]:
    """Yield parsed records matching the topic/object-type filters."""
    for url in batch_urls:
        resp = request_with_retry(session, "GET", url)
        if resp is None:
            print(f"[warn] batch fetch failed: {url}")
            continue
        for line in resp.content.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = parse_record(record)
            if topic and not topic_matches(parsed, topic):
                continue
            if object_type and not object_type_matches(parsed, object_type):
                continue
            yield parsed


def run(out_dir: str, limit: int, topic: str = "Chinese",
        object_type: Optional[str] = None, delay: float = 0.3) -> None:
    session = make_session()
    writer = JsonlWriter(f"{out_dir}/metadata.jsonl")
    batch_urls = fetch_index(session)
    print(f"index: {len(batch_urls)} batches (topic={topic!r}, object_type={object_type!r})")
    n_img = n_obj = n_media_skipped = 0
    for parsed in iter_records(session, batch_urls, topic, object_type):
        if n_obj >= limit:
            break
        n_obj += 1
        for media in parsed["media"]:
            if media.get("type") != "Images":
                continue
            if not cc0_media(media):
                n_media_skipped += 1
                continue
            res = pick_largest_resource(media)
            if not res:
                continue
            media_id = media.get("idsId") or res.get("url", "").split("?id=")[-1]
            image_id = f"fsg-{parsed['record_id']}-{media_id}"
            if image_id in writer.seen:
                continue
            url = res["url"]
            resp = request_with_retry(session, "GET", url)
            if resp is None or len(resp.content) < 1024 or not resp.content.startswith(b"\xff\xd8\xff"):
                print(f"[skip] {image_id}: image fetch failed")
                continue
            path = f"{out_dir}/images/{image_id}.jpg"
            save_jpeg(resp.content, path)
            writer.write({
                "image_id": image_id,
                "source": "fsg",
                "object_id": parsed["record_id"],
                "title": parsed["title"],
                "artist": parsed["artist"],
                "date": parsed["date"],
                "period": parsed["period"],
                "culture": parsed["culture"],
                "medium": parsed["medium"],
                "classification": parsed["classification"],
                "object_type": parsed["object_type"],
                "topics": parsed["topics"],
                "accession": parsed["accession"],
                "credit_line": parsed["credit_line"],
                "dimensions": parsed["dimensions"],
                "object_url": parsed["object_url"],
                "image_url": url,
                "width": res.get("width"),
                "height": res.get("height"),
                "license": LICENSE,
                "local_path": path,
            })
            n_img += 1
            time.sleep(delay)
        print(f"[ok] {parsed['record_id']} {parsed['title'][:40]!r}")
    print(f"done: {n_img} images from {n_obj} records, "
          f"{n_media_skipped} non-CC0 media skipped -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Smithsonian NMAA (FSG) open-access images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20, help="max matching records per run (idempotent)")
    ap.add_argument("--topic", default="Chinese", help="freetext topic substring filter (empty = all)")
    ap.add_argument("--object-type", default=None, help="freetext objectType substring filter (e.g. Painting)")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    run(args.out, args.limit, args.topic, args.object_type, args.delay)


if __name__ == "__main__":
    main()
