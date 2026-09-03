"""Fetcher for the Princeton University Art Museum (PUAM) collection.

Metadata: weekly static dump, a zip with one JSON per object (58,887 objects):
    https://static.artmuseum.princeton.edu/collection-data-sets/objects.zip
Reuse a previously downloaded zip via --objects-zip (otherwise downloaded
to <out>/objects.zip, streamed to disk).

Object JSON fields used:
- objectnumber / displaytitle / department / classification / displaydate
- makers[] (displayname, role="Artist")
- primaryimage[] -> IIIF Image API v3 service base urls
- restrictions ("Restricted" and any non-empty value are skipped)
- nowebuse (string "True"/"False"; truthy = not web-usable, skipped)

Images: <primaryimage[0]>/full/<size>/0/default.jpg
(size "max" by default; "!1600,1600" caps the longest side).

License: public-domain works are openly usable -> "PD (Princeton AM image use policy)".

CLI:
    python -m src.dataset.fetchers.princeton --out /tmp/princeton_probe --limit 10
    python -m src.dataset.fetchers.princeton --out data/raw/princeton --objects-zip /tmp/puam_objects.zip --limit 200
"""

import argparse
import json
import os
import re
import time
import zipfile
from typing import Dict, Iterator, Optional

import requests

from .common import JsonlWriter, make_session, request_with_retry, save_jpeg

ZIP_URL = "https://static.artmuseum.princeton.edu/collection-data-sets/objects.zip"
LICENSE = "PD (Princeton AM image use policy)"
DEFAULT_CLASSIFICATIONS = ("Paintings", "Calligraphy")

# objectnumbers may contain spaces / other path-hostile chars ("1998-111 d")
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_objectnumber(objectnumber: str) -> str:
    """Make an objectnumber safe for use in image_ids / file names."""
    return UNSAFE_CHARS.sub("_", objectnumber).strip("._")


def download_objects_zip(zip_path: str, session=None) -> str:
    """Stream the objects.zip to disk if missing/empty; return its path."""
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        return zip_path
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    tmp = zip_path + ".part"
    session = session or make_session()
    print(f"downloading {ZIP_URL} -> {zip_path}")
    with session.get(ZIP_URL, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        got = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r{got / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(tmp, zip_path)
    return zip_path


def parse_object(obj: Dict) -> Dict:
    """Normalize one object JSON into a flat dict."""
    makers = obj.get("makers") or []
    artist = " | ".join(
        m.get("displayname", "") for m in makers
        if isinstance(m, dict) and m.get("role") == "Artist" and m.get("displayname")
    )
    primary = obj.get("primaryimage") or []
    if isinstance(primary, str):
        primary = [primary]
    return {
        "objectnumber": obj.get("objectnumber", ""),
        "title": obj.get("displaytitle", "") or (obj.get("titles") or [{}])[0].get("title", ""),
        "department": obj.get("department", ""),
        "classification": obj.get("classification", ""),
        "displaydate": obj.get("displaydate") or "",
        "artist": artist,
        "medium": obj.get("medium") or "",
        "primaryimage": primary[0] if primary else "",
        "restrictions": (obj.get("restrictions") or "").strip(),
        "nowebuse": obj.get("nowebuse"),
    }


def is_web_usable(parsed: Dict) -> bool:
    """Skip restricted and not-web-usable objects; require a primary image."""
    if parsed["restrictions"]:
        return False
    nb = parsed["nowebuse"]
    if isinstance(nb, str):
        if nb.strip().lower() in ("true", "1"):
            return False
    elif nb:  # booleans/ints: truthy means not web-usable
        return False
    return bool(parsed["primaryimage"])


def matches_filters(parsed: Dict, department: Optional[str],
                    classifications: Optional[tuple]) -> bool:
    if not is_web_usable(parsed):
        return False
    if department and parsed["department"] != department:
        return False
    if classifications and parsed["classification"] not in classifications:
        return False
    return True


def build_image_url(primaryimage: str, size: str = "max") -> str:
    """IIIF Image API v3 image URL; size 'max' or e.g. '!1600,1600'."""
    return f"{primaryimage}/full/{size}/0/default.jpg"


def iter_objects(zip_path: str) -> Iterator[Dict]:
    """Yield parsed objects from the zip (lazy, stops on caller break)."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                yield parse_object(json.loads(zf.read(name)))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue


def run(out_dir: str, limit: int, objects_zip: Optional[str] = None,
        department: str = "Asian Art", classifications: Optional[tuple] = None,
        image_size: str = "max", delay: float = 0.3) -> None:
    if objects_zip is None:
        objects_zip = download_objects_zip(f"{out_dir}/objects.zip")
    if not os.path.exists(objects_zip):
        raise FileNotFoundError(objects_zip)
    session = make_session()
    writer = JsonlWriter(f"{out_dir}/metadata.jsonl")
    classifications = tuple(classifications) if classifications else DEFAULT_CLASSIFICATIONS
    print(f"zip: {objects_zip} (dept={department!r}, classes={classifications})")
    n_img = n_obj = n_skipped = 0
    for parsed in iter_objects(objects_zip):
        if n_obj >= limit:
            break
        if not matches_filters(parsed, department, classifications):
            n_skipped += 1
            continue
        n_obj += 1
        image_id = f"princeton-{sanitize_objectnumber(parsed['objectnumber'])}"
        if image_id in writer.seen:
            continue
        url = build_image_url(parsed["primaryimage"], image_size)
        resp = request_with_retry(session, "GET", url)
        if resp is None or len(resp.content) < 1024 or not resp.content.startswith(b"\xff\xd8\xff"):
            print(f"[skip] {image_id}: image fetch failed")
            n_skipped += 1
            continue
        path = f"{out_dir}/images/{image_id}.jpg"
        save_jpeg(resp.content, path)
        writer.write({
            "image_id": image_id,
            "source": "princeton",
            "object_id": parsed["objectnumber"],
            "title": parsed["title"],
            "artist": parsed["artist"],
            "date": parsed["displaydate"],
            "medium": parsed["medium"],
            "classification": parsed["classification"],
            "department": parsed["department"],
            "object_url": f"https://artmuseum.princeton.edu/collections/objects/{parsed['objectnumber']}",
            "image_url": url,
            "license": LICENSE,
            "local_path": path,
        })
        n_img += 1
        time.sleep(delay)
        if n_img % 25 == 0:
            print(f"[progress] images={n_img}")
    print(f"done: {n_img} images from {n_obj} objects, {n_skipped} skipped -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Princeton University Art Museum images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20, help="max matching objects per run (idempotent)")
    ap.add_argument("--objects-zip", default=None, help="reuse a downloaded objects.zip")
    ap.add_argument("--department", default="Asian Art", help="empty = all departments")
    ap.add_argument("--classification", action="append", default=None,
                    help="accepted classification (repeatable); default: Paintings, Calligraphy")
    ap.add_argument("--image-size", default="max", help="IIIF size, e.g. 'max' or '!1600,1600'")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    run(args.out, args.limit, args.objects_zip, args.department,
        tuple(args.classification) if args.classification else None,
        args.image_size, args.delay)


if __name__ == "__main__":
    main()
