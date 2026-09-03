"""Rijksmuseum fetcher via the keyless data.rijksmuseum.nl Linked Art APIs.

Flow (3 JSON resolutions per object):
1. GET https://data.rijksmuseum.nl/search/collection?type=<type>&imageAvailable=true
   (100 ids/page, pageToken pagination)
2. GET <object id>  (Accept: application/ld+json) -> metadata + shows[0] VisualItem id
3. GET <visual item id> -> digitally_shown_by[0] DigitalObject id
4. GET <digital object id> -> access_point[0].id = IIIF image URL (iiif.micr.io)

CLI:
    python -m src.dataset.fetchers.rijksmuseum --out data/raw/rijksmuseum \
        --type painting --limit 5000 --delay 0.3
"""

import argparse
import os
import time
from typing import Dict, Optional

from src.dataset.fetchers.common import (JsonlWriter, make_session,
                                         request_with_retry, save_jpeg)

SEARCH = "https://data.rijksmuseum.nl/search/collection"
ACCEPT_LD = {"Accept": "application/ld+json"}
LICENSE_PD = "PD (Rijksmuseum open data)"


def get_json(session, url: str) -> Optional[Dict]:
    resp = request_with_retry(session, "GET", url, headers=ACCEPT_LD)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def search_object_ids(session, obj_type: str, creator: Optional[str],
                      limit: int, delay: float) -> list:
    """Page the search API; return Linked Art object ids."""
    params = f"type={obj_type}&imageAvailable=true"
    if creator:
        params += f"&creator={creator}"
    url = f"{SEARCH}?{params}"
    ids = []
    while url and (not limit or len(ids) < limit):
        data = get_json(session, url)
        if data is None:
            raise RuntimeError("search page failed after retries")
        ids.extend(it["id"] for it in data.get("orderedItems", []))
        url = (data.get("next") or {}).get("id")
        time.sleep(delay)
    print(f"search: {len(ids)} object ids (type={obj_type}, creator={creator!r})")
    return ids[:limit] if limit else ids


def _first_name(items: list) -> str:
    for it in items or []:
        if it.get("type") == "Name" and it.get("content"):
            return it["content"]
    return ""


def _first_identifier(items: list) -> str:
    for it in items or []:
        if it.get("type") == "Identifier" and it.get("content"):
            return it["content"]
    return ""


def _person_name(person: Dict) -> str:
    return _first_name(person.get("identified_by"))


def _as_dict(x) -> Dict:
    """Some Linked Art fields alternate between dict and list-of-dict."""
    if isinstance(x, list):
        return x[0] if x else {}
    return x or {}


def parse_object(obj: Dict) -> Dict:
    produced = _as_dict(obj.get("produced_by"))
    artists = [_person_name(_as_dict(p)) for p in produced.get("carried_out_by") or []]
    timespan = _as_dict(produced.get("timespan"))
    return {
        "title": _first_name(obj.get("identified_by")),
        "object_number": _first_identifier(obj.get("identified_by")),
        "artist": "; ".join(a for a in artists if a),
        "date": _first_name(timespan.get("identified_by")),
    }


def resolve_image_url(session, obj: Dict, delay: float) -> Optional[str]:
    """Object -> VisualItem -> DigitalObject -> IIIF access point."""
    shows = obj.get("shows") or []
    if isinstance(shows, dict):
        shows = [shows]
    if not shows:
        return None
    time.sleep(delay)
    vi = get_json(session, shows[0]["id"])
    if not vi:
        return None
    dsb = vi.get("digitally_shown_by") or []
    if not dsb:
        return None
    time.sleep(delay)
    do = get_json(session, dsb[0]["id"])
    if not do:
        return None
    ap = do.get("access_point") or []
    return ap[0]["id"] if ap else None


def run(out_dir: str, obj_type: str, creator: Optional[str], limit: int,
        image_width: int, delay: float) -> None:
    session = make_session()
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    writer = JsonlWriter(os.path.join(out_dir, "metadata.jsonl"))
    ids = search_object_ids(session, obj_type, creator, limit, delay)
    n_img = 0
    for n_obj, oid_url in enumerate(ids, 1):
        lod_id = oid_url.rstrip("/").rsplit("/", 1)[-1]
        image_id = f"rkm-{lod_id}"
        if image_id in writer.seen:
            continue
        obj = get_json(session, oid_url)
        if obj is None:
            print(f"[skip] {image_id}: object JSON failed")
            continue
        image_url = resolve_image_url(session, obj, delay)
        if not image_url:
            print(f"[skip] {image_id}: no image access point")
            continue
        # IIIF: bound width if the URL follows the /full/max/ pattern
        if "/full/max/" in image_url and image_width:
            image_url = image_url.replace("/full/max/", f"/full/{image_width},/")
        time.sleep(delay)
        resp = request_with_retry(session, "GET", image_url)
        if resp is None or not resp.content.startswith(b"\xff\xd8\xff"):
            print(f"[skip] {image_id}: image fetch failed")
            continue
        path = os.path.join(out_dir, "images", image_id + ".jpg")
        save_jpeg(resp.content, path)
        meta = parse_object(obj)
        writer.write({
            "image_id": image_id,
            "source": "rijksmuseum",
            "object_id": lod_id,
            "object_number": meta["object_number"],
            "title": meta["title"],
            "artist": meta["artist"],
            "date": meta["date"],
            "object_url": oid_url,
            "image_url": image_url,
            "license": LICENSE_PD,
            "local_path": path,
        })
        n_img += 1
        if n_obj % 100 == 0:
            print(f"[progress] objects={n_obj} images={n_img}", flush=True)
    print(f"done: {n_img} images from {len(ids)} objects -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Rijksmuseum collection images (keyless)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="painting")
    ap.add_argument("--creator", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--image-width", type=int, default=843)
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    run(args.out, args.type, args.creator, args.limit, args.image_width, args.delay)


if __name__ == "__main__":
    main()
