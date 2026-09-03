"""Fetcher for the National Palace Museum, Taipei Open Data (國立故宮博物院 Open Data).

Flow per item (藏品):
1. POST /opendata/Pub/Search (JSON body) -> HTML result page with detail ids.
2. GET  /opendata/Pub/Detail/<cid>?dep=P&mode=full -> metadata tables (HTML).
3. GET  /opendata/Integrate/GetJson?cid=<cid>&dept=P&imageName= -> IIIF manifest
   (no captcha), listing canvases (views) with IIIF image service ids.
4. GET  https://iiifod.npm.gov.tw/iiif/2/<service_id>/full/<size>/0/default.jpg -> image.

License: 1MP tier CC0, 6MP tier CC BY 4.0 (attribution required), see
https://theme.npm.edu.tw/opendata/ (recorded per-item as license field).

CLI:
    python -m src.dataset.fetchers.npm_tw --out data/raw/npm_tw --limit 20
"""

import argparse
import html
import itertools
import re
import time
from typing import Dict, Iterator, List, Optional

from .common import JsonlWriter, make_session, request_with_retry, save_jpeg

BASE = "https://digitalarchive.npm.gov.tw"
IIIF_BASE = "https://iiifod.npm.gov.tw/iiif/2"
LICENSE = "CC0 (1MP) / CC-BY-4.0 (6MP), NPM Taipei Open Data"

DETAIL_ID_RE = re.compile(r"/opendata/Pub/Detail/(\d+)\?dep=P")
META_ROW_RE = re.compile(r"<tr>\s*<td>([^<]+)</td>\s*<td>(.*?)</td>\s*</tr>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
SIZE_ROW_RE = re.compile(r"<tr>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*</tr>", re.S)


def parse_search_ids(html_text: str) -> List[str]:
    """Extract unique detail ids (order preserved) from a Search result page."""
    seen, ids = set(), []
    for m in DETAIL_ID_RE.finditer(html_text):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))
    return ids


def parse_detail_metadata(html_text: str) -> Dict[str, str]:
    """Parse the first metadata table of a Detail page into a flat dict."""
    meta = {}
    for key, value in META_ROW_RE.findall(html_text):
        key = html.unescape(key).strip()
        # split bilingual values (品名 has zh <br/> en); keep raw parts
        parts = [html.unescape(TAG_RE.sub("", p)).strip() for p in re.split(r"<br\s*/?>", value)]
        parts = [p for p in parts if p]
        meta[key] = " | ".join(parts)
    return meta


def iter_detail_ids(session, register_type: str = "繪畫", page_size: int = 15,
                    max_items: Optional[int] = None, delay: float = 0.5) -> Iterator[str]:
    """Yield detail ids by paging the Search endpoint."""
    page, yielded = 1, 0
    while True:
        body = {
            "RegisterType": register_type, "IndexYear": None,
            "WestBeginYear": 0, "WestEndYear": 0, "YearDisplay": None,
            "SearchContent": None, "RegisterTypeEng": None,
            "PageInfo": {"PageIndex": page, "PageSize": page_size, "PageCount": 7244},
        }
        resp = request_with_retry(session, "POST", f"{BASE}/opendata/Pub/Search", json=body)
        if resp is None:
            raise RuntimeError(f"Search page {page} failed after retries")
        ids = parse_search_ids(resp.text)
        if not ids:
            break
        for cid in ids:
            yield cid
            yielded += 1
            if max_items is not None and yielded >= max_items:
                return
        page += 1
        time.sleep(delay)


def fetch_manifest(session, cid: str, attempts: int = 2) -> Optional[Dict]:
    """Fetch the IIIF manifest JSON for one object.

    Broken manifests (empty/non-JSON 200 responses) are usually deterministic
    server-side failures, so one retry is enough; the Detail-page fallback in
    run() recovers the images for these objects.
    """
    url = f"{BASE}/opendata/Integrate/GetJson?cid={cid}&dept=P&imageName="
    for attempt in range(attempts):
        resp = request_with_retry(session, "GET", url)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            # json.JSONDecodeError and simplejson's (used by requests when
            # installed) both subclass ValueError.
            time.sleep(session._backoff * (attempt + 1))
    print(f"[skip] cid={cid}: manifest not JSON after {attempts} attempts")
    return None


def fetch_detail_html(session, cid: str) -> str:
    """Fetch the Detail page HTML ('' on failure)."""
    resp = request_with_retry(session, "GET", f"{BASE}/opendata/Pub/Detail/{cid}?dep=P&mode=full")
    return resp.text if resp is not None else ""


def fetch_detail_metadata(session, cid: str) -> Dict[str, str]:
    """Fetch and parse the Detail page metadata tables."""
    return parse_detail_metadata(fetch_detail_html(session, cid))


def extract_image_codes(html_text: str) -> List[str]:
    """Extract IIIF image codes from data-image-name attrs on the Detail page.

    Fallback for objects whose server-side manifest generation is broken
    (GetJson returns empty, IIIFViewer 500s) although the images themselves
    exist under https://iiifod.npm.gov.tw/iiif/2/K2A%2F<code>.
    """
    codes: List[str] = []
    for code in re.findall(r'data-image-name="([^"]+)"', html_text):
        if code and code not in codes:
            codes.append(code)
    return codes


def iter_canvases(manifest: Dict) -> Iterator[Dict]:
    """Yield per-canvas dicts: label, width, height, iiif_service."""
    for seq in manifest.get("sequences", []):
        for canvas in seq.get("canvases", []):
            images = canvas.get("images") or []
            service = ((images[0].get("resource") or {}).get("service") or {}) if images else {}
            service_id = service.get("@id")
            if not service_id:
                continue
            yield {
                "label": canvas.get("label", ""),
                "width": canvas.get("width"),
                "height": canvas.get("height"),
                "iiif_service": service_id,
            }


def build_image_url(iiif_service: str, size: str = "full") -> str:
    """IIIF image URL; size 'full' for native res or ',1600' to cap width at 1600px."""
    return f"{iiif_service}/full/{size}/0/default.jpg"


def run(out_dir: str, limit: int, register_type: str = "繪畫", image_size: str = "full",
        delay: float = 0.5, offset: int = 0) -> None:
    session = make_session()
    writer = JsonlWriter(f"{out_dir}/metadata.jsonl")
    n_img = 0
    ids = iter_detail_ids(session, register_type=register_type, max_items=None, delay=delay)
    if offset:
        ids = itertools.islice(ids, offset, None)
    if limit is not None:
        ids = itertools.islice(ids, limit)
    for cid in ids:
        manifest = fetch_manifest(session, cid)
        detail_html = fetch_detail_html(session, cid)
        meta = parse_detail_metadata(detail_html)
        if manifest is None:
            # Server-side manifest generation is broken for some objects
            # (GetJson empty, IIIFViewer 500) although the images exist;
            # recover the canvas list from the Detail page's image codes.
            codes = extract_image_codes(detail_html)
            if not codes:
                print(f"[skip] cid={cid}: no manifest")
                continue
            canvases = [{"label": c, "width": None, "height": None,
                         "iiif_service": f"{IIIF_BASE}/K2A%2F{c}"} for c in codes]
        else:
            canvases = iter_canvases(manifest)
        for canvas in canvases:
            image_id = f"npm_tw-{cid}-{canvas['label']}"
            if image_id in writer.seen:
                continue
            url = build_image_url(canvas["iiif_service"], image_size)
            resp = request_with_retry(session, "GET", url)
            if resp is None or len(resp.content) < 1024 or not resp.content.startswith(b"\xff\xd8\xff"):
                print(f"[skip] {image_id}: image fetch failed")
                continue
            path = f"{out_dir}/images/{image_id}.jpg"
            save_jpeg(resp.content, path)
            writer.write({
                "image_id": image_id,
                "source": "npm_tw",
                "object_id": cid,
                "canvas_label": canvas["label"],
                "title": meta.get("品名", (manifest or {}).get("label", "")),
                "artist": meta.get("作者", ""),
                "category": meta.get("分類", ""),
                "object_no": meta.get("文物統一編號", ""),
                "extra_meta": {k: v for k, v in meta.items() if k not in ("品名", "作者", "分類", "文物統一編號")},
                "object_url": f"{BASE}/opendata/Pub/Detail/{cid}?dep=P&mode=full",
                "iiif_url": url,
                "width": canvas["width"],
                "height": canvas["height"],
                "license": LICENSE,
                "local_path": path,
            })
            n_img += 1
            time.sleep(delay)
        print(f"[ok] cid={cid} ({meta.get('品名', '')[:30]})")
    print(f"done: {n_img} images -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch NPM Taipei Open Data paintings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20, help="max objects (藏品), not images")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N objects (for sharding)")
    ap.add_argument("--register-type", default="繪畫")
    ap.add_argument("--image-size", default="full", help="'full' or e.g. ',1600'")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    run(args.out, args.limit, args.register_type, args.image_size, args.delay, args.offset)


if __name__ == "__main__":
    main()
