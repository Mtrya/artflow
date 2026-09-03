"""Fetcher for the National Gallery of Art, Washington (NGA) Open Data (CC0).

Data: https://github.com/NationalGalleryOfArt/opendata
- data/objects.csv (~82MB): object metadata (title, attribution, years,
  medium, classification); object id column `objectid`.
- data/published_images.csv (~89MB): IIIF image uuids joined to objects via
  `depictstmsobjectid`; only rows with `openaccess=1` are taken.

Images via IIIF: https://api.nga.gov/iiif/<uuid>/full/<w>,/0/default.jpg

The CSVs are large; pass --data-dir to reuse previously downloaded copies
(download is resumable via .part files). pandas is used to load only the
columns we need and join the two tables.

Note: the collection mixes in many Prints/Photographs/"Index of American
Design" objects — the classification filter (default "Painting") is essential.

CLI:
    python -m src.dataset.fetchers.nga --out data/raw/nga --limit 10
    python -m src.dataset.fetchers.nga --out data/raw/nga --limit 50 \
        --year-begin 1850 --year-end 1930 --attribution Monet --data-dir /tmp/nga_data
"""

import argparse
import os
import time
from typing import Dict, Iterator, Optional

import pandas as pd

from .common import JsonlWriter, make_session, request_with_retry, save_jpeg

OBJECTS_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/master/data/objects.csv"
IMAGES_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/master/data/published_images.csv"
IIIF_BASE = "https://api.nga.gov/iiif"
LICENSE = "CC0 (NGA open data / PD images)"

OBJECT_COLS = ["objectid", "title", "attribution", "displaydate",
               "beginyear", "endyear", "medium", "classification"]
IMAGE_COLS = ["uuid", "depictstmsobjectid", "openaccess", "width", "height"]

TIMEOUT = (30, 1800)


def build_image_url(uuid: str, width: int = 843) -> str:
    return f"{IIIF_BASE}/{uuid}/full/{width},/0/default.jpg"


def object_page_url(objectid) -> str:
    return f"https://www.nga.gov/collection/art-object-page.{objectid}.html"


def ensure_csv(session, url: str, path: str, label: str) -> str:
    """Download url to path, resuming partial downloads via a .part file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[csv] {label} already present, reusing ({os.path.getsize(path)} bytes)")
        return path
    tmp = path + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    for attempt in range(session._max_retries):
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                if resp.status_code == 206:
                    mode = "ab"
                elif resp.status_code == 200:
                    mode, done = "wb", 0  # server ignored Range; restart
                else:
                    raise RuntimeError(f"{label}: HTTP {resp.status_code}")
                with open(tmp, mode) as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
                os.replace(tmp, path)
                print(f"[csv] {label} downloaded ({os.path.getsize(path)} bytes)")
                return path
        except (RuntimeError, Exception) as e:
            if attempt == session._max_retries - 1:
                raise RuntimeError(f"download {label} failed: {e}")
            time.sleep(session._backoff * (attempt + 1))
    raise RuntimeError(f"download {label} failed after retries")


def load_tables(data_dir: str, session) -> tuple:
    """Load (objects, published_images) DataFrames, downloading CSVs if needed."""
    obj_path = os.path.join(data_dir, "objects.csv")
    img_path = os.path.join(data_dir, "published_images.csv")
    ensure_csv(session, OBJECTS_URL, obj_path, "objects.csv")
    ensure_csv(session, IMAGES_URL, img_path, "published_images.csv")
    objects = pd.read_csv(obj_path, usecols=OBJECT_COLS)
    images = pd.read_csv(img_path, usecols=IMAGE_COLS)
    return objects, images


def filter_rows(df: pd.DataFrame, classification: Optional[str] = "Painting",
                year_begin: Optional[int] = None, year_end: Optional[int] = None,
                attribution: Optional[str] = None) -> pd.DataFrame:
    """Keep only open-access rows passing the classification/year/attribution filters."""
    df = df[df["openaccess"].astype(str) == "1"]
    if classification:
        df = df[df["classification"] == classification]
    if year_begin is not None:
        df = df[df["beginyear"].fillna(0) >= year_begin]
    if year_end is not None:
        df = df[df["endyear"].fillna(0) <= year_end]
    if attribution:
        df = df[df["attribution"].fillna("").str.contains(attribution, case=False, na=False)]
    return df


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def row_to_record(row: Dict, width: int = 843) -> Dict:
    """Map one joined row to a metadata record (local_path added later)."""
    oid = int(row["objectid"])
    record = {
        "image_id": f"nga-{row['uuid']}",
        "source": "nga",
        "object_id": str(oid),
        "title": row.get("title") or "",
        "artist": row.get("attribution") or "",
        "date": row.get("displaydate") or "",
        "medium": row.get("medium") or "",
        "classification": row.get("classification") or "",
        "object_url": object_page_url(oid),
        "iiif_url": build_image_url(row["uuid"], width),
        "license": LICENSE,
    }
    for key in ("beginyear", "endyear", "width", "height"):
        v = _to_int(row.get(key))
        if v is not None:
            record[key] = v
    return record


def iter_rows(df: pd.DataFrame) -> Iterator[Dict]:
    for _, row in df.iterrows():
        yield row.to_dict()


def run(out_dir: str, limit: int, data_dir: Optional[str], classification: Optional[str],
        year_begin: Optional[int], year_end: Optional[int], attribution: Optional[str],
        image_width: int = 843, delay: float = 0.5) -> None:
    data_dir = data_dir or out_dir
    session = make_session()
    writer = JsonlWriter(f"{out_dir}/metadata.jsonl")
    objects, images = load_tables(data_dir, session)
    merged = objects.merge(images, left_on="objectid", right_on="depictstmsobjectid", how="inner")
    del objects, images
    kept = filter_rows(merged, classification, year_begin, year_end, attribution)
    print(f"[filter] {len(merged)} joined rows -> {len(kept)} kept "
          f"(class={classification!r} years={year_begin}-{year_end} attribution={attribution!r})")
    n_img = 0
    for row in iter_rows(kept):
        if n_img >= limit:
            break
        image_id = f"nga-{row['uuid']}"
        if image_id in writer.seen:
            continue
        record = row_to_record(row, image_width)
        resp = request_with_retry(session, "GET", record["iiif_url"])
        if resp is None or len(resp.content) < 1024 or not resp.content.startswith(b"\xff\xd8\xff"):
            print(f"[skip] {image_id}: image fetch failed")
            continue
        path = f"{out_dir}/images/{image_id}.jpg"
        save_jpeg(resp.content, path)
        record["local_path"] = path
        writer.write(record)
        n_img += 1
        time.sleep(delay)
        if n_img % 20 == 0:
            print(f"[progress] images={n_img}")
    print(f"done: {n_img} images -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch NGA open data paintings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10, help="max images")
    ap.add_argument("--data-dir", default=None, help="dir with objects.csv/published_images.csv (else downloaded)")
    ap.add_argument("--classification", default="Painting", help="None to disable the filter")
    ap.add_argument("--year-begin", type=int, default=None)
    ap.add_argument("--year-end", type=int, default=None)
    ap.add_argument("--attribution", default=None, help="keyword in the attribution column, e.g. Monet")
    ap.add_argument("--image-width", type=int, default=843)
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    run(args.out, args.limit, args.data_dir, args.classification or None,
        args.year_begin, args.year_end, args.attribution, args.image_width, args.delay)


if __name__ == "__main__":
    main()
