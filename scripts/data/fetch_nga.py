"""Probe fetcher: National Gallery of Art, Washington DC (opendata CSVs + IIIF).

Source: github.com/NationalGalleryOfArt/opendata (CC0 data; open-access images
flagged via published_images.openaccess == 1, served over IIIF).

Joins published_images (open-access, primary view, non-empty iiifurl) to objects
on depictstmsobjectid == objectid, keeps painting-like media, samples
~half from the 1860-1910 impressionism core and ~half from a random spread of
the rest, then downloads {iiifurl}/full/!1024,1024/0/default.jpg.

CSVs are cached under <out>/_csv/ and reused across runs.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    download_image, make_session, phash_dedup_report, save_parquet,
)

CSV_BASE = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/master/data"
PAINTING_LIKE = re.compile(
    r"\b(?:oil|tempera|watercolor|watercolour|gouache|acrylic|encaustic|fresco|pastel)\b",
    re.IGNORECASE,
)


def fetch_csv(session, name: str, cache_dir: Path) -> Path:
    dest = cache_dir / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[csv] using cached {dest}")
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{CSV_BASE}/{name}"
    print(f"[csv] downloading {url}")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def build_sample(csv_dir: Path, limit: int, seed: int = 42) -> pd.DataFrame:
    objs = pd.read_csv(
        csv_dir / "objects.csv", low_memory=False,
        usecols=["objectid", "title", "displaydate", "beginyear",
                 "medium", "attribution", "classification", "departmentabbr"],
    )
    imgs = pd.read_csv(
        csv_dir / "published_images.csv", low_memory=False,
        usecols=["uuid", "iiifurl", "viewtype", "width", "height",
                 "openaccess", "depictstmsobjectid"],
    )
    imgs = imgs[(imgs.openaccess == 1) & imgs.iiifurl.notna()
                & (imgs.viewtype == "primary")]
    m = imgs.merge(objs, left_on="depictstmsobjectid",
                   right_on="objectid", how="inner")
    m = m[m.medium.fillna("").str.contains(PAINTING_LIKE)]
    m = m.drop_duplicates(subset="depictstmsobjectid")
    print(f"[join] open-access painting-like primary images: {len(m)}")

    core = m[(m.beginyear >= 1860) & (m.beginyear <= 1910)]
    rest = m.drop(core.index)
    n_core = min(limit // 2, len(core))
    n_rest = min(limit - n_core, len(rest))
    print(f"[sample] core 1860-1910 pool={len(core)} take={n_core}; "
          f"rest pool={len(rest)} take={n_rest}")
    sample = pd.concat([
        core.sample(n=n_core, random_state=seed),
        rest.sample(n=n_rest, random_state=seed),
    ])
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/nga")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    session = make_session(proxy=args.proxy)

    csv_dir = out / "_csv"
    fetch_csv(session, "objects.csv", csv_dir)
    fetch_csv(session, "published_images.csv", csv_dir)

    sample = build_sample(csv_dir, args.limit)

    rows: list[dict] = []
    n_ok = 0
    t0 = time.time()
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        source_id = str(int(r.depictstmsobjectid))
        url = f"{r.iiifurl}/full/!1024,1024/0/default.jpg"
        dest = img_dir / f"{source_id}.jpg"
        w, h = download_image(session, url, dest)
        ok = w is not None
        n_ok += ok
        rows.append({
            "source": "nga",
            "source_id": source_id,
            "title": r.title if pd.notna(r.title) else None,
            "artist": r.attribution if pd.notna(r.attribution) else None,
            "date": r.displaydate if pd.notna(r.displaydate) else None,
            "classification": r.classification if pd.notna(r.classification) else None,
            "department": r.departmentabbr if pd.notna(r.departmentabbr) else None,
            "license": "CC0",
            "image_url": url,
            "local_path": str(dest) if ok else None,
            "width": w,
            "height": h,
            "download_ok": ok,
        })
        if i % 100 == 0:
            dt = time.time() - t0
            print(f"[dl] {i}/{len(sample)} ok={n_ok} "
                  f"({i / dt:.2f} img/s)")
        time.sleep(args.sleep)

    save_parquet(rows, out / "metadata.parquet")
    dt = time.time() - t0
    print(f"[done] {len(rows)} rows, {n_ok} images in {dt:.0f}s "
          f"({len(rows) / dt:.2f} img/s incl. sleeps)")

    ok_paths = [Path(r["local_path"]) for r in rows if r["download_ok"]]
    rep = phash_dedup_report(ok_paths[:300])
    print(f"[phash] n={rep['n_hashed']} dups={rep['n_duplicates']} "
          f"rate={rep['dup_rate']:.4f}")


if __name__ == "__main__":
    main()
