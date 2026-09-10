#!/usr/bin/env python
"""Probe fetcher: The Metropolitan Museum of Art (CC0 public-domain API).

Discovery: GET /search?hasImages=true&departmentId=<id>&q=* -> objectIDs
Metadata:  GET /objects/<id>; keep isPublicDomain==true with non-empty primaryImage.
Downloads primaryImageSmall for all kept records, plus full primaryImage for a
small random sample to measure real full-res dimensions/file sizes.

Writes to data/raw/met/:
  images/            <source_id>.jpg (primaryImageSmall)
  images_full/       <source_id>.jpg (full primaryImage sample)
  metadata.parquet
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    download_image,
    get_json,
    make_session,
    phash_dedup_report,
    save_parquet,
)

BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
DEPTS = (6, 11)  # Asian Art, European Paintings


def fetch_records(session, per_dept_target: dict[int, int]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats: dict[int, dict] = {}
    rng = random.Random(42)
    for dep in DEPTS:
        target = per_dept_target[dep]
        j = get_json(
            session,
            f"{BASE}/search",
            params={"hasImages": "true", "departmentId": dep, "q": "*"},
        )
        ids = j.get("objectIDs") or []
        rng.shuffle(ids)
        kept = fetched = 0
        for oid in ids:
            if kept >= target:
                break
            fetched += 1
            try:
                o = get_json(session, f"{BASE}/objects/{oid}", timeout=20)
            except Exception as e:  # transient; skip and keep going
                print(f"  object {oid}: {type(e).__name__} {e}", flush=True)
                time.sleep(0.12)
                continue
            time.sleep(0.12)  # stay well under the 80 req/s limit
            if not o.get("isPublicDomain") or not o.get("primaryImage"):
                continue
            kept += 1
            rows.append(
                {
                    "source": "met",
                    "source_id": str(o["objectID"]),
                    "title": o.get("title") or "",
                    "artist": o.get("artistDisplayName") or "",
                    "date": o.get("objectDate") or "",
                    "classification": o.get("classification") or "",
                    "department": o.get("department") or "",
                    "culture": o.get("culture") or "",
                    "license": "CC0",
                    "image_url": o.get("primaryImage") or "",
                    "image_url_small": o.get("primaryImageSmall") or "",
                    "local_path": "",
                    "width": None,
                    "height": None,
                    "download_ok": False,
                }
            )
            if kept % 50 == 0:
                print(f"  dept {dep}: kept {kept}/{target} (fetched {fetched})", flush=True)
        stats[dep] = {"total_ids": len(ids), "fetched": fetched, "kept": kept}
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/met")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--full-sample", type=int, default=20)
    ap.add_argument("--dedup-sample", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    half = args.limit // 2
    per_dept = {DEPTS[0]: half, DEPTS[1]: args.limit - half}

    session = make_session()

    t0 = time.time()
    rows, stats = fetch_records(session, per_dept)
    print(f"metadata fetch: {len(rows)} records in {time.time()-t0:.0f}s; stats={stats}")

    # --- download primaryImageSmall for all kept records ---
    img_dir = out / "images"
    t0 = time.time()
    n_ok = 0
    for r in rows:
        dest = img_dir / f"{r['source_id']}.jpg"
        w, h = download_image(session, r["image_url_small"], dest, timeout=60)
        if w:
            r["width"], r["height"] = w, h
            r["local_path"] = str(dest)
            r["download_ok"] = True
            n_ok += 1
        time.sleep(0.05)
        if n_ok and n_ok % 100 == 0:
            el = time.time() - t0
            print(f"  small dl: {n_ok}/{len(rows)} ok, {n_ok/el:.2f} img/s", flush=True)
    el = time.time() - t0
    print(f"small downloads: {n_ok}/{len(rows)} ok in {el:.0f}s ({n_ok/el:.2f} img/s)")

    # --- full primaryImage for a random sample ---
    full_dir = out / "images_full"
    rng = random.Random(123)
    sample = rng.sample(rows, min(args.full_sample, len(rows)))
    full_stats = []
    t0 = time.time()
    for r in sample:
        dest = full_dir / f"{r['source_id']}.jpg"
        w, h = download_image(session, r["image_url"], dest, timeout=180)
        if w:
            mb = dest.stat().st_size / 1e6
            full_stats.append(
                {"source_id": r["source_id"], "width": w, "height": h, "mb": round(mb, 2)}
            )
            print(f"  full {r['source_id']}: {w}x{h} {mb:.1f}MB", flush=True)
        time.sleep(0.1)
    print(f"full-res sample: {len(full_stats)}/{len(sample)} ok in {time.time()-t0:.0f}s")

    save_parquet(rows, out / "metadata.parquet")
    print(f"saved {len(rows)} rows -> {out/'metadata.parquet'}")

    # --- phash dedup on a sample of downloaded smalls ---
    paths = [Path(r["local_path"]) for r in rows if r["download_ok"]]
    rng.shuffle(paths)
    rep = phash_dedup_report(paths[: args.dedup_sample])
    print(
        f"phash dedup: n={rep['n_hashed']} near-dups={rep['n_duplicates']} "
        f"rate={rep['dup_rate']:.4f}"
    )

    # --- field completeness ---
    n = len(rows)
    for f in ("title", "artist", "date", "classification", "culture"):
        frac = sum(1 for r in rows if r[f]) / n if n else 0.0
        print(f"field {f}: {frac:.3f}")


if __name__ == "__main__":
    main()
