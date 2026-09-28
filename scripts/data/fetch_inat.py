#!/usr/bin/env python3
"""Fetch iNaturalist research-grade photos for concept-reinforcement terms.

Source: the iNaturalist API (https://api.inaturalist.org/v1/), which needs no
key.  Photos carry a per-photo licence, so the request asks the server for
``cc0``, ``cc-by``, ``cc-by-sa`` and ``cc-by-nc`` only, and the download is
skipped if a returned photo carries anything else.  What this fetcher keeps is
the original-size picture plus the metadata needed to credit the observer and
link back to the observation: the photo's credit line as ``photographer`` and
its licence code as ``license``.

Each observation's ``photos[].url`` is a square thumbnail; the file is the same
URL with ``/square.`` replaced by ``/original.``.  The photo's own
``original_dimensions`` feed the shape filter before the download, so a
rejected picture costs no bandwidth, and the stored file is the source
original (the training precompute scales it to its buckets).

Output, under ``--out``:
    images/inat-<photo_id>.jpg     the original-size photograph
    metadata.jsonl                 one row per candidate, appended as it goes
    state.json                     per-term page cursor and exhaustion
    fetch.log                      progress, written by the caller's redirect

The run is count-oriented and resumable: ``--queries`` holds ``term | cap``
lines (a missing cap means no per-term limit beyond ``--target``).  A term is
finished once it has ``cap`` kept images or the API stops returning new
results; a rerun skips the recorded photo ids and the finished terms without
re-issuing their requests.  ``--target`` counts the rows that hold an image,
so a resumed run stops after that many more downloads.  API requests are paced
to stay under the hourly limit; the downloads run on a small thread pool,
since one page yields up to 200 photos.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import requests

from scripts.data.fetch_reinforce_common import (
    BOX,
    MAX_RATIO,
    MIN_SIDE,
    Pacer,
    add_skip,
    base_record,
    download_image,
    get_json,
    load_metadata,
    load_state,
    make_session,
    merge_download,
    parse_queries,
    save_state,
    shape_ok,
    term_budget,
    thread_session,
)

API_ROOT = "https://api.inaturalist.org/v1/observations"
# The API refuses to page past this many matches; 200 per page reaches it in
# 50 pages.
MAX_RESULTS = 10000
ALLOWED_LICENSES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc")


def original_url(square_url: str) -> str:
    """The full-size file behind the observation's square thumbnail URL."""
    return (square_url or "").replace("/square.", "/original.")


def taxon_alt(taxon: Dict) -> str:
    """The descriptive text a row carries: taxon name, common name in brackets."""
    name = (taxon.get("name") or "").strip()
    common = (taxon.get("preferred_common_name") or "").strip()
    if name and common:
        return f"{name} ({common})"
    return name or common


def observations(session: requests.Session, term: str, page: int,
                 per_page: int) -> dict:
    return get_json(session, API_ROOT, {
        "taxon_name": term,
        "photos": "true",
        "quality_grade": "research",
        "per_page": per_page,
        "page": page,
        "photo_license": ",".join(ALLOWED_LICENSES),
    })


def record_for(term: str, observation: Dict, photo: Dict, source_name: str) -> Dict:
    photo_id = int(photo["id"])
    dimensions = photo.get("original_dimensions") or {}
    return base_record(
        source=source_name,
        source_id=f"inat-{photo_id}",
        photo_id=photo_id,
        page_url=observation.get("uri"),
        original_url=original_url(photo.get("url")),
        alt=taxon_alt(observation["taxon"]),
        photographer=(photo.get("attribution") or "").strip(),
        photographer_url=None,
        source_width=dimensions.get("width"),
        source_height=dimensions.get("height"),
        query=term,
        license=(photo.get("license_code") or "").lower(),
    )


def harvest(args, session: requests.Session) -> None:
    out = Path(args.out)
    images_dir = out / "images"
    meta_path = out / "metadata.jsonl"
    state_path = out / "state.json"
    out.mkdir(parents=True, exist_ok=True)

    queries = parse_queries(Path(args.queries).read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit(f"no queries in {args.queries}")
    if args.per_page < 1:
        raise SystemExit("--per-page must be at least 1")
    pacer = Pacer(args.per_hour)
    done, kept_by_query = load_metadata(meta_path)
    state = load_state(state_path)
    total_kept = sum(kept_by_query.values())
    print(f"[start] {len(done)} candidates already recorded "
          f"({total_kept} with an image), {len(state)} queries with state",
          flush=True)

    written = 0
    requests_made = 0
    with meta_path.open("a", encoding="utf-8") as meta:
        for term, cap in queries:
            if total_kept >= args.target:
                break
            kept_term = kept_by_query.get(term, 0)
            progress = state.setdefault(term, {"page": 1, "exhausted": False})
            if progress.get("exhausted") or (cap is not None and kept_term >= cap):
                if cap is not None and kept_term >= cap and not progress.get("exhausted"):
                    progress["exhausted"] = True
                    save_state(state_path, state)
                continue
            page = max(1, int(progress.get("page") or 1))
            while True:
                if total_kept >= args.target or (cap is not None and kept_term >= cap):
                    break
                pacer.wait()
                started = time.time()
                try:
                    payload = observations(session, term, page, args.per_page)
                except Exception as exc:  # noqa: BLE001
                    print(f"[api] {term!r} page {page} failed: {exc}", flush=True)
                    break
                pacer.note(time.time() - started)
                requests_made += 1
                results = payload.get("results") or []
                if not results:
                    progress["exhausted"] = True
                    save_state(state_path, state)
                    break
                todo: List[Dict] = []
                queued = set()
                for observation in results:
                    taxon = observation.get("taxon")
                    if not taxon:
                        continue
                    for photo in observation.get("photos") or []:
                        source_id = f"inat-{int(photo['id'])}"
                        if source_id in done or source_id in queued:
                            continue
                        record = record_for(term, observation, photo, args.source_name)
                        license_code = record["license"]
                        if license_code not in ALLOWED_LICENSES:
                            add_skip(record, f"license {license_code or 'unknown'}")
                            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                            done.add(source_id)
                            continue
                        if record["source_width"] and record["source_height"]:
                            ok, why = shape_ok(record["source_width"],
                                               record["source_height"],
                                               args.min_side, args.max_ratio)
                            if not ok:
                                add_skip(record, why)
                                meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                                done.add(source_id)
                                continue
                        record["image_url"] = record["original_url"]
                        record["local_path"] = str(images_dir / f"{source_id}.jpg")
                        queued.add(source_id)
                        todo.append(record)
                budget = term_budget(cap, kept_term)
                if budget is not None:
                    todo = todo[:budget]
                if todo:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(download_image, thread_session(),
                                        record["image_url"], Path(record["local_path"]),
                                        box=BOX,
                                        shape_filter=not record["source_width"],
                                        min_side=args.min_side,
                                        max_ratio=args.max_ratio): record
                            for record in todo
                        }
                        for future in as_completed(futures):
                            record = futures[future]
                            merge_download(record, future.result())
                            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                            done.add(record["source_id"])
                            written += 1
                            if record["download_ok"]:
                                kept_term += 1
                                total_kept += 1
                    meta.flush()
                progress["page"] = page + 1
                if cap is not None and kept_term >= cap:
                    progress["exhausted"] = True
                total_results = int(payload.get("total_results") or 0)
                last_page = max(1, min(math.ceil(total_results / args.per_page),
                                       MAX_RESULTS // args.per_page))
                if page >= last_page:
                    progress["exhausted"] = True
                save_state(state_path, state)
                print(f"[page] term={term!r} page={page} kept={kept_term} "
                      f"total={total_kept} new={written}", flush=True)
                if progress["exhausted"]:
                    break
                page += 1
    print(f"[done] kept={total_kept} newly written={written} "
          f"api_requests={requests_made}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--queries", required=True,
                        help="file with one `term | cap` line per term")
    parser.add_argument("--target", type=int, default=6000,
                        help="stop once this many candidates are on disk")
    parser.add_argument("--per-page", type=int, default=200,
                        help="observations per request (API maximum)")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE,
                        help="smallest short side worth keeping, after bounding")
    parser.add_argument("--max-ratio", type=float, default=MAX_RATIO,
                        help="long side / short side above which the picture is dropped")
    parser.add_argument("--per-hour", type=int, default=3000,
                        help="API requests per hour to stay under")
    parser.add_argument("--workers", type=int, default=12,
                        help="concurrent image downloads")
    parser.add_argument("--source-name", default="reinforce_inat",
                        help="value recorded as each row's `source`")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harvest(args, make_session())


if __name__ == "__main__":
    main()
