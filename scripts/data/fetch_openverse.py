#!/usr/bin/env python3
"""Fetch Openverse CC-licensed photos for concept-reinforcement terms.

Source: the Openverse API (https://api.openverse.org/v1/images/), an
aggregator over Flickr, museums and stock sites.  Anonymous access needs no
key but is capped at 200 requests/day (20/minute burst) with at most 20
results per page, so this leg is a slow-but-steady complement to the
Wikimedia Commons leg: ``excluded_source=wikimedia`` keeps it off the
upload.wikimedia.org thumbnails that throttle this host, and most of what
remains is served by Flickr's CDN.  The licence classes requested
(``cc0,by,by-nc,by-sa``) match the other legs' posture.

Each result's ``url`` is the provider's direct image; dimensions come with
most rows, so the shape filter mostly runs before the download, and the
saved file is bounded to the same 1792 px box.  Rows carry the result title
as ``alt``/``title``, the creator as ``photographer`` and the licence as an
iNat-style short code.

Output, under ``--out``:
    images/ov-<uuid>.jpg
    metadata.jsonl                 one row per candidate, appended as it goes
    state.json                     per-term page cursor and exhaustion
    fetch.log                      progress, written by the caller's redirect

The run is count-oriented and resumable, following fetch_inat: ``--queries``
holds ``term | cap`` lines, a term is finished once it has ``cap`` kept
images or the search stops returning new results, and a rerun skips the
recorded ids and the finished terms without re-issuing their requests.
``--target`` counts the rows that hold an image.  API requests are paced to
stay under the daily budget; the downloads run on a small thread pool.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from scripts.data.fetch_reinforce_common import (
    BOX,
    MAX_RATIO,
    MIN_SIDE,
    Pacer,
    add_skip,
    base_record,
    download_image,
    download_candidates,
    get_json,
    load_metadata,
    load_state,
    make_session,
    merge_download,
    parse_queries,
    save_state,
    shape_ok,
    thread_session,
)

API = "https://api.openverse.org/v1/images/"
LICENSE_CODES = {"cc0": "cc0", "by": "cc-by", "by-nc": "cc-by-nc",
                 "by-sa": "cc-by-sa"}
REQUESTED_LICENSES = ",".join(LICENSE_CODES)


def search_params(term: str, page: int, page_size: int) -> Dict:
    return {
        "q": term,
        "license": REQUESTED_LICENSES,
        "excluded_source": "wikimedia",
        "page_size": page_size,
        "page": page,
    }


def photo_id_of(identifier: str) -> int:
    """A stable integer id from the Openverse UUID (matches base_record's int)."""
    return int(str(identifier).replace("-", "")[:12], 16)


def candidate_from_result(term: str, result: Dict, source_name: str,
                          min_side: int = MIN_SIDE,
                          max_ratio: float = MAX_RATIO) -> Optional[Dict]:
    """One search result as a record, with source-side filters applied."""
    identifier = result.get("id")
    url = (result.get("url") or "").strip()
    if not identifier or not url:
        return None
    title = (result.get("title") or "").strip()
    license_code = LICENSE_CODES.get((result.get("license") or "").lower())
    record = base_record(
        source=source_name,
        source_id=f"ov-{identifier}",
        photo_id=photo_id_of(identifier),
        page_url=result.get("foreign_landing_url"),
        original_url=url,
        alt=title,
        photographer=(result.get("creator") or "").strip(),
        photographer_url=result.get("creator_url"),
        source_width=result.get("width"),
        source_height=result.get("height"),
        query=term,
        license=license_code or "",
        title=title,
    )
    if not license_code:
        return add_skip(record, f"license {result.get('license') or 'unknown'}")
    if record["source_width"] and record["source_height"]:
        ok, why = shape_ok(int(record["source_width"]), int(record["source_height"]),
                           min_side, max_ratio)
        if not ok:
            return add_skip(record, why)
    record["image_url"] = url
    return record


def harvest(args, session: requests.Session) -> None:
    out = Path(args.out)
    images_dir = out / "images"
    meta_path = out / "metadata.jsonl"
    state_path = out / "state.json"
    out.mkdir(parents=True, exist_ok=True)

    queries = parse_queries(Path(args.queries).read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit(f"no queries in {args.queries}")
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
                continue
            page = max(1, int(progress.get("page") or 1))
            while True:
                if total_kept >= args.target or (cap is not None and kept_term >= cap):
                    break
                pacer.wait()
                started = time.time()
                try:
                    payload = get_json(session, API,
                                       search_params(term, page, args.page_size))
                except Exception as exc:  # noqa: BLE001
                    # Most often the daily anonymous budget; the term keeps its
                    # page cursor and a later run resumes here.
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
                for result in results:
                    record = candidate_from_result(term, result, args.source_name,
                                                   args.min_side, args.max_ratio)
                    if record is None:
                        continue
                    source_id = record["source_id"]
                    if source_id in done or source_id in queued:
                        continue
                    if record.get("skip_reason"):
                        meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(source_id)
                        continue
                    record["local_path"] = str(images_dir / f"{source_id}.jpg")
                    queued.add(source_id)
                    todo.append(record)
                budget = min(args.target - total_kept, cap - kept_term if cap is not None else args.target)
                def download(record):
                    return download_image(
                        thread_session(), record["image_url"], Path(record["local_path"]),
                        box=BOX, shape_filter=not record["source_width"], min_side=args.min_side,
                        max_ratio=args.max_ratio)

                for record, result in download_candidates(
                        todo, download, workers=args.workers, max_successes=budget):
                    merge_download(record, result)
                    meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                    meta.flush()
                    done.add(record["source_id"])
                    written += 1
                    if record["download_ok"]:
                        kept_term += 1
                        total_kept += 1
                meta.flush()
                if total_kept >= args.target or (cap is not None and kept_term >= cap):
                    # Revisit this page on resume; metadata skips completed candidates.
                    save_state(state_path, state)
                    break
                page_count = int(payload.get("page_count") or 1)
                progress["page"] = page + 1
                if page >= page_count:
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
    parser.add_argument("--target", type=int, default=30000,
                        help="stop once this many candidates are on disk")
    parser.add_argument("--page-size", type=int, default=20,
                        help="results per request (anonymous maximum)")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE,
                        help="smallest short side worth keeping, after bounding")
    parser.add_argument("--max-ratio", type=float, default=MAX_RATIO,
                        help="long side / short side above which the picture is dropped")
    parser.add_argument("--per-hour", type=int, default=8,
                        help="API requests per hour to stay under (anonymous 200/day)")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent image downloads")
    parser.add_argument("--source-name", default="reinforce_openverse",
                        help="value recorded as each row's `source`")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harvest(args, make_session())


if __name__ == "__main__":
    main()
