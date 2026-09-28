#!/usr/bin/env python3
"""Fetch Wikimedia Commons JPEG photographs for concept-reinforcement terms.

Source: the MediaWiki API on https://commons.wikimedia.org/w/api.php, which
needs no key but does require a descriptive User-Agent (the session sends
one).  A search with ``filetype:bitmap`` returns files of every bitmap kind,
so the mime type is checked and only ``image/jpeg`` survives: that is what
keeps diagrams, maps, logos, screenshots and scans out of a photo batch, and
it is deliberately not loosened.  The shape filter runs on the imageinfo
width and height before the download, using the same 1792 px box maths as
fetch_pexels.  The file that lands on disk is the 1792-wide ``thumburl``.

Each row keeps the file's description (HTML stripped, ``ImageDescription``
falling back to the cleaned file title) as ``alt``, the cleaned title as
``title``, the stripped ``Artist`` as ``artist`` and ``photographer``, and
the ``LicenseShortName`` as ``license``.

Output, under ``--out``:
    images/com-<pageid>.jpg        the 1792-wide thumbnail
    metadata.jsonl                 one row per candidate, appended as it goes
    state.json                     per-term gsrcontinue cursor and exhaustion
    fetch.log                      progress, written by the caller's redirect

The run is count-oriented and resumable: ``--queries`` holds ``term | cap``
lines (a missing cap means no per-term limit beyond ``--target``).  A term is
finished once it has ``cap`` kept images or the search stops continuing; a
rerun skips the recorded page ids and the finished terms without re-issuing
their requests, and a partially walked term resumes from its gsrcontinue
cursor.  ``--target`` counts the rows that hold an image, so a resumed run
stops after that many more downloads.  API requests are paced to stay under
the hourly limit; the downloads run on a small thread pool, since one search
page yields up to 50 files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    get_json,
    load_metadata,
    load_state,
    make_session,
    merge_download,
    parse_queries,
    save_state,
    shape_ok,
    strip_html,
    term_budget,
    thread_session,
)

API = "https://commons.wikimedia.org/w/api.php"
GSR_LIMIT = 50  # the API caps non-bot search pages at 50 titles


def clean_title(title: str) -> str:
    """A file page title as display text: no ``File:`` prefix, no extension."""
    text = strip_html(title)
    if text.lower().startswith("file:"):
        text = text[5:]
    stem, extension = os.path.splitext(text)
    if extension and len(extension) <= 6:
        text = stem
    return text.replace("_", " ").strip()


def extmetadata_value(extmetadata: Dict, key: str) -> str:
    entry = extmetadata.get(key) or {}
    if not isinstance(entry, dict):
        return ""
    return strip_html(str(entry.get("value") or ""))


def search_params(term: str, cursor: Optional[Dict] = None) -> Dict:
    """One search request; ``cursor`` is the previous response's ``continue``."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": 6,
        "gsrlimit": GSR_LIMIT,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": BOX,
    }
    if cursor:
        params.update(cursor)
    return params


def candidate_from_page(term: str, page: Dict, source_name: str,
                        min_side: int = MIN_SIDE,
                        max_ratio: float = MAX_RATIO) -> Optional[Dict]:
    """One search result as a record, with source-side filters applied.

    Returns None for an entry without a pageid or imageinfo; otherwise the
    record is either downloadable (``download_ok`` unset) or marked with a
    ``skip_reason`` for the mime shape filters.
    """
    page_id = page.get("pageid")
    imageinfo = (page.get("imageinfo") or [{}])[0]
    if not page_id or not imageinfo:
        return None
    title = clean_title(page.get("title") or "")
    extmetadata = imageinfo.get("extmetadata") or {}
    description = extmetadata_value(extmetadata, "ImageDescription")
    artist = extmetadata_value(extmetadata, "Artist")
    record = base_record(
        source=source_name,
        source_id=f"com-{int(page_id)}",
        photo_id=int(page_id),
        page_url=imageinfo.get("descriptionurl"),
        original_url=imageinfo.get("url"),
        alt=description or title,
        photographer=artist,
        photographer_url=None,
        source_width=imageinfo.get("width"),
        source_height=imageinfo.get("height"),
        query=term,
        license=extmetadata_value(extmetadata, "LicenseShortName"),
        title=title,
        artist=artist,
    )
    mime = (imageinfo.get("mime") or "").lower()
    if mime != "image/jpeg":
        return add_skip(record, f"mime {mime or 'unknown'}")
    ok, why = shape_ok(int(record["source_width"] or 0),
                       int(record["source_height"] or 0), min_side, max_ratio)
    if not ok:
        return add_skip(record, why)
    record["image_url"] = imageinfo.get("thumburl") or imageinfo.get("url")
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
            progress = state.setdefault(term, {"exhausted": False})
            if progress.get("exhausted") or (cap is not None and kept_term >= cap):
                if cap is not None and kept_term >= cap and not progress.get("exhausted"):
                    progress["exhausted"] = True
                    save_state(state_path, state)
                continue
            while True:
                if total_kept >= args.target or (cap is not None and kept_term >= cap):
                    break
                pacer.wait()
                started = time.time()
                try:
                    payload = get_json(session, API,
                                       search_params(term, progress.get("continue")))
                except Exception as exc:  # noqa: BLE001
                    print(f"[api] {term!r} failed: {exc}", flush=True)
                    break
                pacer.note(time.time() - started)
                requests_made += 1
                if payload.get("error"):
                    print(f"[api] {term!r} error: {payload['error']}", flush=True)
                    break
                pages = list(((payload.get("query") or {}).get("pages") or {}).values())
                pages.sort(key=lambda page: page.get("index") or 0)
                if not pages:
                    progress["exhausted"] = True
                    save_state(state_path, state)
                    break
                todo: List[Dict] = []
                queued = set()
                for page in pages:
                    if not page.get("pageid"):
                        continue
                    source_id = f"com-{int(page['pageid'])}"
                    if source_id in done or source_id in queued:
                        continue
                    record = candidate_from_page(term, page, args.source_name,
                                                 args.min_side, args.max_ratio)
                    if record is None:
                        continue
                    if record.get("skip_reason"):
                        meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(source_id)
                        continue
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
                                        box=None, shape_filter=False,
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
                if cap is not None and kept_term >= cap:
                    progress["exhausted"] = True
                elif payload.get("continue"):
                    progress["continue"] = payload["continue"]
                else:
                    progress["exhausted"] = True
                save_state(state_path, state)
                print(f"[page] term={term!r} kept={kept_term} total={total_kept} "
                      f"new={written}", flush=True)
                if progress["exhausted"]:
                    break
    print(f"[done] kept={total_kept} newly written={written} "
          f"api_requests={requests_made}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--queries", required=True,
                        help="file with one `term | cap` line per term")
    parser.add_argument("--target", type=int, default=35000,
                        help="stop once this many candidates are on disk")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE,
                        help="smallest short side worth keeping, after bounding")
    parser.add_argument("--max-ratio", type=float, default=MAX_RATIO,
                        help="long side / short side above which the picture is dropped")
    parser.add_argument("--per-hour", type=int, default=4000,
                        help="API requests per hour to stay under")
    parser.add_argument("--workers", type=int, default=12,
                        help="concurrent image downloads")
    parser.add_argument("--source-name", default="reinforce_commons",
                        help="value recorded as each row's `source`")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harvest(args, make_session())


if __name__ == "__main__":
    main()
