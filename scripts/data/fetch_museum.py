#!/usr/bin/env python3
"""Fetch Met and AIC public-domain artwork photographs for reinforcement terms.

Sources, both keyless and CC0 for the images this fetcher keeps:

* The Metropolitan Museum of Art Open Access API
  (https://collectionapi.metmuseum.org/public/collection/v1/).  Search returns
  the objectIDs for a term; each object is then read one by one, and only
  objects with a non-empty ``primaryImage`` are downloaded.  Met serves the
  full-size JPEG (several MB), so it is downscaled on save to fit the same
  1792 px box fetch_pexels asks its CDN for, with LANCZOS.  Licence line:
  ``CC0 (Met Open Access)``.
* The Art Institute of Chicago API (https://api.artic.edu/api/v1/).  Search is
  paged (100 works per page) and keeps only ``is_public_domain`` works with an
  ``image_id``; the image is requested through IIIF at ``!1792,1792``, so the
  server bounds both dimensions.  Licence line:
  ``CC0 (Art Institute of Chicago)``.

``--apis`` picks the sources to walk (default both).  A term's cap is shared
across them: Met always runs first and fills what it can, AIC tops up the
remainder.  Records follow fetch_pexels.py plus ``title`` and ``artist``; the
row's short caption is the title and the artist is carried separately, with
``photographer`` equal to it.

Output, under ``--out``:
    images/met-<objectID>.jpg      Met primaryImage, bounded to 1792 px
    images/aic-<id>.jpg            AIC IIIF image, server-bounded to 1792 px
    metadata.jsonl                 one row per candidate, appended as it goes
    state.json                     per-term Met index, AIC page, exhaustion
    fetch.log                      progress, written by the caller's redirect

The run is count-oriented and resumable: ``--queries`` holds ``term | cap``
lines (a missing cap means no per-term limit beyond ``--target``).  A term is
finished once it has ``cap`` kept images or both listed sources stop returning
new results; a rerun skips the recorded source ids and the finished terms
without re-issuing their requests.  ``--target`` counts the rows that hold an
image, so a resumed run stops after that many more downloads.  Met object
calls are numerous and get their own hourly budget (default 6000); AIC's
anonymous traffic is limited to 60 requests per minute, so its budget defaults
to 4000 and is paced separately.
"""

from __future__ import annotations

import argparse
import json
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
    term_budget,
    thread_session,
)

MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
MET_LICENSE = "CC0 (Met Open Access)"

AIC_SEARCH = "https://api.artic.edu/api/v1/artworks/search"
AIC_IIIF = "https://www.artic.edu/iiif/2"
AIC_LICENSE = "CC0 (Art Institute of Chicago)"
AIC_FIELDS = "id,title,image_id,artist_title,is_public_domain"
AIC_PAGE = 100
# Elasticsearch window: page * limit may not exceed 10000, so 100 pages.
MAX_AIC_PAGES = 100

API_ORDER = ("met", "aic")


def parse_apis(spec: str) -> List[str]:
    """Read ``--apis`` into the order the sources are walked in."""
    names = [part.strip().lower() for part in spec.split(",") if part.strip()]
    unknown = [name for name in names if name not in API_ORDER]
    if unknown:
        raise SystemExit(f"unknown --apis entries: {', '.join(unknown)}")
    if not names:
        raise SystemExit("--apis must list met and/or aic")
    return [name for name in API_ORDER if name in names]


def met_search(session: requests.Session, term: str) -> List[int]:
    payload = get_json(session, MET_SEARCH, {"hasImages": "true", "q": term})
    return [int(object_id) for object_id in (payload.get("objectIDs") or [])]


def met_object(session: requests.Session, object_id: int) -> dict:
    return get_json(session, f"{MET_OBJECT}/{object_id}", {})


def met_record(term: str, obj: Dict, source_name: str) -> Dict:
    object_id = int(obj["objectID"])
    title = (obj.get("title") or "").strip()
    artist = (obj.get("artistDisplayName") or "").strip()
    return base_record(
        source=source_name,
        source_id=f"met-{object_id}",
        photo_id=object_id,
        page_url=obj.get("objectURL"),
        original_url=(obj.get("primaryImage") or "").strip(),
        alt=title,
        photographer=artist,
        photographer_url=None,
        source_width=None,
        source_height=None,
        query=term,
        license=MET_LICENSE,
        title=title,
        artist=artist,
    )


def aic_search(session: requests.Session, term: str, page: int) -> dict:
    return get_json(session, AIC_SEARCH, {
        "q": term, "page": page, "limit": AIC_PAGE, "fields": AIC_FIELDS,
    })


def aic_image_url(iiif_prefix: str, image_id: str) -> str:
    return (f"{iiif_prefix.rstrip('/')}/{image_id}"
            f"/full/!{BOX},{BOX}/0/default.jpg")


def aic_record(term: str, item: Dict, source_name: str) -> Dict:
    artwork_id = int(item["id"])
    title = (item.get("title") or "").strip()
    artist = (item.get("artist_title") or "").strip()
    return base_record(
        source=source_name,
        source_id=f"aic-{artwork_id}",
        photo_id=artwork_id,
        page_url=f"https://www.artic.edu/artworks/{artwork_id}",
        original_url="",
        alt=title,
        photographer=artist,
        photographer_url=None,
        source_width=None,
        source_height=None,
        query=term,
        license=AIC_LICENSE,
        title=title,
        artist=artist,
    )


def harvest(args, session: requests.Session) -> None:
    apis = parse_apis(args.apis)
    out = Path(args.out)
    images_dir = out / "images"
    meta_path = out / "metadata.jsonl"
    state_path = out / "state.json"
    out.mkdir(parents=True, exist_ok=True)

    queries = parse_queries(Path(args.queries).read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit(f"no queries in {args.queries}")
    pacer = Pacer(args.per_hour)
    aic_pacer = Pacer(args.per_hour_aic)
    done, kept_by_query = load_metadata(meta_path)
    state = load_state(state_path)
    total_kept = sum(kept_by_query.values())
    print(f"[start] apis={','.join(apis)} {len(done)} candidates already recorded "
          f"({total_kept} with an image), {len(state)} queries with state", flush=True)

    written = 0
    requests_made = 0
    with meta_path.open("a", encoding="utf-8") as meta:
        for term, cap in queries:
            if total_kept >= args.target:
                break
            kept_term = kept_by_query.get(term, 0)
            progress = state.setdefault(term, {})
            if progress.get("exhausted") or (cap is not None and kept_term >= cap):
                if cap is not None and kept_term >= cap and not progress.get("exhausted"):
                    progress["exhausted"] = True
                    save_state(state_path, state)
                continue

            if "met" in apis and not progress.get("met_done"):
                try:
                    pacer.wait()
                    started = time.time()
                    object_ids = met_search(session, term)
                    pacer.note(time.time() - started)
                    requests_made += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[api] met {term!r} search failed: {exc}", flush=True)
                    object_ids = None
                if object_ids is not None:
                    index = max(0, int(progress.get("met_index") or 0))
                    objects = 0
                    while index < len(object_ids):
                        if total_kept >= args.target or (cap is not None and kept_term >= cap):
                            break
                        object_id = object_ids[index]
                        index += 1
                        source_id = f"met-{object_id}"
                        if source_id in done:
                            continue
                        pacer.wait()
                        started = time.time()
                        try:
                            obj = met_object(session, object_id)
                        except Exception as exc:  # noqa: BLE001 - one object is not the run
                            print(f"[api] met object {object_id} failed: {exc}", flush=True)
                            continue
                        pacer.note(time.time() - started)
                        requests_made += 1
                        objects += 1
                        record = met_record(term, obj, args.source_name)
                        if not record["original_url"]:
                            small = (obj.get("primaryImageSmall") or "").strip()
                            add_skip(record, "primaryImageSmall only" if small
                                     else "no primaryImage")
                            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                            done.add(source_id)
                            continue
                        record["image_url"] = record["original_url"]
                        record["local_path"] = str(images_dir / f"{source_id}.jpg")
                        merge_download(record, download_image(
                            thread_session(), record["image_url"],
                            Path(record["local_path"]), box=BOX, shape_filter=True,
                            min_side=args.min_side, max_ratio=args.max_ratio))
                        meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(source_id)
                        written += 1
                        if record["download_ok"]:
                            kept_term += 1
                            total_kept += 1
                        if objects % 100 == 0:
                            meta.flush()
                            print(f"[met] term={term!r} object={index}/{len(object_ids)} "
                                  f"kept={kept_term} total={total_kept}", flush=True)
                    progress["met_index"] = index
                    if index >= len(object_ids):
                        progress["met_done"] = True
                    save_state(state_path, state)

            if ("aic" in apis and not progress.get("aic_done")
                    and not (cap is not None and kept_term >= cap)
                    and total_kept < args.target):
                page = max(1, int(progress.get("aic_page") or 1))
                while True:
                    if total_kept >= args.target or (cap is not None and kept_term >= cap):
                        break
                    aic_pacer.wait()
                    started = time.time()
                    try:
                        payload = aic_search(session, term, page)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[api] aic {term!r} page {page} failed: {exc}", flush=True)
                        break
                    aic_pacer.note(time.time() - started)
                    requests_made += 1
                    entries = payload.get("data") or []
                    if not entries:
                        progress["aic_done"] = True
                        save_state(state_path, state)
                        break
                    prefix = ((payload.get("config") or {}).get("iiif_url")
                              or AIC_IIIF)
                    todo: List[Dict] = []
                    queued = set()
                    for item in entries:
                        source_id = f"aic-{int(item['id'])}"
                        if source_id in done or source_id in queued:
                            continue
                        record = aic_record(term, item, args.source_name)
                        image_id = item.get("image_id")
                        if not image_id:
                            add_skip(record, "no image_id")
                            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                            done.add(source_id)
                            continue
                        if not item.get("is_public_domain"):
                            add_skip(record, "not public domain")
                            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                            done.add(source_id)
                            continue
                        record["original_url"] = aic_image_url(prefix, image_id)
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
                                            record["image_url"],
                                            Path(record["local_path"]), box=None,
                                            shape_filter=True, min_side=args.min_side,
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
                    progress["aic_page"] = page + 1
                    total_pages = int((payload.get("pagination") or {}).get("total_pages") or 0)
                    if not total_pages or page >= min(total_pages, MAX_AIC_PAGES):
                        progress["aic_done"] = True
                    save_state(state_path, state)
                    print(f"[aic] term={term!r} page={page} kept={kept_term} "
                          f"total={total_kept} new={written}", flush=True)
                    if progress["aic_done"]:
                        break
                    page += 1

            met_finished = "met" not in apis or progress.get("met_done")
            aic_finished = "aic" not in apis or progress.get("aic_done")
            if cap is not None and kept_term >= cap:
                progress["exhausted"] = True
            elif met_finished and aic_finished:
                progress["exhausted"] = True
            save_state(state_path, state)

    print(f"[done] kept={total_kept} newly written={written} "
          f"api_requests={requests_made}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--queries", required=True,
                        help="file with one `term | cap` line per term")
    parser.add_argument("--apis", default="met,aic",
                        help="comma-separated sources to walk (met, aic)")
    parser.add_argument("--target", type=int, default=20000,
                        help="stop once this many candidates are on disk")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE,
                        help="smallest short side worth keeping, after bounding")
    parser.add_argument("--max-ratio", type=float, default=MAX_RATIO,
                        help="long side / short side above which the picture is dropped")
    parser.add_argument("--per-hour", type=int, default=6000,
                        help="Met API requests per hour to stay under")
    parser.add_argument("--per-hour-aic", type=int, default=4000,
                        help="AIC API requests per hour to stay under")
    parser.add_argument("--workers", type=int, default=12,
                        help="concurrent image downloads")
    parser.add_argument("--source-name", default="reinforce_museum",
                        help="value recorded as each row's `source`")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harvest(args, make_session())


if __name__ == "__main__":
    main()
