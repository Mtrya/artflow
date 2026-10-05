#!/usr/bin/env python3
"""Fetch GBIF occurrence photos for concept-reinforcement terms.

Source: the GBIF API (https://api.gbif.org/v1/), which needs no key.  Each
term is resolved to a taxon once (iNaturalist's ``taxa`` endpoint handles the
vernacular name, GBIF's ``species/match`` the scientific name; the resolved
key is cached in the state file, so a resume does not re-match) and its
occurrences are paged with ``mediaType=StillImage`` plus the three GBIF
licence classes (``CC0_1_0``, ``CC_BY_4_0``, ``CC_BY_NC_4_0``) — the same
licence posture as the iNat leg.  GBIF re-exports the iNaturalist research-grade dataset, which
the iNat leg already harvests, so rows carrying that dataset key are recorded
as skips instead of duplicating it; everything else (museum specimens, eBird,
smaller citizen-science portals) is new supply.

Each occurrence's ``media[]`` carries the image URL.  Dimensions are usually
absent, so the shape filter runs on the decoded picture, the way the iNat
fetcher handles photos without ``original_dimensions``.  The saved file is
bounded to the same 1792 px box.

Output, under ``--out``:
    images/gbif-<occurrence_key>-<media_index>.jpg
    metadata.jsonl                 one row per candidate, appended as it goes
    state.json                     per-term taxon key, offset and exhaustion
    fetch.log                      progress, written by the caller's redirect

The run is count-oriented and resumable, following fetch_inat: ``--queries``
holds ``term | cap`` lines, a term is finished once it has ``cap`` kept
images or the search stops returning new results, and a rerun skips the
recorded ids and the finished terms without re-issuing their requests.
``--target`` counts the rows that hold an image.  API requests are paced to
stay under the hourly limit; the downloads run on a small thread pool, since
one page yields up to 300 occurrences.
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

MATCH = "https://api.gbif.org/v1/species/match"
SEARCH = "https://api.gbif.org/v1/occurrence/search"
# GBIF's species/match no longer resolves vernacular names; iNaturalist's
# taxa endpoint does (it is what the iNat leg's taxon_name search uses), so
# term resolution goes through it first and the scientific name it returns
# is matched against GBIF.
INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
# The API refuses to page past this many matches.
MAX_RESULTS = 100000
LICENSES = (("license", "CC0_1_0"), ("license", "CC_BY_4_0"),
            ("license", "CC_BY_NC_4_0"))
# GBIF republishes iNaturalist research-grade observations; the iNat leg
# already has them.
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"


def license_code(value: str) -> str:
    """A GBIF licence URL or enum as an iNat-style short code."""
    text = (value or "").lower().replace("_", " ")
    if "cc0" in text or "publicdomain" in text or "public domain" in text:
        return "cc0"
    if "by nc" in text or "by-nc" in text:
        return "cc-by-nc"
    if "by" in text:
        return "cc-by"
    return ""


def occurrence_alt(result: Dict) -> str:
    """The descriptive text a row carries: scientific name, common in brackets."""
    scientific = (result.get("scientificName") or "").strip()
    vernacular = (result.get("vernacularName") or "").strip()
    if scientific and vernacular:
        return f"{scientific} ({vernacular})"
    return scientific or vernacular


def match_taxon(session: requests.Session, term: str) -> dict:
    """Resolve a vernacular term to a GBIF taxon via iNaturalist's resolver.

    Returns the GBIF match payload (with ``resolved_via`` set to the iNat
    scientific name), or ``{}`` when either step fails to resolve.
    """
    taxa = get_json(session, INAT_TAXA, {"q": term, "per_page": 1})
    results = taxa.get("results") or []
    scientific = (results[0].get("name") or "").strip() if results else ""
    if not scientific:
        return {}
    matched = get_json(session, MATCH, {"name": scientific})
    if matched.get("usageKey"):
        matched["resolved_via"] = scientific
    return matched


def search_params(taxon_key: int, limit: int, offset: int) -> List:
    """One search request; repeated ``license`` params are OR'd by the API."""
    return [("taxon_key", taxon_key), ("mediaType", "StillImage"),
            ("limit", limit), ("offset", offset), *LICENSES]


def candidate_from_occurrence(term: str, result: Dict, media_index: int,
                              media: Dict, source_name: str) -> Optional[Dict]:
    """One occurrence's media item as a record, with source-side filters applied.

    Returns None when the occurrence has no usable key; otherwise the record
    is either downloadable (``download_ok`` unset) or marked with a
    ``skip_reason`` for the licence / reexport / shape filters.
    """
    key = result.get("key")
    identifier = (media.get("identifier") or "").strip()
    if not key or not identifier:
        return None
    record = base_record(
        source=source_name,
        source_id=f"gbif-{int(key)}-{media_index}",
        photo_id=int(key),
        page_url=result.get("references") or f"https://www.gbif.org/occurrence/{int(key)}",
        original_url=identifier,
        alt=occurrence_alt(result),
        photographer=(media.get("creator") or result.get("recordedBy") or "").strip(),
        photographer_url=None,
        source_width=media.get("width"),
        source_height=media.get("height"),
        query=term,
        license=license_code(media.get("license") or result.get("license")),
        title=(media.get("title") or "").strip(),
    )
    if result.get("datasetKey") == INAT_DATASET:
        return add_skip(record, "inat reexport")
    if not record["license"]:
        return add_skip(record, f"license {result.get('license') or 'unknown'}")
    if record["source_width"] and record["source_height"]:
        ok, why = shape_ok(int(record["source_width"]), int(record["source_height"]),
                           MIN_SIDE, MAX_RATIO)
        if not ok:
            return add_skip(record, why)
    record["image_url"] = identifier
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
            progress = state.setdefault(term, {"offset": 0, "exhausted": False})
            if progress.get("exhausted") or (cap is not None and kept_term >= cap):
                continue
            taxon_key = progress.get("taxon_key")
            if taxon_key is None:
                pacer.wait()
                started = time.time()
                try:
                    matched = match_taxon(session, term)
                except Exception as exc:  # noqa: BLE001
                    print(f"[match] {term!r} failed: {exc}", flush=True)
                    continue
                pacer.note(time.time() - started)
                requests_made += 1
                taxon_key = matched.get("usageKey")
                progress["taxon_key"] = taxon_key
                progress["matched_name"] = matched.get("scientificName") or ""
                save_state(state_path, state)
                if not taxon_key:
                    progress["exhausted"] = True
                    save_state(state_path, state)
                    print(f"[match] {term!r}: no taxon", flush=True)
                    continue
                print(f"[match] {term!r} -> {progress['matched_name']!r} "
                      f"key={taxon_key} confidence={matched.get('confidence')} "
                      f"via={matched.get('resolved_via')!r}", flush=True)
            offset = max(0, int(progress.get("offset") or 0))
            while True:
                if total_kept >= args.target or (cap is not None and kept_term >= cap):
                    break
                pacer.wait()
                started = time.time()
                try:
                    payload = get_json(session, SEARCH,
                                       search_params(taxon_key, args.per_page, offset))
                except Exception as exc:  # noqa: BLE001
                    print(f"[api] {term!r} offset {offset} failed: {exc}", flush=True)
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
                    for media_index, media in enumerate(result.get("media") or []):
                        if media.get("type") not in (None, "StillImage"):
                            continue
                        record = candidate_from_occurrence(
                            term, result, media_index, media, args.source_name)
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
                offset += len(results)
                progress["offset"] = offset
                if payload.get("endOfRecords", True) or offset >= MAX_RESULTS:
                    progress["exhausted"] = True
                save_state(state_path, state)
                print(f"[page] term={term!r} offset={offset} kept={kept_term} "
                      f"total={total_kept} new={written}", flush=True)
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
    parser.add_argument("--target", type=int, default=5000,
                        help="stop once this many candidates are on disk")
    parser.add_argument("--per-page", type=int, default=300,
                        help="occurrences per request (API maximum)")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE,
                        help="smallest short side worth keeping, after bounding")
    parser.add_argument("--max-ratio", type=float, default=MAX_RATIO,
                        help="long side / short side above which the picture is dropped")
    parser.add_argument("--per-hour", type=int, default=3000,
                        help="API requests per hour to stay under")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent image downloads")
    parser.add_argument("--source-name", default="reinforce_gbif",
                        help="value recorded as each row's `source`")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harvest(args, make_session())


if __name__ == "__main__":
    main()
