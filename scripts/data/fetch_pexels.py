#!/usr/bin/env python3
"""Fetcher: Pexels photo search, restricted to photographs of people.

Source: the Pexels API (https://www.pexels.com/api/).  The licence
(https://www.pexels.com/license/) allows free use and modification, forbids
resale of unaltered copies and redistribution on stock-photo or wallpaper
platforms, and asks that photographers be credited and Pexels be linked.  What
this fetcher keeps locally is the bounded-size image plus the metadata needed to
credit the photographer and link back, so a later publication can carry the
credit and a download recipe instead of the image bytes.

The search terms are chosen for the gap this source fills: photographs of people
across regions and in traditional dress, which the corpus is otherwise short of.

Images are fetched through the CDN's own scaling parameters rather than at full
resolution: ``?auto=compress&cs=tinysrgb&w=1792&h=1792`` fits the picture inside
a 1792 px box (aspect kept, no crop), which is ~200 kB instead of several MB and
still resolves the largest training bucket with room to spare.

Output, under ``--out``:
    images/pexels-<photo_id>.jpg     the bounded-size photograph
    metadata.jsonl                   one row per candidate, appended as it goes
    fetch.log                        progress, written by the caller's redirect

The run is resumable: photo ids already present in metadata.jsonl are skipped,
and a download that already landed on disk is not repeated.  API requests are
paced to stay under the published hourly limit; the downloads themselves run on
a small thread pool, since one page of results yields up to 80 pictures.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from PIL import Image

API_ROOT = "https://api.pexels.com/v1/search"
LICENSE = "Pexels License (https://www.pexels.com/license/)"
USER_AGENT = "artflow-research/0.1 (personal research project)"

# Bounding box handed to the CDN.  The long side lands at 1792 px, so a picture
# with an aspect ratio up to 2.0 keeps a short side of at least 896 px — enough
# for every training bucket except the largest, which the rounder shapes reach.
CDN_BOX = "auto=compress&cs=tinysrgb&w=1792&h=1792"

# Search terms.  Grouped by what they are meant to bring in: named regions,
# garments that have their own name, and everyday scenes with people in them.
QUERIES: List[str] = [
    # Named regions, portrait framing.
    "east asian woman portrait", "east asian man portrait",
    "chinese woman portrait", "chinese man portrait",
    "japanese woman portrait", "korean woman portrait",
    "south asian woman portrait", "indian woman portrait",
    "indian man portrait", "pakistani woman portrait",
    "southeast asian woman", "vietnamese woman portrait",
    "thai woman traditional", "filipino woman portrait",
    "african woman portrait", "african man portrait",
    "nigerian woman portrait", "kenyan man portrait",
    "ethiopian woman portrait", "ghanaian man portrait",
    "middle eastern woman portrait", "arab man portrait",
    "turkish woman portrait", "persian woman portrait",
    "latin american woman", "mexican woman portrait",
    "brazilian man portrait", "peruvian woman portrait",
    "colombian woman portrait", "european woman portrait",
    "scandinavian man portrait", "slavic woman portrait",
    "elderly asian man portrait", "elderly black woman portrait",
    "elderly woman face closeup", "child portrait africa",
    "children asia village", "teenager portrait diverse",
    # Garments with their own name.
    "hanfu", "qipao", "cheongsam", "kimono", "hanbok", "sari",
    "salwar kameez", "ao dai", "abaya", "hijab", "kaftan", "thobe",
    "dashiki", "kente", "lederhosen", "dirndl", "flamenco dress",
    "mongolian traditional clothing", "tibetan traditional clothing",
    "chinese minority costume", "native american regalia",
    "african traditional attire", "indian wedding couple",
    "mexican traditional dress", "scottish kilt",
    # Everyday scenes, so the people in the corpus are not all studio shots.
    "street market vendor asia", "street market vendor africa",
    "farmer working field asia", "farmer working field africa",
    "craftsman workshop asia", "artisan woman weaving",
    "monk temple asia", "family portrait outdoors",
    "group of friends diverse", "students group diverse",
    "crowd street asia", "crowd street africa",
    "woman carrying basket village", "fisherman boat asia",
    "dancer traditional costume", "musician playing traditional instrument",
]

# A photograph is kept as a candidate when its own alt text says there is a
# person in it.  The alternative — captioning everything and dropping the rows
# whose caption mentions nobody — costs money on pictures we can rule out for
# free.
PERSON_WORDS = re.compile(
    r"\b(man|men|woman|women|person|people|girl|girls|boy|boys|child|children|"
    r"kid|kids|baby|toddler|guy|lady|ladies|male|female|model|portrait|"
    r"dancer|worker|farmer|fisherman|vendor|craftsman|artisan|family|couple|"
    r"bride|groom|grandmother|grandfather|mother|father|son|daughter|friend|"
    r"student|teacher|doctor|nurse|soldier|singer|athlete|musician|chef|monk|"
    r"priest|shopkeeper|trader|shepherd|villager|teen|teenager|adult|elder|"
    r"elderly|face|figure|portrait|self)\b",
    re.IGNORECASE,
)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(total=4, backoff_factor=1.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


class Pacer:
    """Keep API requests under ``per_hour`` using a sliding window."""

    def __init__(self, per_hour: int) -> None:
        self.per_hour = per_hour
        self.stamps: deque = deque()

    def wait(self) -> None:
        now = time.time()
        while self.stamps and now - self.stamps[0] > 3600:
            self.stamps.popleft()
        if len(self.stamps) >= self.per_hour:
            sleep_for = 3600 - (now - self.stamps[0]) + 1
            print(f"[pace] hourly budget spent, sleeping {sleep_for:.0f}s", flush=True)
            time.sleep(sleep_for)
            return self.wait()
        self.stamps.append(time.time())

    def note(self, seconds: float) -> None:
        """Charge a request that was issued ``seconds`` ago."""
        if self.stamps:
            self.stamps[-1] = time.time() - seconds


def search(session: requests.Session, key: str, query: str, page: int,
           per_page: int) -> tuple:
    response = session.get(
        API_ROOT, headers={"Authorization": key},
        params={"query": query, "per_page": per_page, "page": page},
        timeout=60)
    response.raise_for_status()
    return response.json(), response.headers


def thread_session() -> requests.Session:
    """One session per download worker, so connections are not shared."""
    local = getattr(thread_session, "_local", None)
    if local is None:
        local = threading.local()
        thread_session._local = local
    session = getattr(local, "session", None)
    if session is None:
        session = make_session()
        local.session = session
    return session


def download(session: requests.Session, url: str, dest: Path) -> Optional[tuple]:
    """Fetch one picture, save it as JPEG, and report its size."""
    if dest.exists() and dest.stat().st_size > 20000:
        try:
            with Image.open(dest) as existing:
                return existing.size
        except Exception:  # noqa: BLE001 - a broken leftover is refetched
            dest.unlink(missing_ok=True)
    try:
        response = session.get(url, timeout=120)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a failed picture is not fatal
        print(f"[img] {url} failed: {exc}", flush=True)
        return None
    try:
        with Image.open(io.BytesIO(response.content)) as image:
            image = image.convert("RGB")
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest, "JPEG", quality=92, optimize=True)
            return image.size
    except Exception as exc:  # noqa: BLE001
        print(f"[img] {url} undecodable: {exc}", flush=True)
        return None


def long_side_bounded(width: int, height: int, box: int = 1792) -> tuple:
    """Size the CDN will deliver for a ``w=box&h=box`` request."""
    long_side = max(width, height)
    if long_side <= box:
        return width, height
    scale = box / long_side
    return int(round(width * scale)), int(round(height * scale))


def load_done(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(int(json.loads(line)["photo_id"]))
            except Exception:  # noqa: BLE001 - a torn last line is expected
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--target", type=int, default=45000,
                        help="stop once this many candidates are on disk")
    parser.add_argument("--pages-per-query", type=int, default=8)
    parser.add_argument("--per-page", type=int, default=80)
    parser.add_argument("--min-side", type=int, default=896,
                        help="smallest short side worth keeping, after CDN scaling")
    parser.add_argument("--max-ratio", type=float, default=2.0,
                        help="long side / short side above which the picture is dropped")
    parser.add_argument("--per-hour", type=int, default=180,
                        help="API requests per hour to stay under")
    parser.add_argument("--workers", type=int, default=12,
                        help="concurrent image downloads")
    parser.add_argument("--queries", default=None,
                        help="file with one search term per line (default: built-in list)")
    parser.add_argument("--all-photos", action="store_true",
                        help="keep photographs whose alt text names no person")
    args = parser.parse_args()

    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise SystemExit("PEXELS_API_KEY is not set")

    out = Path(args.out)
    images_dir = out / "images"
    meta_path = out / "metadata.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    queries = QUERIES
    if args.queries:
        queries = [line.strip() for line in Path(args.queries).read_text().splitlines()
                   if line.strip() and not line.startswith("#")]

    session = make_session()
    pacer = Pacer(args.per_hour)
    done = load_done(meta_path)
    # Which pages each query has already been walked.  Search results come back
    # in a stable order, so one highest-page mark per query is enough: without
    # it a restart re-requests every page it has already seen, and the hourly
    # request budget goes on pages whose pictures are already on disk.
    state_path = out / "pages.json"
    fetched: Dict[str, int] = {}
    if state_path.exists():
        try:
            fetched = json.loads(state_path.read_text())
        except Exception:  # noqa: BLE001 - a damaged state file just re-walks
            fetched = {}
    print(f"[start] {len(done)} candidates already recorded, "
          f"{len(fetched)} queries already walked", flush=True)

    kept = len(done)
    written = 0
    requests_made = 0
    with meta_path.open("a", encoding="utf-8") as meta:
        for query in queries:
            if kept >= args.target:
                break
            for page in range(1, args.pages_per_query + 1):
                if kept >= args.target:
                    break
                if fetched.get(query, 0) >= page:
                    continue
                pacer.wait()
                started = time.time()
                try:
                    payload, headers = search(session, key, query, page, args.per_page)
                except Exception as exc:  # noqa: BLE001
                    print(f"[api] {query!r} page {page} failed: {exc}", flush=True)
                    time.sleep(5)
                    continue
                pacer.note(time.time() - started)
                requests_made += 1
                remaining = headers.get("x-ratelimit-remaining")
                photos = payload.get("photos") or []
                fetched[query] = max(fetched.get(query, 0), page)
                state_path.write_text(json.dumps(fetched))
                if not photos:
                    fetched[query] = args.pages_per_query
                    state_path.write_text(json.dumps(fetched))
                    break
                todo: List[tuple] = []
                for photo in photos:
                    photo_id = int(photo["id"])
                    if photo_id in done:
                        continue
                    width, height = int(photo["width"]), int(photo["height"])
                    ratio = max(width, height) / max(1, min(width, height))
                    scaled_w, scaled_h = long_side_bounded(width, height)
                    alt = (photo.get("alt") or "").strip()
                    person_hint = bool(PERSON_WORDS.search(alt))
                    record: Dict = {
                        "source": "pexels_people",
                        "source_id": f"pexels-{photo_id}",
                        "photo_id": photo_id,
                        "page_url": photo.get("url"),
                        "original_url": (photo.get("src") or {}).get("original"),
                        "alt": alt,
                        "photographer": " ".join((photo.get("photographer") or "").split()),
                        "photographer_url": photo.get("photographer_url"),
                        "avg_color": photo.get("avg_color"),
                        "source_width": width,
                        "source_height": height,
                        "query": query,
                        "person_hint": person_hint,
                        "license": LICENSE,
                    }
                    if ratio > args.max_ratio or min(scaled_w, scaled_h) < args.min_side:
                        record.update(download_ok=False,
                                      skip_reason=f"shape {width}x{height}")
                        meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(photo_id)
                        continue
                    if not person_hint and not args.all_photos:
                        record.update(download_ok=False, skip_reason="no person in alt text")
                        meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(photo_id)
                        continue
                    original = record["original_url"]
                    image_url = f"{original}?{CDN_BOX}"
                    dest = images_dir / f"pexels-{photo_id}.jpg"
                    record["image_url"] = image_url
                    record["local_path"] = str(dest)
                    todo.append((photo_id, record))
                if todo:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(download, thread_session(), record["image_url"], Path(record["local_path"])): (photo_id, record)
                            for photo_id, record in todo
                        }
                        for future in as_completed(futures):
                            photo_id, record = futures[future]
                            size = future.result()
                            if size is None:
                                record.update(download_ok=False, skip_reason="download failed")
                            else:
                                record.update(width=size[0], height=size[1], download_ok=True)
                                kept += 1
                            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                            done.add(photo_id)
                            written += 1
                            if written % 200 == 0:
                                meta.flush()
                                print(f"[progress] kept={kept} written={written} "
                                      f"query={query!r} page={page} api_left={remaining}",
                                      flush=True)
                meta.flush()
    print(f"[done] kept={kept} newly written={written} api_requests={requests_made}",
          flush=True)


if __name__ == "__main__":
    main()
