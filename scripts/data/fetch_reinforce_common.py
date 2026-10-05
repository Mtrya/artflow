"""Shared pieces for the d4-extra2 reinforcement fetchers.

``fetch_inat.py``, ``fetch_museum.py`` and ``fetch_commons.py`` each harvest
one concept-reinforcement batch from a source that needs no API key.  Unlike
``fetch_pexels.py``, whose query file budgets pages per term, these runs are
count-oriented: a query line is ``term | cap`` and a term is finished once it
has ``cap`` kept images or the source stops returning new results.

Shared here is what the three loops do the same way: the hourly pacer, the
HTTP session (one per download worker), the shape test, the JPEG q92 RGB save
(optionally bounded to a 1792 px box with LANCZOS), the ``term | cap`` parser,
the metadata/state readers and the record base fields.  Record layout, state
layout and CLI follow ``fetch_pexels.py``.
"""

from __future__ import annotations

import html
import io
import json
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from PIL import Image

# A descriptive User-Agent: Wikimedia requires one, and the other APIs use it
# to attribute the traffic.
USER_AGENT = ("artflow-research/0.1 (concept-reinforcement image harvest; "
              "personal research project)")

# Local counterpart of fetch_pexels' CDN_BOX.  The long side lands at 1792 px,
# so a picture with an aspect ratio up to 2.0 keeps a short side of at least
# 896 px — enough for every training bucket except the largest.
BOX = 1792
MIN_SIDE = 896
MAX_RATIO = 2.0
TAG_RE = re.compile(r"<[^>]+>")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(total=8, backoff_factor=2.0,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET",), respect_retry_after_header=True)
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


def get_json(session: requests.Session, url: str, params: Dict) -> dict:
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def long_side_bounded(width: int, height: int, box: int = BOX) -> tuple:
    """Size a ``box``-bounded request delivers for an image of this shape."""
    long_side = max(width, height)
    if long_side <= box:
        return width, height
    scale = box / long_side
    return int(round(width * scale)), int(round(height * scale))


def shape_ok(width: int, height: int, min_side: int = MIN_SIDE,
             max_ratio: float = MAX_RATIO, box: int = BOX) -> Tuple[bool, str]:
    """The fetch_pexels shape filter: aspect ratio and short side after bounding."""
    if not width or not height or width < 0 or height < 0:
        return False, f"shape {width}x{height}"
    ratio = max(width, height) / max(1, min(width, height))
    scaled_w, scaled_h = long_side_bounded(width, height, box)
    if ratio > max_ratio or min(scaled_w, scaled_h) < min_side:
        return False, f"shape {width}x{height}"
    return True, ""


def parse_queries(text: str) -> List[Tuple[str, Optional[int]]]:
    """Read the ``--queries`` file: ``term`` or ``term | cap`` per line."""
    queries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, _, cap = line.partition("|")
        term = term.strip()
        if not term:
            continue
        queries.append((term, int(cap.strip()) if cap.strip() else None))
    return queries


def load_metadata(path: Path) -> Tuple[Set[str], Counter]:
    """Source ids already recorded, and how many kept images per query.

    Ids are the prefixed ``source_id`` values, so a Met object and an AIC
    artwork that happen to share a number stay distinct.  The per-query counts
    are the run's progress ledger: they come from metadata.jsonl rather than
    the state file, so losing the state only costs re-walking a page, never the
    count a term's cap is measured against.
    """
    done: Set[str] = set()
    kept: Counter = Counter()
    if not path.exists():
        return done, kept
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn last line is expected
                continue
            source_id = record.get("source_id")
            if source_id:
                done.add(str(source_id))
            if record.get("download_ok"):
                kept[str(record.get("query") or "")] += 1
    return done, kept


def load_state(path: Path) -> Dict[str, dict]:
    """Per-term progress, keyed by query term."""
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a damaged state file just re-walks
        return {}
    return state if isinstance(state, dict) else {}


def save_state(path: Path, state: Dict[str, dict]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def strip_html(value: str) -> str:
    """Plain text from an API field that carries HTML markup."""
    text = TAG_RE.sub(" ", value or "")
    return " ".join(html.unescape(text).split())


def base_record(*, source: str, source_id: str, photo_id: int, page_url: str,
                original_url: str, alt: str, photographer: str = "",
                photographer_url: Optional[str] = None,
                source_width: Optional[int] = None,
                source_height: Optional[int] = None, query: str,
                license: str, title: str = "", artist: str = "") -> Dict:
    """The record fields every reinforcement fetcher shares.

    ``keep_hint`` is always True here: filtering happens at the source, so
    every row that reaches metadata.jsonl is meant to be kept.  A row is
    marked ``download_ok`` False with a ``skip_reason`` when the source-side
    filter still let something through (a licence the server filter missed, a
    shape that is too extreme, a missing image).
    """
    return {
        "source": source,
        "source_id": source_id,
        "photo_id": photo_id,
        "page_url": page_url,
        "original_url": original_url,
        "alt": alt,
        "photographer": photographer,
        "photographer_url": photographer_url,
        "source_width": source_width,
        "source_height": source_height,
        "query": query,
        "keep_hint": True,
        "license": license,
        "title": title,
        "artist": artist,
    }


def add_skip(record: Dict, reason: str) -> Dict:
    """Mark a candidate row as filtered out before or during download."""
    record["download_ok"] = False
    record["skip_reason"] = reason
    return record


def download_image(session: requests.Session, url: str, dest: Path,
                   box: Optional[int] = None, shape_filter: bool = False,
                   min_side: int = MIN_SIDE, max_ratio: float = MAX_RATIO) -> Dict:
    """Fetch one picture, save it as JPEG q92 RGB, and report its size.

    ``box`` bounds the saved image the way fetch_pexels' CDN box bounds the
    requested one: the long side lands at ``box`` px, aspect kept, LANCZOS.
    ``shape_filter`` applies the shape test to the decoded image, for sources
    that publish no dimensions before the download.  The result carries the
    decoded source size when a download actually happened; it is not set on
    the cached path, where only the saved size is known.
    """
    if dest.exists() and dest.stat().st_size > 20000:
        try:
            with Image.open(dest) as existing:
                width, height = existing.size
                if shape_filter:
                    ok, why = shape_ok(width, height, min_side, max_ratio)
                    if not ok:
                        return {"download_ok": False, "skip_reason": why}
                if box and max(width, height) > box:
                    bounded = existing.convert("RGB")
                    bounded.thumbnail((box, box), Image.LANCZOS)
                    bounded.save(dest, "JPEG", quality=92, optimize=True)
                    width, height = bounded.size
                return {"download_ok": True, "width": width, "height": height}
        except Exception:  # noqa: BLE001 - a broken leftover is refetched
            dest.unlink(missing_ok=True)
    try:
        response = session.get(url, timeout=120)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a failed picture is not fatal
        print(f"[img] {url} failed: {exc}", flush=True)
        return {"download_ok": False, "skip_reason": "download failed"}
    try:
        with Image.open(io.BytesIO(response.content)) as image:
            image = image.convert("RGB")
            source_width, source_height = image.size
            if shape_filter:
                ok, why = shape_ok(source_width, source_height, min_side, max_ratio)
                if not ok:
                    return {"download_ok": False, "skip_reason": why,
                            "source_width": source_width, "source_height": source_height}
            if box and max(source_width, source_height) > box:
                scale = box / max(source_width, source_height)
                image = image.resize(
                    (max(1, int(round(source_width * scale))),
                     max(1, int(round(source_height * scale)))), Image.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest, "JPEG", quality=92, optimize=True)
            return {"download_ok": True, "width": image.size[0], "height": image.size[1],
                    "source_width": source_width, "source_height": source_height}
    except Exception as exc:  # noqa: BLE001
        print(f"[img] {url} undecodable: {exc}", flush=True)
        return {"download_ok": False, "skip_reason": "download failed"}


def merge_download(record: Dict, result: Dict) -> None:
    """Apply a download result to a record, keeping source dims already known."""
    record["download_ok"] = bool(result.get("download_ok"))
    for key in ("width", "height"):
        if result.get(key) is not None:
            record[key] = result[key]
    if result.get("source_width") and not record.get("source_width"):
        record["source_width"] = result["source_width"]
        record["source_height"] = result["source_height"]
    if not record["download_ok"]:
        record["skip_reason"] = result.get("skip_reason") or "download failed"


def download_candidates(records, download, *, workers, max_successes):
    """Refill failed downloads from this page without exceeding the success cap.

    At most the remaining success budget is in flight. The caller records each
    yielded result before more work is submitted and retains the page cursor
    whenever its budget stops the page early.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    if workers < 1 or max_successes < 0:
        raise ValueError("workers must be positive and success budget nonnegative")
    pending = {}
    remaining = iter(records)
    successes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            while len(pending) < min(workers, max_successes - successes):
                record = next(remaining, None)
                if record is None:
                    break
                pending[pool.submit(download, record)] = record
            if not pending:
                return
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record = pending.pop(future)
                result = future.result()
                successes += bool(result.get("download_ok"))
                yield record, result
