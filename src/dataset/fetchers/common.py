"""Shared utilities for museum data fetchers.

Common pieces: HTTP session with retries, atomic-ish image saving,
and append-only JSONL metadata writing (idempotent re-runs).
"""

import json
import os
import time
from typing import Dict, Optional

import requests

DEFAULT_UA = "artflow-fetcher/0.1 (research dataset curation)"


def make_session(max_retries: int = 4, backoff: float = 2.0) -> requests.Session:
    """Create a requests session with a browser-ish UA; retries handled by caller loop."""
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})
    session._max_retries = max_retries
    session._backoff = backoff
    return session


def request_with_retry(session: requests.Session, method: str, url: str, **kwargs) -> Optional[requests.Response]:
    """Request with simple retry/backoff. Returns None after exhausting retries."""
    kwargs.setdefault("timeout", (10, 60))
    for attempt in range(session._max_retries):
        try:
            resp = session.request(method, url, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(session._backoff * (attempt + 1))
                continue
            return resp
        except requests.RequestException:
            time.sleep(session._backoff * (attempt + 1))
    return None


def save_jpeg(content: bytes, path: str) -> bool:
    """Save image bytes to path, skipping if the file already exists (resume-safe)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, path)
    return True


class JsonlWriter:
    """Append-only JSONL writer; tracks written keys to stay idempotent across re-runs."""

    def __init__(self, path: str, key_field: str = "image_id"):
        self.path = path
        self.key_field = key_field
        self.seen = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.seen.add(json.loads(line)[key_field])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def write(self, record: Dict) -> bool:
        key = record.get(self.key_field)
        if key in self.seen:
            return False
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.seen.add(key)
        return True
