"""Shared helpers for stage-1 data fetchers.

All fetchers write to data/raw/<source>/:
  images/            downloaded image files named <source_id>.<ext>
  metadata.parquet   one row per record

Metadata columns (superset; fetchers fill what the source provides):
  source, source_id, title, artist, date, classification, department,
  license, image_url, local_path, width, height, download_ok
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = "artflow-research/0.1 (personal research project; contact: local)"


def workspace_root() -> str:
    """The shared-workspace root, from $ARTFLOW_ROOT.

    The path names a project and an account on the cluster, so it is taken from
    the environment rather than written into the source.  Every script here is
    run on the machine that holds the corpus, which already has it set.
    """
    root = os.environ.get("ARTFLOW_ROOT")
    if not root:
        raise SystemExit("set ARTFLOW_ROOT to the shared-workspace root")
    return root


def make_session(proxy: str | None = None) -> requests.Session:
    """Session with retries. Proxy: explicit arg, else $ARTFLOW_PROXY, else direct."""
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retries = Retry(
        total=4, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    p = proxy if proxy is not None else os.environ.get("ARTFLOW_PROXY")
    if p:
        s.proxies.update({"http": p, "https": p})
    return s


def get_json(session: requests.Session, url: str, params: dict | None = None,
             timeout: int = 30) -> dict:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_image(session: requests.Session, url: str, dest: Path,
                   timeout: int = 60, min_bytes: int = 2000,
                   verify: bool = True) -> tuple[int | None, int | None]:
    """Download url -> dest. Returns (width, height) or (None, None) on failure.

    Skips download if dest already exists and is a valid image.
    """
    from PIL import Image

    if dest.exists() and dest.stat().st_size >= min_bytes:
        if verify:
            try:
                with Image.open(dest) as im:
                    return im.size
            except Exception:
                dest.unlink()
        else:
            return None, None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = session.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(1 << 16):
            buf.write(chunk)
            if buf.tell() > 200 * 1024 * 1024:  # 200MB sanity cap
                return None, None
        data = buf.getvalue()
    except Exception:
        return None, None
    if len(data) < min_bytes:
        return None, None
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        return None, None
    dest.write_bytes(data)
    return w, h


def save_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def phash_dedup_report(image_paths: list[Path], hash_size: int = 16,
                       max_hamming: int = 6) -> dict:
    """Compute phashes and near-duplicate rate over a set of images.

    Returns {n_hashed, n_duplicates, dup_rate, hashes: {path_str: hex}}.
    Two images are near-duplicates if hamming distance <= max_hamming.
    O(n^2) in a cheap integer loop — fine for probe scale (~1K), do not
    use for the full harvest without indexing.
    """
    import imagehash
    from PIL import Image

    hashes: dict[str, int] = {}
    for p in image_paths:
        try:
            with Image.open(p) as im:
                hashes[str(p)] = int(str(imagehash.phash(im, hash_size=hash_size)), 16)
        except Exception:
            continue
    vals = list(hashes.values())
    dup = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if bin(vals[i] ^ vals[j]).count("1") <= max_hamming:
                dup += 1
                break
    n = len(vals)
    return {
        "n_hashed": n,
        "n_duplicates": dup,
        "dup_rate": (dup / n) if n else 0.0,
        "hashes": hashes,
    }


def rate_sleep(seconds: float) -> None:
    time.sleep(seconds)
