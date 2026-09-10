"""Fetch National Palace Museum (Taipei) open-data Chinese paintings.

Source: https://digitalarchive.npm.gov.tw/opendata/ (mirror: theme.npm.edu.tw/opendata/)

Discovery: POST /opendata/Pub/Search with JSON model, RegisterType="繪畫",
PageInfo.PageSize up to 100 -> HTML cards with id, dep, GetImage preview URL, title.
Detail: GET /opendata/Pub/Detail/<id>?dep=<dep>&mode=full -> full metadata table
(品名/作者/分類/主題/技法...). Images: /opendata/Image/GetImage?imageId=..&randomCode=..
serves ~600px previews without auth. The official 1MP (CC0) / 6MP (CC BY 4.0)
download endpoints (/opendata/ImageDownload/Download100|600) are captcha-gated;
the IIIF manifest endpoints (/Antique/setJson*) currently return HTTP 500.

License note: the site's open-data declaration releases low/mid-tier images for
unrestricted use without application; the 1MP tier is explicitly CC0 and the 6MP
tier CC BY 4.0. We record license="CC0" for the <=1MP web tier fetched here.

Usage: python fetch_npm_tw.py [--out data/raw/npm_tw] [--limit 1000]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import download_image, make_session, phash_dedup_report, save_parquet

BASE = "https://digitalarchive.npm.gov.tw"
PROXY = "http://127.0.0.1:7897"
CATEGORY = "繪畫"
PAGE_SIZE = 100

# dynasty/era prefixes seen in NPM painting titles
DYNASTIES = [
    "六朝梁", "六朝齊", "六朝陳", "六朝", "五代後蜀", "五代南唐", "五代後梁",
    "五代後唐", "五代後晉", "五代", "北宋", "南宋", "宋", "遼", "金", "元",
    "明", "清", "民國", "唐", "隋", "漢", "魏晋", "魏", "晉",
]
# NPM theme taxonomy first-level values (人物 doubles as anatomy data)
THEMES = ["人物", "山水", "花草", "花鳥", "翎毛", "走獸", "草蟲", "蟲魚",
          "樹木", "建築", "器用", "船", "佛道人物", "仕女"]

THEME_ROW_RE = re.compile(
    r"<td>(主要主題|次要主題|其他主題)</td>\s*<td>([^<]+)</td>", re.S)

_sess = {"s": None, "proxy": False}


def _new_session(proxy: bool):
    return make_session(proxy=PROXY if proxy else None)


def robust_get(url: str, tries: int = 8, **kw):
    """GET that flips between direct and local proxy on connection/TLS errors."""
    last = None
    for i in range(tries):
        if _sess["s"] is None:
            _sess["s"] = _new_session(_sess["proxy"])
        try:
            r = _sess["s"].get(url, timeout=45, **kw)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:  # SSLError/ConnectionError/Timeout/Chunked
            last = e
            _sess["proxy"] = not _sess["proxy"]
            _sess["s"] = None
        time.sleep(min(1 + i, 5))
    raise last


def search_page(page_index: int) -> str:
    model = {
        "RegisterType": CATEGORY, "IndexYear": None,
        "WestBeginYear": 0, "WestEndYear": 0, "YearDisplay": None,
        "SearchContent": None, "RegisterTypeEng": None,
        "PageInfo": {"PageIndex": page_index, "PageSize": PAGE_SIZE, "PageCount": 1},
    }
    payload = json.dumps(model)
    last = None
    for i in range(8):
        if _sess["s"] is None:
            _sess["s"] = _new_session(_sess["proxy"])
        try:
            r = _sess["s"].post(
                f"{BASE}/opendata/Pub/Search", data=payload, timeout=60,
                headers={"Content-Type": "application/json",
                         "X-Requested-With": "XMLHttpRequest",
                         "Referer": f"{BASE}/opendata/"},
                allow_redirects=False)
            if r.status_code == 200 and "card-item" in r.text:
                return r.text
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last = e
            _sess["proxy"] = not _sess["proxy"]
            _sess["s"] = None
        time.sleep(min(1 + i, 5))
    raise last


CARD_RE = re.compile(
    r"onclick=\"Detail\('(\d+)', '([A-Z])'\).*?"
    r"/opendata/Image/GetImage\?imageId=(\d+)&randomCode=(\d+).*?"
    r'<div class="card-title">(.*?)</div>', re.S)


def parse_cards(html: str) -> list[dict]:
    return [{"id": m[0], "dep": m[1], "image_id": m[2], "code": m[3],
             "card_title": re.sub(r"\s+", " ", m[4]).strip()}
            for m in CARD_RE.findall(html)]


def total_pages(html: str) -> int:
    m = re.search(r'"PageCount":(\d+)', html)
    return int(m.group(1)) if m else 0


TD_RE = re.compile(r"<td>(.*?)</td>\s*<td>(.*?)</td>", re.S)


def _clean(v: str) -> str:
    v = re.sub(r"<br\s*/?>", " | ", v)
    v = re.sub(r"<[^>]+>", " ", v)
    return re.sub(r"\s+", " ", v).strip(" |")


def parse_detail(html: str) -> dict:
    out = {"accession": "", "title": "", "title_en": "", "artist": "",
           "classification": "", "themes": [], "techniques": []}
    rows = TD_RE.findall(html)
    for k, v in rows:
        k = re.sub(r"\s+", " ", k).strip()
        if k == "文物統一編號":
            out["accession"] = _clean(v)
        elif k == "品名":
            parts = [p.strip() for p in _clean(v).split(" | ")]
            out["title"] = parts[0]
            if len(parts) > 1:
                out["title_en"] = parts[1]
        elif k == "作者":
            out["artist"] = _clean(v).split(" | ")[0].strip()
        elif k == "分類":
            out["classification"] = _clean(v)
    # 主題 table: rows of <td>主要主題|次要主題|其他主題</td><td>山水</td>...
    # primary subgenre = the 主要主題 first-level value; keep all, ordered.
    primary, secondary = [], []
    for kind, val in THEME_ROW_RE.findall(html):
        val = val.strip()
        if val not in THEMES:
            continue
        (primary if kind == "主要主題" else secondary).append(val)
    seen, themes = set(), []
    for v in primary + secondary:
        if v not in seen:
            seen.add(v)
            themes.append(v)
    out["themes"] = themes
    # 技法 pane: keyword scan within the techniques tab section
    m2 = re.search(r'id="details-7"(.{0,4000}?)(?:id="details-8"|參考資料)', html, re.S)
    tech_frag = m2.group(1) if m2 else ""
    out["techniques"] = [t for t in ["界畫", "雙鉤", "白描", "沒骨", "寫意", "工筆"]
                         if t in tech_frag]
    return out


def dynasty_from_title(title: str) -> str:
    for d in DYNASTIES:
        if title.startswith(d):
            return d
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/npm_tw")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    det_dir = out / "_detail"
    det_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out / "_csv"  # kept for parity with the brief; no bulk CSV exists
    csv_dir.mkdir(exist_ok=True)

    # 1) discovery: spread search pages evenly across the full result set
    first = search_page(1)
    tp = total_pages(first)
    total_records = tp * PAGE_SIZE
    print(f"category={CATEGORY} pages={tp} (~{total_records} records)")
    n_pages = max(1, (args.limit + PAGE_SIZE - 1) // PAGE_SIZE)
    step = max(1, tp // n_pages)
    pages = sorted({1 + i * step for i in range(n_pages)} | {1})
    cards: list[dict] = parse_cards(first)
    for p in pages:
        if p == 1:
            continue
        if len(cards) >= args.limit + PAGE_SIZE:
            break
        try:
            cards.extend(parse_cards(search_page(p)))
        except Exception as e:
            print(f"search page {p} failed: {e}")
        time.sleep(args.sleep)
    # dedupe by (id, dep), keep site order
    seen, records = set(), []
    for c in cards:
        key = (c["id"], c["dep"])
        if key not in seen:
            seen.add(key)
            records.append(c)
    records = records[: args.limit]
    print(f"sampled {len(records)} records from {len(pages)} pages")

    # 2) detail metadata + 3) image download
    rows = []
    t0 = time.time()
    n_dl = 0
    for i, rec in enumerate(records):
        sid = f"{rec['dep']}-{rec['id']}"
        det_path = det_dir / f"{rec['dep']}_{rec['id']}.html"
        if det_path.exists():
            html = det_path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                html = robust_get(
                    f"{BASE}/opendata/Pub/Detail/{rec['id']}?dep={rec['dep']}&mode=full"
                ).text
                det_path.write_text(html, encoding="utf-8")
            except Exception as e:
                print(f"detail {sid} failed: {e}")
                html = ""
            time.sleep(args.sleep)
        meta = parse_detail(html) if html else {
            "accession": "", "title": "", "title_en": "", "artist": "",
            "classification": "", "themes": [], "techniques": []}
        title = meta["title"] or rec["card_title"]
        dynasty = dynasty_from_title(title)
        tags = meta["themes"]
        classification = meta["classification"] or CATEGORY
        if tags:
            classification = classification + ";" + ";".join(tags)

        image_url = (f"{BASE}/opendata/Image/GetImage?imageId={rec['image_id']}"
                     f"&randomCode={rec['code']}")
        dest = img_dir / f"{sid}.jpg"
        w, h = download_image(_sess["s"] if _sess["s"] else _new_session(False),
                              image_url, dest)
        if w is None:  # retry once through the other route
            _sess["proxy"] = not _sess["proxy"]
            _sess["s"] = None
            _sess["s"] = _new_session(_sess["proxy"])
            w, h = download_image(_sess["s"], image_url, dest)
        ok = w is not None
        n_dl += ok
        rows.append({
            "source": "npm_tw",
            "source_id": sid,
            "title": title,
            "artist": meta["artist"],
            "date": dynasty,
            "classification": classification,
            "department": "書畫" if rec["dep"] == "P" else rec["dep"],
            "license": "CC0",
            "image_url": image_url,
            "local_path": str(dest) if ok else "",
            "width": w or 0,
            "height": h or 0,
            "download_ok": ok,
        })
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(records)} dl_ok={n_dl} "
                  f"({n_dl/el:.2f} img/s)")
        time.sleep(args.sleep)

    save_parquet(rows, out / "metadata.parquet")
    df = pd.DataFrame(rows)
    print(f"\nrows={len(df)} download_ok={df.download_ok.sum()} "
          f"({df.download_ok.mean():.1%})")
    for col in ["title", "artist", "date", "classification"]:
        print(f"  {col} coverage: {(df[col].astype(str).str.len() > 0).mean():.1%}")
    ok = df[df.download_ok]
    if len(ok):
        print(f"  size: median {ok.width.median():.0f}x{ok.height.median():.0f}, "
              f">=1024px both: {((ok.width >= 1024) & (ok.height >= 1024)).mean():.1%}")
    # phash on up to 300 downloaded images
    paths = [Path(p) for p in ok.local_path][:300]
    if paths:
        rep = phash_dedup_report(paths)
        print(f"  phash: n={rep['n_hashed']} dups={rep['n_duplicates']} "
              f"rate={rep['dup_rate']:.3f}")


if __name__ == "__main__":
    main()
