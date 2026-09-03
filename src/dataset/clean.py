"""Cleaning pipeline for museum scans (borders, mounting margins, documentation photos).

Rules, developed against NPM-TW IIIF scans (see notes/dataset_plan.md Domain 1):

1. chart recovery: documentation shots carry a color-calibration chart (a compact
   cluster of vivid saturated patches sitting on the black studio background, near an
   edge). If the chart can be isolated, crop that side away and KEEP the scroll — the
   painting is fully visible in these shots. Reject only when the chart overlaps the
   central artwork region or almost nothing remains.
2. detect_mount_border: per-edge inward walk — a row/column belongs to the border while
   ≥`fill` of its pixels are within `bg_tol` of that edge's own border color (estimated
   from the outermost strip). Handles cream mounts, gold strips, black doc background.
   Keeps 詩堂 (calligraphy headers) — they are part of the artwork. Applied in multiple
   passes: after the outer background is gone, accession-label strips become the new
   edge and are removed by the next pass.
3. junk residual policy: what rules cannot confidently classify (rolled scrolls, odd
   views) is kept and left to the VLM `view_type` tag at caption time; the assembly
   step filters on that tag.
4. view tagging: canvas label suffix (PAA, PAB, ...) becomes view_index; rejects
   override any suffix.

CLI:
    python -m src.dataset.clean --in data/raw/npm_tw --out data/clean/npm_tw
"""

import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Museum scans legitimately exceed PIL's default ~178MP decompression-bomb
# guard (FSG handscrolls run to 490MP); the unreadable-file guard in
# clean_image still catches genuinely corrupt files.
Image.MAX_IMAGE_PIXELS = None

VIEW_RE_SUFFIX = 3  # canvas labels end like "...PAA"


def _to_hsv(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rough HSV saturation/value planes from an RGB uint8 array (fast, no cv2)."""
    a = arr.astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    diff = mx - mn
    sat = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0)
    return sat, mx


def black_fraction(arr: np.ndarray, black_thresh: int = 40) -> float:
    gray = arr.astype(np.float32).mean(axis=2)
    return float((gray < black_thresh).mean())


def _hue_degrees(arr: np.ndarray) -> np.ndarray:
    """Hue plane (0-360) from an RGB uint8 array."""
    a = arr.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    mr = m & (mx == r)
    mg = m & (mx == g) & ~mr
    mb = m & ~mr & ~mg
    h[mr] = ((g - b)[mr] / d[mr]) % 6
    h[mg] = (b - r)[mg] / d[mg] + 2
    h[mb] = (r - g)[mb] / d[mb] + 4
    return (h * 60) % 360


def _components(grid: np.ndarray) -> list:
    """4-connected components of a boolean grid; yields lists of (y, x) cells."""
    seen = np.zeros_like(grid, dtype=bool)
    comps = []
    for sy, sx in zip(*np.nonzero(grid)):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] \
                        and grid[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def _ring_pixels(gray: np.ndarray, box: Tuple[int, int, int, int], pad_b: int,
                 bh: int, bw: int, blocks: int) -> np.ndarray:
    """Gray values of the frame AROUND a grid bbox (the bbox interior excluded)."""
    y0, y1, x0, x1 = box
    ry0, ry1 = max(0, y0 - pad_b) * bh, min(blocks, y1 + pad_b) * bh
    rx0, rx1 = max(0, x0 - pad_b) * bw, min(blocks, x1 + 1 + pad_b) * bw
    ix0, ix1, iy0, iy1 = x0 * bw, x1 * bw, y0 * bh, y1 * bh
    slabs = [gray[ry0:iy0, rx0:rx1], gray[iy1:ry1, rx0:rx1],
             gray[max(ry0, iy0):min(ry1, iy1), rx0:ix0],
             gray[max(ry0, iy0):min(ry1, iy1), ix1:rx1]]
    flat = [s.ravel() for s in slabs if s.size]
    return np.concatenate(flat) if flat else np.empty(0)


def detect_chart(arr: np.ndarray, vivid_sat: float = 0.45, vivid_val: float = 0.35,
                 blocks: int = 24, block_frac: float = 0.05, min_blocks: int = 1,
                 max_area_frac: float = 0.06, margin: float = 0.25,
                 ring_med_max: float = 110.0, ring_uniform_min: float = 0.55,
                 min_hue_bins: int = 4, dark_thresh: int = 70) -> Optional[Tuple[int, int, int, int]]:
    """Locate a color-calibration chart; returns its bbox (l, t, r, b) or None.

    A chart is a compact connected cluster of grid blocks dense in vivid pixels with
    high hue diversity (red/green/blue/yellow patches), sitting on uniform dark studio
    background near the image edge. Brown ink paintings are vivid in the saturation
    sense but monohue and spatially spread (their clusters exceed max_area_frac); red
    seals are compact but monohue; colorful regions inside a painting fail the ring
    test because their surroundings are varied, not uniform dark. The ring test uses
    the median + a uniformity fraction so a white ruler/label next to the chart does
    not defeat it.
    """
    gray = arr.astype(np.float32).mean(axis=2)
    if float((gray < dark_thresh).mean()) < 0.10:
        return None
    sat, val = _to_hsv(arr)
    vivid = (sat > vivid_sat) & (val > vivid_val)
    h, w = vivid.shape
    bh, bw = h // blocks, w // blocks
    if bh == 0 or bw == 0:
        return None
    grid = vivid[:bh * blocks, :bw * blocks].reshape(blocks, bh, blocks, bw).mean(axis=(1, 3)) > block_frac
    for comp in _components(grid):
        if len(comp) < min_blocks:
            continue
        ys = [c[0] for c in comp]
        xs = [c[1] for c in comp]
        y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
        if (y1 - y0) * bh * (x1 - x0) * bw > max_area_frac * h * w:
            continue
        # candidate pixel bbox, expanded by one block to cover the whole chart card
        px0, px1 = max(0, x0 - 1) * bw, min(blocks, x1 + 1) * bw
        py0, py1 = max(0, y0 - 1) * bh, min(blocks, y1 + 1) * bh
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        if not (cx < w * margin or cx > w * (1 - margin) or cy < h * margin or cy > h * (1 - margin)):
            continue
        ring = _ring_pixels(gray, (y0, y1, x0, x1), 2, bh, bw, blocks)
        if ring.size:
            med = float(np.median(ring))
            if med > ring_med_max or float((np.abs(ring - med) < 25).mean()) < ring_uniform_min:
                continue
        region = arr[py0:py1, px0:px1]
        rsat, rval = _to_hsv(region)
        rmag = (rsat > vivid_sat) & (rval > vivid_val)
        if rmag.sum() < 20:
            continue
        hues = _hue_degrees(region)[rmag]
        if len(np.unique((hues // 30).astype(int))) < min_hue_bins:
            continue
        return (int(px0), int(py0), int(px1), int(py1))
    return None


def _chart_cut_bbox(h: int, w: int, chart: Tuple[int, int, int, int],
                    pad_frac: float = 0.01) -> Optional[Tuple[int, int, int, int]]:
    """Content bbox after cropping away the chart side. None if chart is central."""
    l, t, r, b = chart
    cx, cy = (l + r) / 2, (t + b) / 2
    # chart overlapping the central region cannot be isolated
    if w * 0.3 < cx < w * 0.7 and h * 0.3 < cy < h * 0.7:
        return None
    px, py = int(w * pad_frac), int(h * pad_frac)
    dists = {"l": cx, "r": w - cx, "t": cy, "b": h - cy}
    side = min(dists, key=dists.get)
    if side == "l":
        return (min(w, r + px), 0, w, h)
    if side == "r":
        return (0, 0, max(0, l - px), h)
    if side == "t":
        return (0, min(h, b + py), w, h)
    return (0, 0, w, max(0, t - py))


def cut_chart_side(arr: np.ndarray, chart: Tuple[int, int, int, int],
                   pad_frac: float = 0.01) -> Optional[np.ndarray]:
    """Crop away the side of the image carrying the chart. None if chart is central."""
    h, w = arr.shape[:2]
    bbox = _chart_cut_bbox(h, w, chart, pad_frac)
    if bbox is None:
        return None
    l, t, r, b = bbox
    return arr[t:b, l:r]


def detect_mount_border(arr: np.ndarray, bg_tol: float = 18.0, fill: float = 0.90,
                        max_frac: float = 0.20, strip: int = 8) -> Tuple[int, int, int, int]:
    """Estimate the content bbox by per-edge inward walk. Returns (l, t, r, b)."""
    a = arr.astype(np.float32)
    h, w = a.shape[:2]

    def edge_bg(region: np.ndarray) -> np.ndarray:
        return np.median(region.reshape(-1, 3), axis=0)

    def walk(line_is_bg: np.ndarray, limit: int) -> int:
        n = 0
        for flag in line_is_bg[:limit]:
            if not flag:
                break
            n += 1
        return n

    # left/right edges: per-column bg test against that side's strip color
    left_bg = edge_bg(a[:, :strip])
    right_bg = edge_bg(a[:, -strip:])
    col_sim_l = (np.abs(a - left_bg).max(axis=2) < bg_tol).mean(axis=0) >= fill
    col_sim_r = (np.abs(a - right_bg).max(axis=2) < bg_tol).mean(axis=0) >= fill
    top_bg = edge_bg(a[:strip, :])
    bottom_bg = edge_bg(a[-strip:, :])
    row_sim_t = (np.abs(a - top_bg).max(axis=2) < bg_tol).mean(axis=1) >= fill
    row_sim_b = (np.abs(a - bottom_bg).max(axis=2) < bg_tol).mean(axis=1) >= fill

    lim_w, lim_h = int(w * max_frac), int(h * max_frac)
    wl = walk(col_sim_l, lim_w)
    wr = walk(col_sim_r[::-1], lim_w)
    wt = walk(row_sim_t, lim_h)
    wb = walk(row_sim_b[::-1], lim_h)
    # uniform or near-uniform image: no content boundary on any side — refuse to crop
    if wl == lim_w and wr == lim_w and wt == lim_h and wb == lim_h:
        return (0, 0, w, h)
    # a single side at the limit just means a wide border (e.g. archival black
    # background); clamping there is conservative and safe
    l, r, t, b = wl, w - wr, wt, h - wb
    if r - l < w * 0.3 or b - t < h * 0.3:
        return (0, 0, w, h)  # degenerate: refuse to crop
    return (l, t, r, b)


def crop_borders(im: Image.Image, max_passes: int = 6) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Multi-pass border crop; returns cropped image and bbox in original coords.

    Several passes are needed because a very wide border (archival black background)
    is clamped at `max_frac` per pass, and accession-label strips only become the
    outer edge after the true background is gone.
    """
    total = (0, 0, im.width, im.height)
    for _ in range(max_passes):
        arr = np.array(im)
        l, t, r, b = detect_mount_border(arr)
        if (l, t, r, b) == (0, 0, im.width, im.height):
            break
        im = im.crop((l, t, r, b))
        total = (total[0] + l, total[1] + t, total[0] + r, total[1] + b)
    return im, total


def detect_edge_charts(arr: np.ndarray, band: float = 0.12, vivid_sat: float = 0.45,
                       vivid_val: float = 0.35, min_vivid_frac: float = 0.04,
                       min_hue_bins: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """Residual chart pass for AFTER border cropping, for charts on light mounts.

    detect_chart only runs on dark studio backgrounds (its ring test assumes
    black), and component clustering fails on cream/sepia mounts because the
    mount itself is "vivid" (sat>0.45) and merges everything into one giant
    component. Instead, scan the four outer strips directly: a junk strip
    (color chart + ruler + accession label) is dense in vivid pixels with high
    hue diversity (5+ distinct hue bins); painting edges rarely combine both.
    Returns the strip bbox (l, t, r, b) for the caller to cut away, or None.
    """
    h, w = arr.shape[:2]
    hues = _hue_degrees(arr)
    sat, val = _to_hsv(arr)
    vivid = (sat > vivid_sat) & (val > vivid_val)
    bt, bb = int(h * band), int(h * (1 - band))
    bl, br = int(w * band), int(w * (1 - band))
    strips = {
        "b": (0, bb, w, h),
        "t": (0, 0, w, bt),
        "l": (0, 0, bl, h),
        "r": (br, 0, w, h),
    }
    best = None
    for side, (l, t, r, b) in strips.items():
        # split the strip into quarters along its long axis so a small corner
        # chart is not diluted by the empty rest of the strip
        segs = []
        if side in ("t", "b"):
            for q in range(4):
                ql = l + (r - l) * q // 4
                qr = l + (r - l) * (q + 1) // 4
                segs.append((ql, t, qr, b))
        else:
            for q in range(4):
                qt = t + (b - t) * q // 4
                qb = t + (b - t) * (q + 1) // 4
                segs.append((l, qt, r, qb))
        hit = False
        for sl, st, sr, sb in segs:
            sv = vivid[st:sb, sl:sr]
            frac = float(sv.mean())
            if frac < min_vivid_frac:
                continue
            sh = hues[st:sb, sl:sr][sv]
            nb = len(np.unique((sh // 30).astype(int)))
            if nb >= min_hue_bins or (nb >= min_hue_bins - 1 and frac >= 2 * min_vivid_frac):
                hit = True
                break
        if not hit:
            continue
        if best is None:
            best = (l, t, r, b)
    return best


def view_tag(canvas_label: str) -> int:
    """View index from canvas label suffix (PAA->0, PAB->1, ...). -1 if unknown."""
    m = canvas_label[-VIEW_RE_SUFFIX:] if canvas_label else ""
    if len(m) == 3 and m[0] == "P" and m[1] == "A" and "A" <= m[2] <= "Z":
        return ord(m[2]) - ord("A")
    return -1


def clean_image(path: str, out_path: str, max_analyze_side: int = 4096) -> Dict:
    """Clean one image. Returns a record dict with reject/crop/view info.

    Chart/border analysis runs on a proxy capped at max_analyze_side px (FSG
    handscrolls reach 490MP — full-res numpy passes would take minutes and
    multiple GB each); crops are applied at full resolution with bboxes
    scaled back. The detectors are resolution-independent (proportional
    edge walks, 24x24 block grids), so results are equivalent.
    """
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        # Corrupt/truncated/non-image download (e.g. error page saved as .jpg).
        return {"orig_width": 0, "orig_height": 0, "rejected": True,
                "reject_reason": "unreadable"}
    w0, h0 = im.size
    rec: Dict = {"orig_width": w0, "orig_height": h0, "rejected": False, "reject_reason": ""}
    scale = max(1.0, max(w0, h0) / max_analyze_side)
    if scale > 1.0:
        proxy = im.resize((max(1, round(w0 / scale)), max(1, round(h0 / scale))), Image.BILINEAR)
    else:
        proxy = im
    arr = np.array(proxy)
    ph, pw = arr.shape[:2]
    if black_fraction(arr) >= 0.90:
        rec.update(rejected=True, reject_reason="mostly_black")
        return rec
    chart = detect_chart(arr)
    if chart is not None:
        rec["chart_bbox"] = [round(v * scale) for v in chart]
        bbox = _chart_cut_bbox(ph, pw, chart)
        if bbox is None or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) == 0:
            rec.update(rejected=True, reject_reason="chart_overlaps_art")
            return rec
        im = im.crop(tuple(min(round(v * scale), s) for v, s in zip(bbox, (w0, h0, w0, h0))))
        proxy = proxy.crop(bbox)
    proxy, total = crop_borders(proxy)
    # residual chart pass: charts sitting on light mounts evade detect_chart's
    # dark-background gate, so re-scan the border-cropped result per edge.
    for _ in range(2):  # at most: chart on one side + ruler strip w/ chart on another
        arr2 = np.array(proxy)
        ch = detect_edge_charts(arr2)
        if ch is None:
            break
        bbox = _chart_cut_bbox(arr2.shape[0], arr2.shape[1], ch)
        if bbox is None:
            break
        # safety: never sacrifice more than 40% of the current area in one cut
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 0.6 * arr2.shape[0] * arr2.shape[1]:
            break
        proxy = proxy.crop(bbox)
        total = (total[0] + bbox[0], total[1] + bbox[1],
                 total[0] + bbox[2], total[1] + bbox[3])
        rec["chart_bbox"] = [round(v * scale) for v in ch]
    rec["crop_bbox"] = [round(v * scale) for v in total]
    if scale > 1.0:
        im = im.crop(tuple(min(round(v * scale), s) for v, s in zip(total, (w0, h0, w0, h0))))
    else:
        im = proxy
    if im.width * im.height < 0.08 * w0 * h0:
        rec.update(rejected=True, reject_reason="content_too_small")
        return rec
    if out_path != os.devnull:  # devnull = dry-run probe, skip the write
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path, quality=95)
    rec["width"], rec["height"] = im.size
    return rec


def _clean_one(job):
    """Pool worker: clean one image, returning its record and result."""
    rec, out_path, dry_run = job
    result = clean_image(rec["local_path"], out_path if not dry_run else os.devnull)
    return rec, result


def run(in_dir: str, out_dir: str, dry_run: bool = False, jobs: int = 1) -> None:
    meta_path = os.path.join(in_dir, "metadata.jsonl")
    out_meta = os.path.join(out_dir, "metadata.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    stats = {"total": 0, "rejected": 0, "cropped": 0, "chart_cut": 0}
    written = set()
    if os.path.exists(out_meta) and not dry_run:
        with open(out_meta, encoding="utf-8") as f:
            for line in f:
                try:
                    written.add(json.loads(line)["image_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    pending = []
    with open(meta_path, encoding="utf-8") as f_in:
        for line in f_in:
            rec = json.loads(line)
            if rec["image_id"] not in written:
                pending.append(rec)

    def handle(rec: Dict, result: Dict, f_out) -> None:
        stats["total"] += 1
        out_path = os.path.join(out_dir, "images", f"{rec['image_id']}.jpg")
        rec.update(result)
        rec["view_index"] = view_tag(rec.get("canvas_label", ""))
        rec["local_path"] = out_path
        if result["rejected"]:
            stats["rejected"] += 1
            if os.path.exists(out_path) and not dry_run:
                os.remove(out_path)
        else:
            if result.get("chart_bbox"):
                stats["chart_cut"] += 1
            if result.get("crop_bbox") and result["crop_bbox"] != [0, 0, result["orig_width"], result["orig_height"]]:
                stats["cropped"] += 1
        if not dry_run:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if stats["total"] % 25 == 0:
                f_out.flush()

    with open(out_meta, "a", encoding="utf-8") as f_out:
        if jobs <= 1:
            for rec in pending:
                out_path = os.path.join(out_dir, "images", f"{rec['image_id']}.jpg")
                handle(rec, clean_image(rec["local_path"], out_path if not dry_run else os.devnull), f_out)
        else:
            work = [(rec, os.path.join(out_dir, "images", f"{rec['image_id']}.jpg"), dry_run)
                    for rec in pending]
            with Pool(jobs) as pool:
                for rec, result in pool.imap_unordered(_clean_one, work):
                    handle(rec, result, f_out)
    print(f"clean stats: {stats} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Clean museum scans (borders/doc photos)")
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (large scans are CPU-bound)")
    args = ap.parse_args()
    run(args.in_dir, args.out_dir, args.dry_run, args.jobs)


if __name__ == "__main__":
    main()
