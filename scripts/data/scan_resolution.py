#!/usr/bin/env python3
"""Scan image dimensions (header-only, no decode) across dataset roots and
report per-dataset resolution availability for 256/640/896/1024 training.

Usage (on the machine that holds the dataset roots):
    python scripts/data/scan_resolution.py \
        --roots rootA rootB ... --out data/meta/res_stats.jsonl --workers 8

Each root's basename is the dataset id; --datasets lets you override per-root
labels ("root1=d1" form). Output: one JSON line per dataset with
{dataset, total, unparsed, lt256, g256, g640, g896, g1024, min_median}.
"""
import argparse
import json
import os
import struct
from concurrent.futures import ProcessPoolExecutor

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
HEAD = 64 * 1024


def parse_jpeg(b):
    if b[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(b)
    while i + 4 <= n:
        if b[i] != 0xFF:
            i += 1
            continue
        marker = b[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return None
        seg_len = struct.unpack(">H", b[i + 2:i + 4])[0]
        if seg_len < 2:
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return (w, h)
        i += 2 + seg_len
    return None


def parse_png(b):
    if b[:8] != b"\x89PNG\r\n\x1a\n" or len(b) < 24:
        return None
    w, h = struct.unpack(">II", b[16:24])
    return (w, h)


def parse_webp(b):
    if len(b) < 30 or b[:4] != b"RIFF" or b[8:12] != b"WEBP":
        return None
    fmt = b[12:16]
    if fmt == b"VP8 ":
        if b[23:26] == b"\x9d\x01\x2a":
            w, h = struct.unpack("<HH", b[26:30])
            return (w & 0x3FFF, h & 0x3FFF)
    elif fmt == b"VP8L":
        if b[20] == 0x2F:
            val = int.from_bytes(b[21:25], "little")
            return ((val & 0x3FFF) + 1, ((val >> 14) & 0x3FFF) + 1)
    elif fmt == b"VP8X" and len(b) >= 30:
        return (int.from_bytes(b[24:27], "little") + 1,
                int.from_bytes(b[27:30], "little") + 1)
    return None


PARSERS = {".jpg": parse_jpeg, ".jpeg": parse_jpeg,
           ".png": parse_png, ".webp": parse_webp}


def detect_and_parse(head):
    if head.startswith(b"\xff\xd8"):
        return parse_jpeg(head)
    if head.startswith(b"\x89PNG"):
        return parse_png(head)
    if head.startswith(b"RIFF"):  # parse_webp verifies WEBP inside
        return parse_webp(head)
    return None


def scan_batch(paths):
    out = []
    for p in paths:
        wh = None
        try:
            with open(p, "rb") as f:
                wh = detect_and_parse(f.read(HEAD))
        except OSError:
            wh = None
        out.append(wh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="optional 'root=label' overrides, one per root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    labels = {}
    if args.labels:
        for kv in args.labels:
            r, l = kv.split("=", 1)
            labels[r] = l

    tasks = []  # (dataset, [paths])
    for root in args.roots:
        ds = labels.get(root) or os.path.basename(os.path.normpath(root))
        files = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in EXTS:
                    files.append(os.path.join(dirpath, fn))
        batches = [files[i:i + args.batch]
                   for i in range(0, len(files), args.batch)]
        tasks += [(ds, b) for b in batches]

    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for (ds, batch), whs in zip(tasks, ex.map(scan_batch,
                                                  [b for _, b in tasks])):
            st = stats.setdefault(ds, {"total": 0, "unparsed": 0,
                                       "lt256": 0, "g256": 0, "g640": 0,
                                       "g896": 0, "g1024": 0,
                                       "r_lt_0_25": 0, "r_0_25_0_5": 0,
                                       "r_0_5_0_75": 0, "r_0_75_1_33": 0,
                                       "r_1_33_2": 0, "r_2_4": 0,
                                       "r_gt_4": 0,
                                       "min_sum": 0, "n_min": 0})
            st["total"] += len(batch)
            for wh in whs:
                if wh is None:
                    st["unparsed"] += 1
                    continue
                w, h = wh
                m = min(w, h)
                st["lt256"] += m < 256
                st["g256"] += m >= 256
                st["g640"] += m >= 640
                st["g896"] += m >= 896
                st["g1024"] += m >= 1024
                st["min_sum"] += m
                st["n_min"] += 1
                r = w / h if h else 0.0
                if r < 0.25:
                    st["r_lt_0_25"] += 1
                elif r < 0.5:
                    st["r_0_25_0_5"] += 1
                elif r < 0.75:
                    st["r_0_5_0_75"] += 1
                elif r <= 1.3333:
                    st["r_0_75_1_33"] += 1
                elif r <= 2:
                    st["r_1_33_2"] += 1
                elif r <= 4:
                    st["r_2_4"] += 1
                else:
                    st["r_gt_4"] += 1

    with open(args.out, "w") as f:
        for ds in sorted(stats):
            st = stats[ds]
            st["dataset"] = ds
            st["min_median"] = (st["min_sum"] // st["n_min"]
                                if st["n_min"] else None)
            del st["min_sum"], st["n_min"]
            f.write(json.dumps(st) + "\n")
            print(ds, st["total"], "lt256", st["lt256"],
                  "| g256", st["g256"], "g640", st["g640"],
                  "g896", st["g896"], "g1024", st["g1024"],
                  "| ar<0.5", st["r_lt_0_25"] + st["r_0_25_0_5"],
                  "ar>2", st["r_2_4"] + st["r_gt_4"],
                  "| unparsed", st["unparsed"],
                  "| med_min", st["min_median"], flush=True)
    print("SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
