#!/usr/bin/env python3
"""
Build unified precompute manifests from all data sources under the shared
workspace root (ARTFLOW_ROOT).

Each output row: {image_id, local_path, captions, width, height, bbox, source}
- local_path: absolute path under the shared workspace root
- captions:   list[str] (zh/en/etc., empty strings dropped)
- width/height: int or null (null -> resolution pre-filter skipped)
- bbox:       normalized [0,1000]^2 [x1,y1,x2,y2] or null (D1 artifact crops)

A stable hash carve-out (~eval_frac of every source) goes to light_eval.jsonl
and is excluded from the train manifests. light_eval is precomputed into its
own dataset for held-out validation during training.

Runs with stdlib + pyarrow only (no GPU environment needed).
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter

MUSEUM_SETS = [
    "met_impression", "met_portrait", "nga_impression", "nga_paintings_all",
    "rijksmuseum",
]
# + all met_imp_<artist> dirs (globbed)


def resolve_path(work_root: str, lp: str) -> str:
    if os.path.isabs(lp):
        return lp
    if lp.startswith("../"):
        return os.path.normpath(os.path.join(work_root, lp[3:]))
    return os.path.normpath(os.path.join(work_root, lp))


def is_eval(image_id: str, eval_frac: float) -> bool:
    h = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < eval_frac


def clean_caps(*caps):
    out = []
    seen = set()
    for c in caps:
        if isinstance(c, str):
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def norm_bbox(bb):
    """Return bbox as float list, or None."""
    if not bb or len(bb) != 4:
        return None
    try:
        return [float(v) for v in bb]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- sources

def src_d1(w):
    from src.dataset.ocr_merge import append_ocr_block, clean_ocr_lines

    p = os.path.join(w, "data/meta/d1/d1_metadata.jsonl")
    uncroppable = 0
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            bb = norm_bbox(d.get("bbox"))
            if d.get("artifacts") and bb is None:
                # Artifacts were seen but no box was produced for them.  The
                # row is still kept: dropping it would lose artwork to a
                # detector's failure to localise a colour chart.
                uncroppable += 1
            # The earlier labelling pass kept a transcription of the text in
            # the image in its own field, which nothing consumed.  Fold it into
            # both captions so the characters reach training.
            zh = d.get("caption_zh")
            en = d.get("caption_en")
            ocr = clean_ocr_lines(d.get("ocr_text") or "")
            if ocr:
                if zh:
                    zh = append_ocr_block(zh, ocr, "zh")
                if en:
                    en = append_ocr_block(en, ocr, "en")
            yield {
                "image_id": d["image_id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": clean_caps(zh, en),
                "width": None,
                "height": None,
                "bbox": bb,
                "source": "d1_" + d.get("source", "unknown"),
                # Kept so a caption can name the artist or the title when it
                # reads naturally.  The caption prompt is free to omit them.
                "artist": d.get("artist"),
                "title": d.get("title"),
            }
    print(f"  d1: artifacts seen but no box produced: {uncroppable} (kept)")


def src_d2_wikiart(w):
    p = os.path.join(w, "data/raw/wikiart215k/metadata.jsonl")
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if d.get("rejected"):
                continue
            caps = clean_caps(d.get("wikiart_caption"),
                              d.get("caption_florence_long"),
                              d.get("caption_florence"))
            if not caps:
                continue
            yield {
                "image_id": d["image_id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": caps,
                "width": None,
                "height": None,
                "bbox": None,
                "source": "d2_wikiart",
            }


def src_d2_museum(w):
    sets = list(MUSEUM_SETS)
    for d in sorted(os.listdir(os.path.join(w, "data/clean"))):
        if d.startswith("met_imp_"):
            sets.append(d)
    for s in sets:
        meta_p = os.path.join(w, f"data/clean/{s}/metadata.jsonl")
        label_files = sorted(glob.glob(os.path.join(w, f"data/labels/{s}/labels_*.jsonl")))
        if not os.path.exists(meta_p) or not label_files:
            print(f"  d2_museum: skip {s} (meta or labels missing)")
            continue
        labels = {}
        for lf in label_files:
            with open(lf) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("error"):
                        continue
                    lab = d.get("label")
                    if not isinstance(lab, dict):
                        continue
                    caps = clean_caps(lab.get("caption_zh"), lab.get("caption_en"))
                    if caps:
                        labels[d["image_id"]] = caps
        n_join = 0
        with open(meta_p) as f:
            for line in f:
                d = json.loads(line)
                if d.get("rejected"):
                    continue
                caps = labels.get(d["image_id"])
                if not caps:
                    continue
                n_join += 1
                yield {
                    "image_id": d["image_id"],
                    "local_path": resolve_path(w, d["local_path"]),
                    "captions": caps,
                    "width": d.get("width"),
                    "height": d.get("height"),
                    "bbox": None,  # clean images are already chart-cropped on disk
                    "source": f"d2_{s}",
                }
        print(f"  d2_museum: {s}: {n_join} joined")


def src_d3_human(w):
    p = os.path.join(w, "data/meta/d3/human_filtered.jsonl")
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            caps = clean_caps(d.get("caption_zh"), d.get("caption_en"))
            if not caps:
                continue
            yield {
                "image_id": d["image_id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": caps,
                "width": d.get("width"),
                "height": d.get("height"),
                "bbox": None,
                "source": "d3_human_recaption",
            }


def src_d3_people(w):
    p = os.path.join(w, "data/raw/people_supp/metadata.jsonl")
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if d.get("rejected"):
                continue
            caps = clean_caps(d.get("caption_en"))
            if not caps:
                continue
            yield {
                "image_id": d.get("image_id") or d["id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": caps,
                "width": None,
                "height": None,
                "bbox": None,
                "source": "d3_people_" + str(d.get("source", "supp")),
            }


def src_d3_pexels(w):
    """Photographs of people harvested from Pexels (see fetch_pexels.py).

    The uploader's own alt text becomes the row's short caption, so these rows
    enter the corpus with a human-written caption and pick up a generated one
    from the enrichment pass like every other source.
    """
    p = os.path.join(w, "data/raw/pexels_people/metadata.jsonl")
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("download_ok"):
                continue
            caps = clean_caps(d.get("alt"))
            if not caps:
                continue
            yield {
                "image_id": d["source_id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": caps,
                "width": d.get("width"),
                "height": d.get("height"),
                "bbox": None,
                "source": "d3_pexels",
            }


def _d4_from_metadata(w, sub, source_name):
    p = os.path.join(w, f"data/raw/{sub}/metadata.jsonl")
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if d.get("rejected"):
                continue
            caps = clean_caps(d.get("caption_zh"), d.get("caption_en"), d.get("caption"))
            if not caps:
                continue
            yield {
                "image_id": d.get("image_id") or d["id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": caps,
                "width": d.get("width"),
                "height": d.get("height"),
                "bbox": None,
                "source": source_name,
            }


def src_d4_vintage(w):
    yield from _d4_from_metadata(w, "vintage_photo", "d4_vintage")


def src_d4_relaion(w):
    yield from _d4_from_metadata(w, "relaion_art", "d4_relaion")


def src_d4_pd12m(w):
    import pyarrow.parquet as pq

    img_dir = os.path.join(w, "data/raw/pd12m/images")
    file_ids = {}
    for fn in os.listdir(img_dir):
        if fn.startswith("pd12m-") and fn.endswith(".jpg"):
            file_ids[fn[6:-4]] = os.path.join(img_dir, fn)
    print(f"  d4_pd12m: {len(file_ids)} files on disk")

    found = {}
    for pq_path in sorted(glob.glob(os.path.join(w, "data/meta/pd12m/pd12m.*.parquet"))):
        try:
            t = pq.read_table(pq_path, columns=["id", "caption", "width", "height"])
        except Exception as e:
            print(f"  d4_pd12m: skip corrupt {os.path.basename(pq_path)}: {e}")
            continue
        for pid, cap, pw, ph in zip(t.column("id").to_pylist(),
                                    t.column("caption").to_pylist(),
                                    t.column("width").to_pylist(),
                                    t.column("height").to_pylist()):
            if pid in file_ids and pid not in found and cap:
                found[pid] = (cap, pw, ph)
    print(f"  d4_pd12m: {len(found)} joined with captions")

    for pid, (cap, pw, ph) in found.items():
        yield {
            "image_id": f"pd12m-{pid}",
            "local_path": file_ids[pid],
            "captions": clean_caps(cap),
            "width": pw,
            "height": ph,
            "bbox": None,
            "source": "d4_pd12m",
        }

    # small re-download batch kept in raw/pd12m/metadata.jsonl
    extra_p = os.path.join(w, "data/raw/pd12m/metadata.jsonl")
    if os.path.exists(extra_p):
        n_extra = 0
        with open(extra_p) as f:
            for line in f:
                d = json.loads(line)
                if d.get("rejected"):
                    continue
                iid = d.get("image_id") or d["id"]
                pid = iid[6:] if iid.startswith("pd12m-") else iid
                if pid in found:
                    continue
                caps = clean_caps(d.get("caption"))
                if not caps:
                    continue
                n_extra += 1
                yield {
                    "image_id": iid,
                    "local_path": resolve_path(w, d["local_path"]),
                    "captions": caps,
                    "width": d.get("width"),
                    "height": d.get("height"),
                    "bbox": None,
                    "source": "d4_pd12m",
                }
        print(f"  d4_pd12m: +{n_extra} from metadata.jsonl re-download batch")


def _d4_extracted(w, sub, source_name):
    """Sources materialized by extract_parquet_images.py."""
    p = os.path.join(w, f"data/raw/{sub}/extracted/metadata.jsonl")
    if not os.path.exists(p):
        print(f"  {source_name}: {p} missing (run extract_parquet_images.py first), skipped")
        return
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            caps = clean_caps(d.get("caption"))
            if not caps:
                continue
            yield {
                "image_id": d["image_id"],
                "local_path": resolve_path(w, d["local_path"]),
                "captions": caps,
                "width": d.get("width"),
                "height": d.get("height"),
                "bbox": None,
                "source": source_name,
            }


def src_d4_zimage(w):
    yield from _d4_extracted(w, "z_image_turbo_gen", "d4_zimage")


def src_d4_megalith(w):
    yield from _d4_extracted(w, "megalith_opus", "d4_megalith_opus")


def src_d4_inat(w):
    yield from _d4_extracted(w, "inat_opus", "d4_inat_opus")


SOURCES = {
    "d1": src_d1,
    "d2_wikiart": src_d2_wikiart,
    "d2_museum": src_d2_museum,
    "d3_human": src_d3_human,
    "d3_people": src_d3_people,
    "d3_pexels": src_d3_pexels,
    "d4_vintage": src_d4_vintage,
    "d4_relaion": src_d4_relaion,
    "d4_pd12m": src_d4_pd12m,
    "d4_zimage": src_d4_zimage,
    "d4_megalith": src_d4_megalith,
    "d4_inat": src_d4_inat,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_root",
                    default=os.environ.get(
                        "ARTFLOW_ROOT"))
    ap.add_argument("--out_dir", default=None,
                    help="default: <work_root>/data/meta/precompute")
    ap.add_argument("--eval_frac", type=float, default=0.0015,
                    help="fraction hash-carved into light_eval.jsonl")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES.keys()))
    ap.add_argument("--append-eval", action="store_true",
                    help="append to light_eval.jsonl instead of truncating it; "
                         "use when rebuilding a subset of sources, after removing "
                         "that subset's old eval rows")
    args = ap.parse_args()

    w = args.work_root
    out_dir = args.out_dir or os.path.join(w, "data/meta/precompute")
    os.makedirs(out_dir, exist_ok=True)

    eval_path = os.path.join(out_dir, "light_eval.jsonl")
    stats = {}
    src_counter = Counter()

    with open(eval_path, "a" if args.append_eval else "w") as eval_f:
        for name in args.sources:
            builder = SOURCES[name]
            train_path = os.path.join(out_dir, f"{name}.jsonl")
            n_train = n_eval = n_nocap = 0
            with open(train_path, "w") as tf:
                for row in builder(w):
                    if not row["captions"]:
                        n_nocap += 1
                        continue
                    if is_eval(row["image_id"], args.eval_frac):
                        eval_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_eval += 1
                    else:
                        tf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_train += 1
                    src_counter[row["source"]] += 1
            stats[name] = {"train": n_train, "eval": n_eval, "dropped_no_caption": n_nocap}
            print(f"{name}: train={n_train} eval={n_eval} dropped_no_caption={n_nocap}")

    total_train = sum(s["train"] for s in stats.values())
    total_eval = sum(s["eval"] for s in stats.values())
    stats["total_train"] = total_train
    stats["total_eval"] = total_eval
    stats["by_source"] = dict(src_counter)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"TOTAL: train={total_train} eval={total_eval}")
    print(f"Manifests written to {out_dir}")


if __name__ == "__main__":
    main()
