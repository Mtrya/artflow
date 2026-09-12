#!/usr/bin/env python3
"""Build consolidated D1 (Chinese painting) metadata.

Merges clean metadata + VLM labels (+ bbox for npm_tw, + artifacts pass for
museum sets) into one JSONL with absolute image paths under the shared
workspace root.

Filters: npm_tw keeps view_type full/mounted/detail; museum sets keep
full/mounted AND culture==chinese. rolled/junk/error rows are dropped.

Run on the machine that holds the workspace:
    python $ARTFLOW_ROOT/repo/scripts/data/build_d1_metadata.py
Output: $ARTFLOW_ROOT/data/meta/d1/d1_metadata.jsonl
"""
import glob
import json
import os

from scripts.data.common import workspace_root

W = workspace_root()
OUT = os.path.join(W, "data/meta/d1/d1_metadata.jsonl")

NPM_SHARDS = ["npm_tw_c0", "npm_tw_c1", "npm_tw_c2", "npm_tw_c3"]
MUSEUM_SETS = ["aic_china", "met_china", "fsg", "princeton"]
NPM_VIEWS = {"full", "mounted", "detail"}
MUSEUM_VIEWS = {"full", "mounted"}


def load_jsonl_map(pattern, key="image_id"):
    out = {}
    for f in glob.glob(pattern):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # crash-truncated lines
                out[r[key]] = r
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for s in NPM_SHARDS + MUSEUM_SETS:
        labels = load_jsonl_map(f"{W}/data/labels/{s}/labels_*.jsonl")
        bbox = load_jsonl_map(f"{W}/data/labels/{s}_bbox/bboxes_*.jsonl")
        arts = load_jsonl_map(f"{W}/data/labels/{s}_artifacts/artifacts_*.jsonl")
        meta = os.path.join(W, "data/clean", s, "metadata.jsonl")
        kept = 0
        with open(meta, encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if m.get("rejected"):
                    continue
                iid = m["image_id"]
                lab = (labels.get(iid) or {}).get("label") or {}
                vt = lab.get("view_type")
                if s in NPM_SHARDS:
                    if vt not in NPM_VIEWS:
                        continue
                else:
                    if vt not in MUSEUM_VIEWS or lab.get("culture") != "chinese":
                        continue
                lp = m["local_path"]
                if not os.path.isabs(lp):
                    lp = os.path.join(W, lp)  # local_path is repo-root-relative
                lp = os.path.normpath(lp)  # collapse "../" from museum metadata
                rows.append({
                    "image_id": iid, "source": "npm_tw" if s.startswith("npm") else s,
                    "shard": s, "local_path": lp,
                    "title": m.get("title"), "artist": m.get("artist"),
                    "category": m.get("category"), "object_no": m.get("object_no"),
                    "view_type": vt, "culture": lab.get("culture"),
                    "caption_zh": lab.get("caption_zh"), "caption_en": lab.get("caption_en"),
                    "ocr_text": lab.get("ocr_text") or "",
                    "artifacts": lab.get("artifacts") or (arts.get(iid) or {}).get("artifacts") or [],
                    "bbox": (bbox.get(iid) or {}).get("bbox"),
                })
                kept += 1
        print(s, "kept", kept, flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("TOTAL", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
