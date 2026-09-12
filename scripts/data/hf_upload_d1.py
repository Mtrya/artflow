#!/usr/bin/env python3
"""Upload D1 (Chinese painting) to HuggingFace as webdataset tar shards.

Run on the machine that holds the shared workspace (the images live there and
it reaches HuggingFace directly):

    HF_TOKEN=... python $ARTFLOW_ROOT/repo/scripts/data/hf_upload_d1.py \
        [--components npm_tw_c1,npm_tw_c2,...]

Layout in repo:  data/<component>/shard-00000.tar  (each entry: <image_id>.jpg
+ <image_id>.json sidecar), metadata.parquet and README.md at root.

Idempotent: shard names already in the repo are skipped; staging files are
deleted after each successful upload. Components whose images are not yet on
disk (no clean/<c>/images dir) are skipped with a note.
"""
import argparse
import io
import json
import os
import tarfile
import time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from scripts.data.common import workspace_root

META_NAME = "data/meta/d1/d1_metadata.jsonl"
STAGING_NAME = "hf_d1_staging"
REPO_ID = "kaupane/chinese-painting-collection"
SHARD_BYTES = 4 * 1024**3  # ~4GB per shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", default=None,
                    help="comma list; default: all shards in metadata")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    args = ap.parse_args()

    root = workspace_root()
    META = os.path.join(root, META_NAME)
    STAGING = os.path.join(root, STAGING_NAME)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    existing = {f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
                if f.endswith(".tar")}
    print("existing shards:", len(existing), flush=True)

    rows = []
    with open(META, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    by_comp = {}
    for r in rows:
        by_comp.setdefault(r["shard"], []).append(r)
    comps = args.components.split(",") if args.components else sorted(by_comp)

    os.makedirs(STAGING, exist_ok=True)
    for comp in comps:
        comp_rows = by_comp.get(comp, [])
        img_root = os.path.join(root, "data/clean", comp, "images")
        if not os.path.isdir(img_root):
            print(f"[{comp}] images not present under the workspace root yet, skip", flush=True)
            continue
        shard_idx, tar, tar_path, cur = 0, None, None, 0
        n_files = 0

        def shard_name(i):
            return f"data/{comp}/shard-{i:05d}.tar"

        def open_shard(i):
            p = os.path.join(STAGING, f"{comp}-{i:05d}.tar")
            return p, tarfile.open(p, "w")

        def close_and_upload(i, p, t):
            t.close()
            repo_name = shard_name(i)
            if repo_name in existing:
                print(f"[{comp}] {repo_name} already in repo, skip upload", flush=True)
                os.remove(p)
                return
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=p, path_in_repo=repo_name,
                                    repo_id=REPO_ID, repo_type="dataset")
                    break
                except Exception as exc:
                    print(f"[{comp}] upload {repo_name} attempt {attempt} failed: {exc}",
                          flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"upload failed permanently: {repo_name}")
            os.remove(p)
            print(f"[{comp}] uploaded {repo_name}", flush=True)

        t0 = time.time()
        for r in comp_rows:
            img = r["local_path"]
            if not os.path.exists(img):
                continue
            sz = os.path.getsize(img)
            if tar is None:
                tar_path, tar = open_shard(shard_idx)
            if cur + sz > args.shard_bytes and cur > 0:
                close_and_upload(shard_idx, tar_path, tar)
                shard_idx += 1
                tar_path, tar = open_shard(shard_idx)
                cur = 0
            ext = os.path.splitext(img)[1].lower().lstrip(".") or "jpg"
            tar.add(img, arcname=f"{r['image_id']}.{ext}")
            meta = {k: v for k, v in r.items() if k != "local_path"}
            buf = json.dumps(meta, ensure_ascii=False).encode()
            ti = tarfile.TarInfo(f"{r['image_id']}.json")
            ti.size = len(buf)
            tar.addfile(ti, io.BytesIO(buf))
            cur += sz
            n_files += 1
        if tar is not None:
            close_and_upload(shard_idx, tar_path, tar)
        print(f"[{comp}] done: {n_files} images, {shard_idx + 1} shards, "
              f"{time.time() - t0:.0f}s", flush=True)

    print("UPLOAD_SHARDS_DONE", flush=True)


if __name__ == "__main__":
    main()
