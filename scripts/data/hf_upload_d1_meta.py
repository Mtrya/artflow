#!/usr/bin/env python3
"""Convert d1_metadata.jsonl to parquet and upload metadata + README to the HF repo.

Run on the machine that holds the shared workspace after shard uploads finish:

    HF_TOKEN=... python $ARTFLOW_ROOT/repo/scripts/data/hf_upload_d1_meta.py
"""
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from scripts.data.common import workspace_root

W = workspace_root()
META = os.path.join(W, "data/meta/d1/d1_metadata.jsonl")
STAGING = os.path.join(W, "hf_d1_staging")
REPO_ID = "kaupane/chinese-painting-collection"

rows = []
with open(META, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        r.pop("local_path", None)
        r["artifacts"] = json.dumps(r.get("artifacts") or [], ensure_ascii=False)
        r["bbox"] = json.dumps(r["bbox"]) if r.get("bbox") else None
        rows.append(r)

cols = sorted({k for r in rows for k in r})
table = pa.table({c: [r.get(c) for r in rows] for c in cols})
out = os.path.join(STAGING, "metadata.parquet")
pq.write_table(table, out, compression="zstd")
print("parquet rows:", table.num_rows, "cols:", cols)

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=out, path_in_repo="metadata/metadata.parquet",
                repo_id=REPO_ID, repo_type="dataset")
print("uploaded metadata/metadata.parquet")
api.upload_file(path_or_fileobj=os.path.join(STAGING, "README.md"),
                path_in_repo="README.md", repo_id=REPO_ID, repo_type="dataset")
print("uploaded README.md")
print("META_UPLOAD_DONE")
