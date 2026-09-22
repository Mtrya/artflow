#!/bin/bash
# Download precomputed data from the private HF mirror repo onto sj-ssd3.
# Runs on ascend-dl-staging (2-card notebook in 昇腾卡公共空间).
# Usage: HF_TOKEN=<token> bash migrate_download.sh 256p|640p|896p
set -e
RES=${1:?usage: migrate_download.sh 256p|640p|896p}
DEST=/inspire/sj-ssd3/project/cq-scientific-cooperation-zone/ky26021/artflow/precomputed_dataset
mkdir -p "$DEST"
export HF_ENDPOINT=https://hf-mirror.com
: "${HF_TOKEN:?need HF_TOKEN env}"
python3 - "$RES" "$DEST" <<'EOF'
import sys
from huggingface_hub import snapshot_download
res, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(f"kaupane/artflow-precomputed-{res}",
                         repo_type="dataset", local_dir=dest, max_workers=8)
print("SNAPSHOT_OK", path)
EOF
du -sh "$DEST"
echo DOWNLOAD_DONE $RES
