#!/bin/bash
# Ascend-side download of precomputed data from private HF repo onto sj-ssd3.
# Submit as a job (fallback for the 2-card notebook whose JupyterTerminal is
# unreliable). __HF_TOKEN__ and __RES_LIST__ are substituted at submit time.
# Token is exported before `set -x` so it never lands in job logs.
export HF_TOKEN=__HF_TOKEN__
set -x
pip3 config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple/
pip3 config set global.trusted-host nexus.sii.shaipower.online
pip3 install -q --progress-bar off "huggingface_hub==1.32.0" 2>&1 | tail -1
export HF_ENDPOINT=https://hf-mirror.com
# Files are xet-stored; classic HTTP falls back to the xet-bridge CDN either
# way (reachable from sj but throttled), so skip the hf_xet codepath.
export HF_HUB_DISABLE_XET=1
DEST=/inspire/sj-ssd3/project/cq-scientific-cooperation-zone/ky26021/artflow/precomputed_dataset
mkdir -p $DEST
python3 - "$DEST" <<'EOF'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
for res in __RES_LIST__:
    snapshot_download(f"kaupane/artflow-precomputed-{res}", repo_type="dataset",
                      local_dir=dest, max_workers=8)
    print("SNAPSHOT_OK", res, flush=True)
EOF
du -sh $DEST
echo DOWNLOAD_ALL_DONE
