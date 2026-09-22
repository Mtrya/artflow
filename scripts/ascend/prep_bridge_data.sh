#!/bin/bash
# Build mini precomputed dataset + stage VAE/bucket plan to the HF bridge repo.
set -ex
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow
pip3 config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple/ || true
pip3 config set global.trusted-host nexus.sii.shaipower.online || true
pip3 install -q --break-system-packages --progress-bar off datasets huggingface_hub pyarrow torch 2>&1 | tail -1

WORK=/tmp/ascend-stage
mkdir -p $WORK && cd $WORK

python3 - <<'EOF'
from datasets import load_from_disk
ds = load_from_disk("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow/precomputed_dataset/d1@256p")
print("full rows:", len(ds), "cols:", ds.column_names)
mini = ds.select(range(2048))
mini.save_to_disk("/tmp/ascend-stage/mini-d1-256p")
print("mini saved:", len(mini))
EOF

tar czf mini-d1-256p.tar.gz -C /tmp/ascend-stage mini-d1-256p
tar czf e2e-qwenimage-vae.tar.gz -C $W/models e2e-qwenimage-vae
cp $W/bucket_plans/hero/batch-targets-0914/hero-256p-k20.json .
ls -la

python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for f in ["mini-d1-256p.tar.gz", "e2e-qwenimage-vae.tar.gz", "hero-256p-k20.json"]:
    api.upload_file(path_or_fileobj=f"/tmp/ascend-stage/{f}", path_in_repo=f,
                    repo_id="kaupane/artflow-code-bridge", repo_type="dataset")
    print("uploaded", f)
EOF
echo PREP_DONE
