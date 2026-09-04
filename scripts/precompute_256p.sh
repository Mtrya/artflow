#!/bin/bash
# Run 256p precompute over all manifests on the GPU notebook.
# Idempotent per-source: skip if output dir already has dataset_info.json.
# Usage: bash scripts/precompute_256p.sh [source ...]   (default: all + light_eval)
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow
REPO=$W/repo
MANI=$W/data/meta/precompute
OUT=$W/precomputed_dataset
if [ -n "${VAE_DIR:-}" ]; then
  VAE=$VAE_DIR
else
  VAE=$(ls -d $W/models/e2e-qwenimage-vae/snapshots/*/ 2>/dev/null | head -1)
fi
BUCKETS="[(256,256),(336,192),(192,336),(288,224),(224,288)]"

if [ -z "$VAE" ]; then echo "VAE not found under $W/models/e2e-qwenimage-vae"; exit 1; fi
VENV=${VENV_DIR:-$W/venv-precompute}
source $VENV/bin/activate
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# keep datasets map cache on GPFS so a notebook restart resumes mid-source
export HF_DATASETS_CACHE=$W/hf_cache/datasets
export HF_HUB_CACHE=$W/hf_cache/hub
cd $REPO

SOURCES="$@"
if [ -z "$SOURCES" ]; then
  SOURCES="light_eval d1 d2_wikiart d2_museum d3_human d3_people d4_vintage d4_zimage d4_megalith d4_inat d4_pd12m d4_relaion"
fi

for S in $SOURCES; do
  M=$MANI/$S.jsonl
  O=$OUT/${S//_/-}@256p
  if [ ! -f "$M" ]; then echo "== $S: manifest missing, skip"; continue; fi
  if [ -f "$O/dataset_info.json" ]; then echo "== $S: already done, skip"; continue; fi
  echo "== $S: start $(date -Is)"
  python -m src.train.precompute \
    --dataset_name "$M" \
    --image_field local_path \
    --caption_fields captions \
    --bbox_field bbox \
    --vae_path "$VAE" \
    --resolution_buckets "$BUCKETS" \
    --output_dir "$O" \
    --batch_size 200 \
    --min_caption_tokens 1 \
    --max_caption_tokens 512 \
    --min_watermark_prob 1.0
  echo "== $S: rc=$? $(date -Is)"
done
echo PRECOMPUTE_256P_ALL_DONE
