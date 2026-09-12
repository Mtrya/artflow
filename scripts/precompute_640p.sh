#!/bin/bash
# 640p precompute over every source manifest, one worker per GPU.
#
# scripts/precompute_256p.sh is the template.  Three differences: the 640p
# resolution buckets, a caption ceiling high enough that no row is dropped for
# a long caption, and the sources spread over eight GPUs, because a 640p VAE
# forward is about six times the 256p one.
#
# The largest source (d4_relaion, 622k rows) is split over five workers and
# written to five directories.  As a single unit it would set the whole job's
# critical path at roughly three times the average worker; split five ways it
# is close to the average.  Training reads each directory as its own mixture
# entry, so the split needs no merge pass afterwards.
#
# Idempotent per source: a source whose output directory already holds
# dataset_info.json is skipped, so re-running continues where the last attempt
# stopped.
#
# Usage: bash scripts/precompute_640p.sh <worker_index> <worker_count>
set -u
W=${ARTFLOW_ROOT:?Set ARTFLOW_ROOT to the shared-workspace root used on the cluster}
REPO=$W/repo
MANI=$W/data/meta/precompute
OUT=$W/precomputed_dataset
VAE=$W/models/e2e-qwenimage-vae
TOK=$W/models/Qwen3-0.6B
BUCKETS="[(640,640),(848,480),(480,848),(736,560),(560,736)]"
MAX_CAPTION_TOKENS=${MAX_CAPTION_TOKENS:-8192}
BATCH=${BATCH:-50}

WORKER=${1:?worker index required}
WORKERS=${2:?worker count required}

# Sources per worker, balanced by manifest row count.  Sizes in thousands:
# wikiart 214, vintage 195, pd12m 162, relaion 622 in five parts of ~124,
# human 120, people 114, d1 91, zimage 45, pexels 37, synth 20, museum 10,
# megalith 5.5, inat 2.8.
case $WORKER in
  0) SOURCES="d2_wikiart" ;;
  1) SOURCES="d4_vintage d4_inat" ;;
  2) SOURCES="d4_pd12m d4_zimage" ;;
  3) SOURCES="d4_relaion_p0 d3_people" ;;
  4) SOURCES="d4_relaion_p1 d3_human" ;;
  5) SOURCES="d4_relaion_p2 d1" ;;
  6) SOURCES="d4_relaion_p3 d3_pexels d3_synth d2_museum d4_megalith" ;;
  7) SOURCES="d4_relaion_p4" ;;
  *) echo "worker $WORKER is outside 0..7"; exit 1 ;;
esac

if [ -n "${VAE_DIR:-}" ]; then
  VAE=$VAE_DIR
elif [ -f $W/models/e2e-qwenimage-vae/diffusion_pytorch_model.safetensors ]; then
  # Flattened copy on the shared disk: from_pretrained cannot read the
  # blobs/refs/snapshots layout of a HuggingFace cache directory.
  VAE=$W/models/e2e-qwenimage-vae
fi
if [ -z "$VAE" ] || [ ! -e "$VAE" ]; then echo "VAE not found under $W/models/e2e-qwenimage-vae"; exit 1; fi

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=$W/hf_cache/datasets
export HF_HUB_CACHE=$W/hf_cache/hub
cd $REPO

echo "worker $WORKER/$WORKERS starting on $(date -Is)"
echo "sources: $SOURCES"

for S in $SOURCES; do
  M=$MANI/$S.jsonl
  [ -f "$M" ] || M=$MANI/relaion_parts/$S.jsonl
  O=$OUT/${S//_/-}@640p
  if [ ! -f "$M" ]; then echo "== $S: manifest missing at $M, skip"; continue; fi
  if [ -f "$O/dataset_info.json" ]; then echo "== $S: already done, skip"; continue; fi
  echo "== $S: start $(date -Is)"
  CUDA_VISIBLE_DEVICES=$WORKER python -m src.train.precompute \
    --dataset_name "$M" \
    --image_field local_path \
    --caption_fields captions \
    --bbox_field bbox \
    --vae_path "$VAE" \
    --resolution_buckets "$BUCKETS" \
    --output_dir "$O" \
    --batch_size "$BATCH" \
    --min_caption_tokens 1 \
    --max_caption_tokens "$MAX_CAPTION_TOKENS" \
    --tokenizer "$TOK" \
    --min_watermark_prob 1.0
  echo "== $S: rc=$? $(date -Is)"
done
echo "worker $WORKER done $(date -Is)"
