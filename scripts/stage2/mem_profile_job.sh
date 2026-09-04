#!/bin/bash
# One-shot memory-profiling job: reproduce the 2.1 training step on the real mix
# and record per-step CUDA memory vs batch composition. ~20 min on 1x4090.
W=${W:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
export HF_HOME=$W/cache/hf
export TOKENIZERS_PARALLELISM=false

MIX="$W/precomputed_dataset/d1@256p:0.150 $W/precomputed_dataset/d2-wikiart@256p:0.191 $W/precomputed_dataset/d2-museum@256p:0.009 $W/precomputed_dataset/d3-human@256p:0.076 $W/precomputed_dataset/d3-people@256p:0.074 $W/precomputed_dataset/d4-vintage@256p:0.094 $W/precomputed_dataset/d4-zimage@256p:0.022 $W/precomputed_dataset/d4-megalith@256p:0.003 $W/precomputed_dataset/d4-inat@256p:0.001 $W/precomputed_dataset/d4-pd12m@256p:0.079 $W/precomputed_dataset/d4-relaion@256p:0.301"

cd "$W/repo"
python scripts/stage2/mem_profile.py \
    --dataset_mix "$MIX" \
    --text_encoder_path "$W/models/Qwen3-0.6B" \
    --vae_path "$W/models/e2e-qwenimage-vae" \
    --output_dir "$W/runs/stage2/mem_profile" \
    --steps "${STEPS:-150}" \
    --batch_size "${BATCH_SIZE:-16}"
