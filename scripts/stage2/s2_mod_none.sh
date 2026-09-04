#!/bin/bash
# 2.2a gate arm A: single-stream modulation = none (per-layer AdaLN),
# h1024 d24 all-single, 8K steps @256p, 1x4090 Inspire GPU job.
W=${W:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
export HF_HOME=$W/cache/hf
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$W/runs/swanlog}
export TORCH_HOME=${TORCH_HOME:-$W/models/torch_home}
export SWANLAB_NETRC=${SWANLAB_NETRC:-$W/cache/swanlab.netrc}

RUN_NAME=${RUN_NAME:-s2-mod-none}
TEXT_ENCODER_PATH=$W/models/Qwen3-0.6B
VAE_PATH=$W/models/e2e-qwenimage-vae
EVAL_DATASET_PATH=$W/precomputed_dataset/light-eval@256p
OUTPUT_DIR=$W/runs/stage2
MAX_STEPS=8000
SINGLE_MOD=none
ROPE_CENTERED=1

MIX="$W/precomputed_dataset/d1@256p:0.150 $W/precomputed_dataset/d2-wikiart@256p:0.191 $W/precomputed_dataset/d2-museum@256p:0.009 $W/precomputed_dataset/d3-human@256p:0.076 $W/precomputed_dataset/d3-people@256p:0.074 $W/precomputed_dataset/d4-vintage@256p:0.094 $W/precomputed_dataset/d4-zimage@256p:0.022 $W/precomputed_dataset/d4-megalith@256p:0.003 $W/precomputed_dataset/d4-inat@256p:0.001 $W/precomputed_dataset/d4-pd12m@256p:0.079 $W/precomputed_dataset/d4-relaion@256p:0.301"
DATASET_MIX=${DATASET_MIX:-$MIX}

source "$(dirname "$0")/common.sh"
