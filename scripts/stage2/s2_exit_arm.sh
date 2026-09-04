#!/bin/bash
# 2.3a text-encoder exit-layer screen arm: k = EXIT, h1024 d24 mod-layer (the
# pre-registered shape — it was designed to run parallel to 2.2a, so modulation
# stays at the literature default regardless of the 2.2a outcome), 4K steps.
# Usage: EXIT=8 bash scripts/stage2/s2_exit_arm.sh  (RUN_NAME defaults to
# s2-exit-k<EXIT>). RELOCATED from Andromeda to Inspire on 2026-09-05: Andromeda
# holds only the 459-sample Monet mini set; pulling a stage-2 mix slice down and
# farming 3 arms out over daytime windows would take ~2 weeks for a screen that
# §3 allows on Inspire at night. Internal fairness holds (all three k arms share
# platform/mix/protocol); ~3 GPU-h per arm is negligible in the night trough.
W=${W:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
export HF_HOME=$W/cache/hf
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$W/runs/swanlog}
export TORCH_HOME=${TORCH_HOME:-$W/models/torch_home}
export SWANLAB_NETRC=${SWANLAB_NETRC:-$W/cache/swanlab.netrc}

: "${EXIT:?set EXIT (=k) to run this arm}"
RUN_NAME=${RUN_NAME:-s2-exit-k$EXIT}
TEXT_ENCODER_PATH=$W/models/Qwen3-0.6B
VAE_PATH=$W/models/e2e-qwenimage-vae
EVAL_DATASET_PATH=$W/precomputed_dataset/light-eval@256p
OUTPUT_DIR=$W/runs/stage2
MAX_STEPS=${MAX_STEPS:-4000}
SINGLE_MOD=layer
ROPE_CENTERED=1
EXTRA_ARGS="--text_encoder_exit_layer $EXIT"

MIX="$W/precomputed_dataset/d1@256p:0.150 $W/precomputed_dataset/d2-wikiart@256p:0.191 $W/precomputed_dataset/d2-museum@256p:0.009 $W/precomputed_dataset/d3-human@256p:0.076 $W/precomputed_dataset/d3-people@256p:0.074 $W/precomputed_dataset/d4-vintage@256p:0.094 $W/precomputed_dataset/d4-zimage@256p:0.022 $W/precomputed_dataset/d4-megalith@256p:0.003 $W/precomputed_dataset/d4-inat@256p:0.001 $W/precomputed_dataset/d4-pd12m@256p:0.079 $W/precomputed_dataset/d4-relaion@256p:0.301"
DATASET_MIX=${DATASET_MIX:-$MIX}

source "$(dirname "$0")/common.sh"
