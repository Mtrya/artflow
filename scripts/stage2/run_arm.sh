#!/bin/bash
# Generic Stage-2 arm launcher — single entry point for every arm after 2.2a.
# Shapes/protocol come from the job environment (inspire job create --env K=V),
# so launching a new arm is one job-create call, no per-arm file needed.
# Required env: RUN_NAME MAX_STEPS HIDDEN_SIZE SINGLE_STREAM_DEPTH SINGLE_MOD
#               DOUBLE_STREAM_DEPTH DOUBLE_MOD
# Optional env: MIX=old|new (default new: stage-2 mix per §4)
#               EXIT=<k> (adds --text_encoder_exit_layer k)
#               EVAL_INTERVAL / CHECKPOINT_INTERVAL / EVAL_LOSS_INTERVAL / ...
#               RESUME=<ckpt_dir> [RESUME_FULL=1]
# Protocol defaults (batch 8 x accum 16 = 128, warmup 500, EMA 0.999, seed 42,
# LR 3e-4 cosine min 0.5e-4, centered-grid RoPE) come from common.sh.
W=${W:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
export HF_HOME=$W/cache/hf
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$W/runs/swanlog}
export TORCH_HOME=${TORCH_HOME:-$W/models/torch_home}
export SWANLAB_NETRC=${SWANLAB_NETRC:-$W/cache/swanlab.netrc}

: "${RUN_NAME:?set RUN_NAME}"
: "${MAX_STEPS:?set MAX_STEPS}"
: "${HIDDEN_SIZE:?set HIDDEN_SIZE}"
: "${SINGLE_STREAM_DEPTH:?set SINGLE_STREAM_DEPTH}"
: "${SINGLE_MOD:?set SINGLE_MOD}"
: "${DOUBLE_STREAM_DEPTH:?set DOUBLE_STREAM_DEPTH}"
: "${DOUBLE_MOD:?set DOUBLE_MOD}"

TEXT_ENCODER_PATH=$W/models/Qwen3-0.6B
VAE_PATH=$W/models/e2e-qwenimage-vae
EVAL_DATASET_PATH=$W/precomputed_dataset/light-eval@256p
OUTPUT_DIR=$W/runs/stage2
ROPE_CENTERED=1

if [ "${MIX:-new}" = "old" ]; then
    # 2.4 old-recipe mix (world .8 / art .1 / portrait .1, within-group ∝ size)
    MIX_STR="$W/precomputed_dataset/d4-vintage@256p:0.151 $W/precomputed_dataset/d4-zimage@256p:0.035 $W/precomputed_dataset/d4-megalith@256p:0.004 $W/precomputed_dataset/d4-inat@256p:0.002 $W/precomputed_dataset/d4-pd12m@256p:0.126 $W/precomputed_dataset/d4-relaion@256p:0.482 $W/precomputed_dataset/d1@256p:0.029 $W/precomputed_dataset/d2-wikiart@256p:0.068 $W/precomputed_dataset/d2-museum@256p:0.003 $W/precomputed_dataset/d3-human@256p:0.051 $W/precomputed_dataset/d3-people@256p:0.049"
else
    MIX_STR="$W/precomputed_dataset/d1@256p:0.150 $W/precomputed_dataset/d2-wikiart@256p:0.191 $W/precomputed_dataset/d2-museum@256p:0.009 $W/precomputed_dataset/d3-human@256p:0.076 $W/precomputed_dataset/d3-people@256p:0.074 $W/precomputed_dataset/d4-vintage@256p:0.094 $W/precomputed_dataset/d4-zimage@256p:0.022 $W/precomputed_dataset/d4-megalith@256p:0.003 $W/precomputed_dataset/d4-inat@256p:0.001 $W/precomputed_dataset/d4-pd12m@256p:0.079 $W/precomputed_dataset/d4-relaion@256p:0.301"
fi
DATASET_MIX=${DATASET_MIX:-$MIX_STR}

if [ -n "${EXIT:-}" ]; then
    EXTRA_ARGS="--text_encoder_exit_layer $EXIT"
fi

source "$(dirname "$0")/common.sh"
