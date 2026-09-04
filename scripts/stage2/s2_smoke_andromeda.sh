#!/bin/bash
# Stage-2 wiring smoke on Andromeda: 300 steps on the 459-sample Monet mini set.
# Exercises: fused conditioning, eval-loss probe, fixed prompt grids, checkpoint +
# full resume, end-of-arm KID. Not an ablation arm — throwaway output.
# Usage: bash scripts/stage2/s2_smoke_andromeda.sh   (from anywhere)
# Resume test: RESUME=<ckpt_dir> RESUME_FULL=1 bash scripts/stage2/s2_smoke_andromeda.sh

RUN_NAME=${RUN_NAME:-s2-smoke}
VENV_PATH=${VENV_PATH:-$HOME/Projects/artflow-reboot/.venv}
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-$HOME/artflow-models/Qwen3-0.6B}
VAE_PATH=${VAE_PATH:-$HOME/artflow-models/e2e-qwenimage-vae}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-$HOME/Projects/artflow-reboot/precomputed_dataset/wikiart-captions-monet@256p}
DATASET_MIX=${DATASET_MIX:-$EVAL_DATASET_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/Projects/artflow/output}

MAX_STEPS=${MAX_STEPS:-300}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}
WARMUP=${WARMUP:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_LOSS_INTERVAL=${EVAL_LOSS_INTERVAL:-50}
EVAL_LOSS_SAMPLES=${EVAL_LOSS_SAMPLES:-64}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
EVAL_BS=${EVAL_BS:-4}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}
KID_AT_END=${KID_AT_END:-1}
KID_NUM_FAKE=${KID_NUM_FAKE:-64}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-artflow-stage2-smoke}
export TORCH_HOME=${TORCH_HOME:-$HOME/artflow-models/torch_home}

source "$(dirname "$0")/common.sh"
