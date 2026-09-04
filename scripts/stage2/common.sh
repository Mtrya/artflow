#!/bin/bash
# Stage-2 shared launcher. Sourced by arm scripts after they set env overrides.
# Protocol values follow notes/stage2_ablations.md §1.
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
# 459M @256p micro-16 peaks at ~45/48GB on a 4090 and OOMed one 2.1 arm mid-run;
# expandable segments + micro 8 keep the same effective batch with headroom.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Optional venv activation (set VENV_PATH to a venv root containing bin/activate)
if [ -n "${VENV_PATH:-}" ]; then source "$VENV_PATH/bin/activate"; fi

# Optional swanlab credentials (set SWANLAB_NETRC to a netrc file to install)
if [ -n "${SWANLAB_NETRC:-}" ]; then
    mkdir -p "$HOME/.swanlab" && cp "$SWANLAB_NETRC" "$HOME/.swanlab/.netrc" && chmod 600 "$HOME/.swanlab/.netrc"
fi

# Resolve repo root from this script's location (scripts/stage2/common.sh)
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"

# --- model shape (2.1-gate neutral default; later arms override via env) ---
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
NUM_HEADS=${NUM_HEADS:-16}
DOUBLE_STREAM_DEPTH=${DOUBLE_STREAM_DEPTH:-0}
SINGLE_STREAM_DEPTH=${SINGLE_STREAM_DEPTH:-24}
DOUBLE_MOD=${DOUBLE_MOD:-none}
SINGLE_MOD=${SINGLE_MOD:-none}
CONDITIONING=${CONDITIONING:-fused}
FFN_TYPE=${FFN_TYPE:-gated}
QKV_BIAS=${QKV_BIAS:-1}              # 1 -> pass --qkv_bias
ROPE_CENTERED=${ROPE_CENTERED:-0}    # 1 -> pass --rope_centered_grid

# --- optimization (protocol) ---
OPTIMIZER=${OPTIMIZER:-adamw}
MUON_LR=${MUON_LR:-0.02}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-0.5e-4}
START_LR=${START_LR:-1e-5}
WARMUP=${WARMUP:-500}
MAX_STEPS=${MAX_STEPS:-6000}
BATCH_SIZE=${BATCH_SIZE:-8}          # per-GPU micro batch
GRAD_ACCUM=${GRAD_ACCUM:-16}         # effective = micro x GPUs x accum
EMA_DECAY=${EMA_DECAY:-0.999}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-8}
NPROC=${NPROC:-1}

# --- telemetry ---
SWANLAB_PROJECT=${SWANLAB_PROJECT:-artflow-stage2}
EVAL_LOSS_INTERVAL=${EVAL_LOSS_INTERVAL:-100}
EVAL_LOSS_SAMPLES=${EVAL_LOSS_SAMPLES:-512}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
EVAL_BS=${EVAL_BS:-8}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1000}
KID_AT_END=${KID_AT_END:-1}
KID_NUM_FAKE=${KID_NUM_FAKE:-2000}
PROMPTS_FILE=${PROMPTS_FILE:-assets/eval/prompts_v1.jsonl}

# --- required paths (arm scripts must set) ---
: "${TEXT_ENCODER_PATH:?set TEXT_ENCODER_PATH}"
: "${VAE_PATH:?set VAE_PATH}"
: "${DATASET_MIX:?set DATASET_MIX}"
: "${EVAL_DATASET_PATH:?set EVAL_DATASET_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${RUN_NAME:?set RUN_NAME}"

RESUME=${RESUME:-None}
EXTRA_ARGS=${EXTRA_ARGS:-}

QKV_FLAG=""
if [ "$QKV_BIAS" = "1" ]; then QKV_FLAG="--qkv_bias"; fi
KID_FLAG=""
if [ "$KID_AT_END" = "1" ]; then KID_FLAG="--kid_eval_at_end"; fi
ROPE_FLAG=""
if [ "$ROPE_CENTERED" = "1" ]; then ROPE_FLAG="--rope_centered_grid"; fi
RESUME_FULL_FLAG=""
if [ "${RESUME_FULL:-0}" = "1" ]; then RESUME_FULL_FLAG="--resume_full"; fi

accelerate launch --num_processes "$NPROC" -m src.train.train \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED \
    --vae_path "$VAE_PATH" \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --eval_dataset_path "$EVAL_DATASET_PATH" \
    --eval_batch_size $EVAL_BS \
    --dataset_mix "$DATASET_MIX" \
    --text_encoder_path "$TEXT_ENCODER_PATH" \
    --learning_rate $LR --start_learning_rate $START_LR --min_learning_rate $MIN_LR \
    --lr_scheduler_type linear_cosine --lr_warmup_steps $WARMUP \
    --max_steps $MAX_STEPS \
    --curriculum_start 0.0 --curriculum_end 1.0 \
    --caption_dropout_prob 0.1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_ema --ema_decay $EMA_DECAY --ema_update_interval 1 \
    --use_logit_normal_sampling --logit_normal_mu 0.0 --logit_normal_sigma 1.0 \
    --hidden_size $HIDDEN_SIZE --num_heads $NUM_HEADS \
    --double_stream_depth $DOUBLE_STREAM_DEPTH \
    --single_stream_depth $SINGLE_STREAM_DEPTH \
    --mlp_ratio 2.67 \
    --conditioning_scheme "$CONDITIONING" \
    $QKV_FLAG \
    --double_stream_modulation "$DOUBLE_MOD" \
    --single_stream_modulation "$SINGLE_MOD" \
    --ffn_type "$FFN_TYPE" \
    --optimizer "$OPTIMIZER" --muon_lr "$MUON_LR" \
    --swanlab_project "$SWANLAB_PROJECT" \
    --eval_loss_interval $EVAL_LOSS_INTERVAL \
    --eval_loss_samples $EVAL_LOSS_SAMPLES \
    --prompts_file "$PROMPTS_FILE" \
    $KID_FLAG --kid_num_fake $KID_NUM_FAKE \
    $ROPE_FLAG \
    --resume "$RESUME" $RESUME_FULL_FLAG \
    $EXTRA_ARGS
