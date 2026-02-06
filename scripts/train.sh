#!/bin/bash

# Stage 1: Conditional Generation (Architecture Ablation)

export PYTHONPATH=.venv/bin/python

# Experiment configuration
OUTPUT_DIR="output"

export TOKENIZERS_PARALLELISM=false
export SWANLAB_LOG_DIR="./output/swanlog"

# Evaluation configuration
VAE_PATH="REPA-E/e2e-qwenimage-vae"
EVAL_DATASET_PATH="./precomputed_dataset/light-eval@640p"
CHECKPOINT_INTERVAL=1000
EVAL_INTERVAL=400
EVAL_SAMPLES=80
EVAL_BS=10
EVAL_BUCKET_RESOLUTIONS="640x640,840x480,480x840,720x560,560x720"

# Training configuration
DATASET_MIX="./precomputed_dataset/world@640p:0.3 ./precomputed_dataset/distills@640p:0.3 ./precomputed_dataset/art@640p:0.2 ./precomputed_dataset/portrait@640p:0.2"
TEXT_ENCODER_PATH="Qwen/Qwen3-VL-2B-Instruct"
START_LR=0.4e-4
LR=0.4e-4
MIN_LR=0.1e-4
LR_SCHEDULER="linear_cosine"
LR_WARMUP_STEPS=0000
MAX_STEPS=10_000
GRAD_ACCUM_STEPS=7
MAX_GRAD_NORM=1.0
EMA_DECAY=0.9999
EMA_INTERVAL=1
BATCH_SIZE=6
NUM_WORKERS=16
CURRICULUM_START=1.0
CURRICULUM_END=1.0
SEED=42
USE_LOGIT_NORMAL=true
LOGIT_NORMAL_MU=0.0
LOGIT_NORMAL_SIGMA=1.0
VP_SHIFT=1.8
TELEMETRY_LOG_INTERVAL=500
CACHE_CLEAR_INTERVAL=200

RUN_NAME="stage2_5"
RESUME_PATH="output/stage2_4/checkpoint_step_025000"

accelerate launch -m src.train.train \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --vae_path $VAE_PATH \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --eval_dataset_path $EVAL_DATASET_PATH \
    --num_eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BS \
    --bucket_resolutions "$EVAL_BUCKET_RESOLUTIONS" \
    --dataset_mix "$DATASET_MIX" \
    --text_encoder_path $TEXT_ENCODER_PATH \
    --learning_rate $LR \
    --start_learning_rate $START_LR \
    --lr_scheduler_type $LR_SCHEDULER \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --min_learning_rate $MIN_LR \
    --max_steps $MAX_STEPS \
    --curriculum_start $CURRICULUM_START \
    --curriculum_end $CURRICULUM_END \
    --caption_dropout_prob 0.1 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_INTERVAL \
    --use_logit_normal \
    --logit_normal_mu $LOGIT_NORMAL_MU \
    --logit_normal_sigma $LOGIT_NORMAL_SIGMA \
    --vp_shift $VP_SHIFT \
    --telemetry_log_interval $TELEMETRY_LOG_INTERVAL \
    --cache_clear_interval $CACHE_CLEAR_INTERVAL \
    --hidden_size 1152 \
    --num_heads 16 \
    --double_stream_depth 0 \
    --single_stream_depth 28 \
    --mlp_ratio 2.67 \
    --conditioning_scheme "pure" \
    --qkv_bias \
    --double_stream_modulation "none" \
    --single_stream_modulation "layer" \
    --ffn_type "gated" \
    --resume $RESUME_PATH
