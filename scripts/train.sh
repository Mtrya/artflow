#!/bin/bash

# Stage 1: Conditional Generation (Architecture Ablation)

export PYTHONPATH=.venv/bin/python

# Experiment configuration
OUTPUT_DIR="output"

export TOKENIZERS_PARALLELISM=false
export SWANLAB_LOG_DIR="./output/swanlog"

# Evaluation configuration
VAE_PATH="REPA-E/e2e-qwenimage-vae"
CHECKPOINT_INTERVAL=6000
EVAL_INTERVAL=1000
EVAL_SAMPLES=256
EVAL_BS=16

# Training configuration
# Dataset mix: "path1:weight1 path2:weight2" or single "path"
DATASET_MIX="./precomputed_dataset/mixed-art@256p:0.9 ./precomputed_dataset/face-caption-hq@256p:0.1"
TEXT_ENCODER_PATH="Qwen/Qwen3-VL-2B-Instruct"
LR=3e-4
START_LR=1e-6
MIN_LR=1e-6
LR_SCHEDULER="linear_cosine"
LR_WARMUP_STEPS=2000
MAX_STEPS=24000
GRAD_ACCUM_STEPS=4
MAX_GRAD_NORM=1.0
EMA_DECAY=0.99
EMA_INTERVAL=1
BATCH_SIZE=16
NUM_WORKERS=4
CURRICULUM_START=0.0
CURRICULUM_END=1.0

accelerate launch -m src.train.train \
    --run_name "baseline" \
    --output_dir $OUTPUT_DIR \
    --vae_path $VAE_PATH \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --num_eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BS \
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
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_INTERVAL \
    --hidden_size 640 \
    --num_heads 8 \
    --double_stream_depth 0 \
    --single_stream_depth 12 \
    --mlp_ratio 2.67 \
    --conditioning_scheme "pure" \
    --qkv_bias \
    --double_stream_modulation "none" \
    --single_stream_modulation "none" \
    --ffn_type "gated"
