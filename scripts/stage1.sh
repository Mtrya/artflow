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
EVAL_INTERVAL=10
EVAL_SAMPLES=32
EVAL_BS=8

# Training configuration
PRECOMPUTED_PATH="./precomputed_dataset/wikiart-captions@256p"
TEXT_ENCODER_PATH="Qwen/Qwen3-VL-2B-Instruct"
LR=5e-4
MIN_LR=1e-4
LR_SCHEDULER="linear_cosine"
LR_WARMUP_STEPS=2000
MAX_STEPS=18000
GRAD_ACCUM_STEPS=1
MIXED_PRECISION="bf16"
MAX_GRAD_NORM=1.0
EMA_DECAY=0.99
EMA_INTERVAL=1
BATCH_SIZE=32
NUM_WORKERS=4

accelerate launch scripts/train_stage1.py \
    --run_name "all-single-depth8" \
    --output_dir $OUTPUT_DIR \
    --vae_path $VAE_PATH \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --num_eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BS \
    --dataset_path $PRECOMPUTED_PATH \
    --text_encoder_path $TEXT_ENCODER_PATH \
    --learning_rate $LR \
    --lr_scheduler_type $LR_SCHEDULER \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --min_learning_rate $MIN_LR \
    --max_steps $MAX_STEPS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --max_grad_norm $MAX_GRAD_NORM \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_INTERVAL \
    --hidden_size 512 \
    --num_heads 8 \
    --double_stream_depth 0 \
    --single_stream_depth 8 \
    --mlp_ratio 2.67 \
    --conditioning_scheme "pure" \
    --double_stream_modulation "none" \
    --single_stream_modulation "none" \
    --ffn_type "gated"

accelerate launch scripts/train_stage1.py \
    --run_name "all-single-depth12" \
    --output_dir $OUTPUT_DIR \
    --vae_path $VAE_PATH \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --num_eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BS \
    --dataset_path $PRECOMPUTED_PATH \
    --text_encoder_path $TEXT_ENCODER_PATH \
    --learning_rate $LR \
    --lr_scheduler_type $LR_SCHEDULER \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --min_learning_rate $MIN_LR \
    --max_steps $MAX_STEPS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --max_grad_norm $MAX_GRAD_NORM \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_INTERVAL \
    --hidden_size 512 \
    --num_heads 8 \
    --double_stream_depth 0 \
    --single_stream_depth 12 \
    --mlp_ratio 2.67 \
    --conditioning_scheme "pure" \
    --double_stream_modulation "none" \
    --single_stream_modulation "none" \
    --ffn_type "gated"

accelerate launch scripts/train_stage1.py \
    --run_name "all-single-depth8-fused" \
    --output_dir $OUTPUT_DIR \
    --vae_path $VAE_PATH \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --num_eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BS \
    --dataset_path $PRECOMPUTED_PATH \
    --text_encoder_path $TEXT_ENCODER_PATH \
    --learning_rate $LR \
    --lr_scheduler_type $LR_SCHEDULER \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --min_learning_rate $MIN_LR \
    --max_steps $MAX_STEPS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --max_grad_norm $MAX_GRAD_NORM \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_INTERVAL \
    --hidden_size 512 \
    --num_heads 8 \
    --double_stream_depth 0 \
    --single_stream_depth 8 \
    --mlp_ratio 2.67 \
    --conditioning_scheme "fused" \
    --double_stream_modulation "none" \
    --single_stream_modulation "none" \
    --ffn_type "gated"