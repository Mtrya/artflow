#!/bin/bash

# Stage 0: Unconditional Generation (Algorithm Ablation)
# Runs: 
# 1. Score matching with diffusion path;
# 2. Flow matching with diffusion path;
# 3. Flow matching with optimal transport path 

export PYTHONPATH=.venv/bin/python

# Experiment configuration
OUTPUT_DIR="output"
DATASET="kaupane/wikiart-captions-monet"
export WANDB_DIR="./output"

# Training hyperparameters
RESOLUTION=256
BATCH_SIZE=32
EPOCHS=500  # ~20k steps for 1.3k images
LR=1e-4
PRECISION="bf16"
MAX_GRAD_NORM=1.0

# Model hyperparameters
MODEL_HIDDEN_SIZE=512
MODEL_DEPTH=8
MODEL_NUM_HEADS=8

# Run names
RUN_NAME_SM_DIFFUSION="stage0_sm_diffusion"
RUN_NAME_FM_DIFFUSION="stage0_fm_diffusion"
RUN_NAME_FM_OT="stage0_fm_ot"

# Evaluation parameters
CHECKPOINT_INTERVAL=100
EVAL_INTERVAL=20
NUM_EVAL_SAMPLES=16

python scripts/train_stage0.py \
    --run_name "$RUN_NAME_SM_DIFFUSION" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET" \
    --resolution "$RESOLUTION" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --algorithm "sm-diffusion" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_eval_samples "$NUM_EVAL_SAMPLES" \
    --mixed_precision "$PRECISION" \
    --model_hidden_size "$MODEL_HIDDEN_SIZE" \
    --model_depth "$MODEL_DEPTH" \
    --model_num_heads "$MODEL_NUM_HEADS" \
    --max_grad_norm "$MAX_GRAD_NORM"

python scripts/train_stage0.py \
    --run_name "$RUN_NAME_FM_DIFFUSION" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET" \
    --resolution "$RESOLUTION" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --algorithm "fm-diffusion" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_eval_samples "$NUM_EVAL_SAMPLES" \
    --mixed_precision "$PRECISION" \
    --model_hidden_size "$MODEL_HIDDEN_SIZE" \
    --model_depth "$MODEL_DEPTH" \
    --model_num_heads "$MODEL_NUM_HEADS" \
    --max_grad_norm "$MAX_GRAD_NORM"

python scripts/train_stage0.py \
    --run_name "$RUN_NAME_FM_OT" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET" \
    --resolution "$RESOLUTION" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --algorithm "fm-ot" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_eval_samples "$NUM_EVAL_SAMPLES" \
    --mixed_precision "$PRECISION" \
    --model_hidden_size "$MODEL_HIDDEN_SIZE" \
    --model_depth "$MODEL_DEPTH" \
    --model_num_heads "$MODEL_NUM_HEADS" \
    --max_grad_norm "$MAX_GRAD_NORM"
