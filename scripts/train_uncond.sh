#!/bin/bash

# Stage 0: Unconditional Generation (Algorithm Ablation)

export PYTHONPATH=.venv/bin/python

# Experiment configuration
OUTPUT_DIR="output"

# Dataset configuration
PRECOMPUTED_PATH="./precomputed_dataset/wikiart-captions-monet@256p"

export TOKENIZERS_PARALLELISM=false
export SWANLAB_LOG_DIR="./output/swanlog"

# Training hyperparameters
BATCH_SIZE=32
EPOCHS=200
LR=1e-4
PRECISION="bf16"
MAX_GRAD_NORM=1.0

# Logging
CHECKPOINT_INTERVAL=100
EVAL_INTERVAL=10
NUM_EVAL_SAMPLES=16
EVAL_RESOLUTION=256

# Model configuration
VAE_PATH="REPA-E/e2e-qwenimage-vae"
HIDDEN_SIZE=512
DEPTH=8
NUM_HEADS=8

# Launch training
python -m src.train.train_uncond \
    --run_name "artflow-stage0-fm-ot" \
    --output_dir $OUTPUT_DIR \
    --precomputed_dataset_path $PRECOMPUTED_PATH \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --learning_rate $LR \
    --mixed_precision $PRECISION \
    --max_grad_norm $MAX_GRAD_NORM \
    --algorithm "fm-ot" \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --num_eval_samples $NUM_EVAL_SAMPLES \
    --vae_path $VAE_PATH \
    --model_hidden_size $HIDDEN_SIZE \
    --model_depth $DEPTH \
    --model_num_heads $NUM_HEADS
    
