#!/bin/bash

# Stage 2.3: Reward Model Training for Flow-GRPO
# Edit the configuration below, then run: ./scripts/train_rm.sh

export PYTHONPATH=.venv/bin/python

# Dataset configuration
DATASET_PATH="./scored_images/scored_dataset"

# Output
OUTPUT_DIR="./output/reward_model"

# Model configuration
ENCODER_NAME="OFA-Sys/chinese-clip-vit-large-patch14-336px"
FEATURE_LAYERS="12 18 23"  # Early, middle, late layers for style perception
HIDDEN_DIM=512
DROPOUT=0.1

# Training hyperparameters
BATCH_SIZE=32
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-5
VAL_SPLIT=0.1

# System
NUM_WORKERS=4
SAVE_EVERY=10
DEVICE="cuda"

# Launch training
python -m src.train.train_rm \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --encoder_name "$ENCODER_NAME" \
    --feature_layers $FEATURE_LAYERS \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --val_split $VAL_SPLIT \
    --num_workers $NUM_WORKERS \
    --save_every $SAVE_EVERY \
    --device $DEVICE
