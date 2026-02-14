#!/bin/bash

# Evaluation script for baseline checkpoints

export PYTHONPATH=.venv/bin/python

export TOKENIZERS_PARALLELISM=false

# Model configuration for stage1 baseline
MODEL_CONFIG='{
  "hidden_size": 1152,
  "num_heads": 16,
  "double_stream_depth": 0,
  "single_stream_depth": 28,
  "mlp_ratio": 2.67,
  "qkv_bias": true,
  "conditioning_scheme": "pure",
  "double_stream_modulation": "none",
  "single_stream_modulation": "layer",
  "ffn_type": "gated",
  "in_channels": 16,
  "patch_size": 2,
  "txt_in_features": 1024
}'

# Evaluation configuration
DATASET_PATH="./precomputed_dataset/light-eval@640p"
VAE_PATH="REPA-E/e2e-qwenimage-vae"
TEXT_ENCODER="Qwen/Qwen3-0.6B"
NUM_SAMPLES=2048
BATCH_SIZE=20
DEVICE="cuda:0"

# Evaluate baseline checkpoints
python -m src.evaluation.evaluate_checkpoints \
  --checkpoint_pattern "output/ema_weights_2500000.pt" \
  --model_config "$MODEL_CONFIG" \
  --vae_path "$VAE_PATH" \
  --text_encoder_path "$TEXT_ENCODER" \
  --dataset_path "$DATASET_PATH" \
  --num_samples $NUM_SAMPLES \
  --batch_size $BATCH_SIZE \
  --device $DEVICE \
  --pooling
