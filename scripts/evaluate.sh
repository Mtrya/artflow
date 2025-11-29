#!/bin/bash

# Evaluation script for baseline checkpoints

export PYTHONPATH=.venv/bin/python

# Model configuration for stage1 baseline (fused conditioning)
MODEL_CONFIG='{
  "hidden_size": 512,
  "num_heads": 8,
  "double_stream_depth": 8,
  "single_stream_depth": 0,
  "mlp_ratio": 2.67,
  "qkv_bias": true,
  "conditioning_scheme": "pure",
  "double_stream_modulation": "none",
  "single_stream_modulation": "none",
  "ffn_type": "gated",
  "in_channels": 16,
  "patch_size": 2,
  "txt_in_features": 2048
}'

# Evaluation configuration
DATASET_PATH="./precomputed_dataset/heavy-eval@256p"
VAE_PATH="REPA-E/e2e-qwenimage-vae"
TEXT_ENCODER="Qwen/Qwen3-VL-2B-Instruct"
NUM_FID_SAMPLES=2048
NUM_CLIP_SAMPLES=2048
BATCH_SIZE=20
DEVICE="cuda:0"

# Evaluate baseline checkpoints
.venv/bin/python scripts/evaluate_checkpoints.py \
  --checkpoint_pattern "output/baseline/checkpoint_step_018000/ema_weights.pt" \
  --model_config "$MODEL_CONFIG" \
  --vae_path "$VAE_PATH" \
  --text_encoder_path "$TEXT_ENCODER" \
  --dataset_path "$DATASET_PATH" \
  --num_fid_samples $NUM_FID_SAMPLES \
  --num_clip_samples $NUM_CLIP_SAMPLES \
  --batch_size $BATCH_SIZE \
  --device $DEVICE
