#!/bin/bash

# Evaluation script for baseline checkpoints

export PYTHONPATH=.venv/bin/python

export TOKENIZERS_PARALLELISM=false

# Model configuration for stage1 baseline
MODEL_CONFIG='{
  "hidden_size": 512,
  "num_heads": 8,
  "double_stream_depth": 8,
  "single_stream_depth": 0,
  "mlp_ratio": 4.0,
  "qkv_bias": true,
  "conditioning_scheme": "fused",
  "double_stream_modulation": "none",
  "single_stream_modulation": "none",
  "ffn_type": "standard",
  "in_channels": 16,
  "patch_size": 2,
  "txt_in_features": 2048
}'

# Evaluation configuration
DATASET_PATH="./precomputed_dataset/wikiart-captions@256p"
VAE_PATH="REPA-E/e2e-qwenimage-vae"
TEXT_ENCODER="Qwen/Qwen3-VL-2B-Instruct"
NUM_SAMPLES=2048
BATCH_SIZE=20
DEVICE="cuda:0"

# Evaluate baseline checkpoints
python -m src.evaluation.evaluate_checkpoints \
  --checkpoint_pattern "output/standard-ffn/checkpoint_step_*/ema_weights.pt" \
  --model_config "$MODEL_CONFIG" \
  --vae_path "$VAE_PATH" \
  --text_encoder_path "$TEXT_ENCODER" \
  --dataset_path "$DATASET_PATH" \
  --num_samples $NUM_SAMPLES \
  --batch_size $BATCH_SIZE \
  --device $DEVICE \
  --pooling
