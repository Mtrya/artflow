#!/bin/bash

# Manual image scoring with VLM prompt generation
# Edit the configuration below, then run: ./scripts/score_images.sh

export PYTHONPATH=.venv/bin/python

# Dataset configuration
DATASET_NAME="kaupane/wikiart-captions"
IMAGE_FIELD="image"
TARGET_SAMPLES=5000

# Quality filters
MIN_RESOLUTION=640
ASPECT_RATIO_MIN=0.8
ASPECT_RATIO_MAX=1.25

# Output
OUTPUT_DIR="./scored_images"

# Interface
PORT=7860

# Launch scoring interface
python -m src.dataset.score_images score \
    --dataset_name "$DATASET_NAME" \
    --image_field "$IMAGE_FIELD" \
    --target_samples $TARGET_SAMPLES \
    --output_dir "$OUTPUT_DIR" \
    --min_resolution $MIN_RESOLUTION \
    --aspect_ratio_min $ASPECT_RATIO_MIN \
    --aspect_ratio_max $ASPECT_RATIO_MAX \
    --port $PORT
