#!/bin/bash

# Manual image scoring with VLM prompt generation
# Edit the configuration below, then run: ./scripts/score_images.sh

export PYTHONPATH=.venv/bin/python

IMAGE_FIELD="image"
TARGET_SAMPLES=5000

# Repeatable dataset specs: name=...,image_field=...,weight=...
DATASETS=(
  "name=kaupane/wikiart-captions,image_field=image,weight=0.2"
  "name=kaupane/relaion-art-recap-zh,image_field=url,weight=0.3"
  "name=kaupane/vintage-photography-captions,image_field=image_url,weight=0.3"
  "name=kaupane/human-recaption,image_field=image_url,weight=0.2"
)

# Quality filters
MIN_RESOLUTION=512
ASPECT_RATIO_MIN=0.6
ASPECT_RATIO_MAX=1.67

# Output
OUTPUT_DIR="./scored_images"

# Interface
PORT=7860

# Launch scoring interface

ARGS=(
  python -m src.dataset.score_images score
  --target_samples "$TARGET_SAMPLES"
  --output_dir "$OUTPUT_DIR"
  --min_resolution $MIN_RESOLUTION
  --aspect_ratio_min $ASPECT_RATIO_MIN
  --aspect_ratio_max $ASPECT_RATIO_MAX
  --port $PORT
)

if [ ${#DATASETS[@]} -gt 0 ]; then
  for ds in "${DATASETS[@]}"; do
    ARGS+=(--dataset "$ds")
  done
else
  ARGS+=(--dataset_name "$DATASET_NAME" --image_field "$IMAGE_FIELD")
fi

"${ARGS[@]}"
