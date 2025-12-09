#!/bin/bash

# Manual image scoring with VLM prompt generation
# Edit the configuration below, then run: ./scripts/score_images.sh

export PYTHONPATH=.venv/bin/python

# Dataset configuration (supports multiple datasets with weights)
DATASET_NAME="kaupane/wikiart-captions"  # legacy single-dataset fields retained for compatibility
IMAGE_FIELD="image"
TARGET_SAMPLES=5000

# Repeatable dataset specs: name=...,image_field=...,weight=...
DATASETS=(
  "name=kaupane/wikiart-captions-monet,image_field=image,weight=0.2"
  "name=CaptionEmporium/laion-pop-llama3.2-11b,image_field=url,weight=0.4"
  "name=OpenFace-CQUPT/HumanCaption-10M,image_field=url,weight=0.4"
)

# Quality filters
MIN_RESOLUTION=640
ASPECT_RATIO_MIN=0.8
ASPECT_RATIO_MAX=1.25

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
