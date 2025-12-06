#!/bin/bash

# Export scored images to HuggingFace dataset format
# Edit OUTPUT_DIR below, then run: ./scripts/export_scores.sh

export PYTHONPATH=.venv/bin/python

# Configuration
OUTPUT_DIR="./scored_images"

# Build command
CMD="python -m src.dataset.score_images export-scores --output_dir $OUTPUT_DIR"

# Export
$CMD
