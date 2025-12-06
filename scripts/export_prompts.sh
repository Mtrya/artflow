#!/bin/bash

# Export generated prompts to HuggingFace dataset format
# Edit OUTPUT_DIR below, then run: ./scripts/export_prompts.sh

export PYTHONPATH=.venv/bin/python

# Configuration
OUTPUT_DIR="./scored_images"

# Export
python -m src.dataset.score_images export-prompts \
    --output_dir "$OUTPUT_DIR"
