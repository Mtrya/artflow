#!/bin/bash

export PYTHONPATH=.venv/bin/python

python -m src.train.precompute \
    --dataset_name "laion/relaion-art" \
    --split "train" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
    --output_dir "./precomputed_dataset/relaion-multi-aesthetic@256p" \
    --batch_size 500 \
    --device "cpu" \
    --non_zh_drop_prob 0.8
