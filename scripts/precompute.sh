#!/bin/bash

# --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]"
# --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]"
# --resolution_buckets "[(1408,768), (1584,672), (848,1264), (1024,1024)]"

export PYTHONPATH=.venv/bin/python

python -m src.train.precompute \
    --dataset_name "laion/relaion-art" \
    --split "train" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
    --index_offset 0 \
    --output_dir "./precomputed_dataset/relaion-multi-aesthetic@256p" \
    --batch_size 500 \
    --device "cpu" \
    --non_zh_drop_prob 0.8
