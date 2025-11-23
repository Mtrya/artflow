#!/bin/bash

export PYTHONPATH=.venv/bin/python

python scripts/precompute_dataset.py \
    --dataset_name "kaupane/wikiart-captions-monet" \
    --split "train" \
    --image_field "image" \
    --caption_fields "" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256)]" \
    --output_dir "./precomputed_dataset/wikiart-captions-monet@256p" \
    --batch_size 50 \
    --range -1

python scripts/precompute_dataset.py \
    --dataset_name "xingjianleng/laion_aesthetics_v2_6.5plus" \
    --split "train" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256)]" \
    --output_dir "./precomputed_dataset/laion_aesthetics_v2_6.5plus@256p" \
    --batch_size 50 \
    --range 5000

python scripts/precompute_dataset.py \
    --dataset_name "kaupane/wikiart-captions" \
    --split "train" \
    --image_field "image" \
    --caption_fields "qwen-direct, qwen-reverse, qwen-spatial, wikiart-caption" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
    --output_dir "./precomputed_dataset/wikiart-captions@256p" \
    --batch_size 50 \
    --range -1

python scripts/precompute_dataset.py \
    --dataset_name "xingjianleng/laion_aesthetics_v2_6.0plus" \
    --split "train" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
    --output_dir "./precomputed_dataset/laion_aesthetics_v2_6.0plus@256p" \
    --batch_size 50 \
    --range -1

python scripts/precompute_dataset.py \
    --dataset_name "kaupane/wikiart-captions" \
    --split "train" \
    --image_field "image" \
    --caption_fields "qwen-direct, qwen-reverse, qwen-spatial, wikiart-caption" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]" \
    --output_dir "./precomputed_dataset/wikiart-captions@640p" \
    --batch_size 20 \
    --range -1