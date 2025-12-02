#!/bin/bash

export PYTHONPATH=.venv/bin/python

#python -m src.train.precompute \
#    --dataset_name "kaupane/wikiart-captions" \
#    --split "train" \
#    --image_field "image" \
#    --caption_fields "wikiart-caption, qwen-direct, qwen-spatial, qwen-reverse" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]" \
#    --output_dir "./precomputed_dataset/wikiart-captions@640p" \
#    --batch_size 12 \

python -m src.train.precompute \
    --dataset_name "laion/relaion2B-multi-aesthetic" \
    --split "train[:10000000]" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
    --output_dir "./precomputed_dataset/relaion-multi-aesthetic@256p" \
    --batch_size 500 \
    --device "cpu" \
    --non_zh_drop_prob 0.9

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
