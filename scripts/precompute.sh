#!/bin/bash

export PYTHONPATH=.venv/bin/python

#python scripts/precompute_dataset.py \
#    --dataset_name "kaupane/wikiart-captions-monet" \
#    --split "train" \
#    --image_field "image" \
#    --caption_fields "" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(256,256)]" \
#    --output_dir "./precomputed_dataset/wikiart-captions-monet@256p" \
#    --batch_size 50 \

#python scripts/precompute_dataset.py \
#    --dataset_name "alfredplpl/artbench-pd-256x256" \
#    --split "train[:2048]" \
#    --image_field "url" \
#    --caption_fields "caption, artist" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
#    --output_dir "./precomputed_dataset/light-eval@256p" \
#    --batch_size 50 \

#python scripts/precompute_dataset.py \
#    --dataset_name "alfredplpl/artbench-pd-256x256" \
#    --split "train[2048:]" \
#    --image_field "url" \
#    --caption_fields "caption, artist" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
#   --output_dir "./precomputed_dataset/heavy-eval@256p" \
#    --batch_size 50 \

#python scripts/precompute_dataset.py \
#    --dataset_name "kaupane/wikiart-captions" \
#    --split "train" \
#    --image_field "image" \
#   --caption_fields "wikiart-caption, qwen-direct, qwen-spatial, qwen-reverse" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
#    --output_dir "./precomputed_dataset/wikiart-captions@256p" \
#    --batch_size 50 \

#python scripts/precompute_dataset.py \
#    --dataset_name "kaupane/wikiart-captions" \
#    --split "train" \
#    --image_field "image" \
#    --caption_fields "wikiart-caption, qwen-direct, qwen-spatial, qwen-reverse" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]" \
#    --output_dir "./precomputed_dataset/wikiart-captions@640p" \
#    --batch_size 12 \

#python scripts/precompute_dataset.py \
#    --dataset_name "OpenFace-CQUPT/HumanCaption-HQ-311K" \
#    --split "train[:240000]" \
#    --image_field "url" \
#    --caption_fields "human_caption_hq" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
#    --output_dir "./precomputed_dataset/human-caption-hq@256p" \
#    --batch_size 50 \
#    --device "cpu"

#python scripts/precompute_dataset.py \
#    --dataset_name "OpenFace-CQUPT/FaceCaptionHQ-4M" \
#    --split "train[:240000]" \
#    --image_field "url" \
#    --caption_fields "image_caption, face_caption, image_short_caption" \
#    --vae_path "REPA-E/e2e-qwenimage-vae" \
#    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
#    --output_dir "./precomputed_dataset/face-caption-hq@256p" \
#    --batch_size 50 \
#    --device "cpu"

python scripts/precompute_dataset.py \
    --dataset_name "OpenFace-CQUPT/HumanCaption-HQ-311K" \
    --split "train[-40000:]" \
    --image_field "url" \
    --caption_fields "human_caption_hq" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]" \
    --output_dir "./precomputed_dataset/human-caption-hq@640p" \
    --batch_size 25 \
    --device "cpu"

python scripts/precompute_dataset.py \
    --dataset_name "OpenFace-CQUPT/FaceCaptionHQ-4M" \
    --split "train[-40000:]" \
    --image_field "url" \
    --caption_fields "image_caption, face_caption, image_short_caption" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]" \
    --output_dir "./precomputed_dataset/face-caption-hq@640p" \
    --batch_size 25 \
    --device "cpu"

python scripts/precompute_dataset.py \
    --dataset_name "laion/relaion2B-multi-aesthetic" \
    --split "train[-40000:]" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(640,640), (848,480), (480,848), (736,560), (560,736)]" \
    --output_dir "./precomputed_dataset/relaion-multi-aesthetic@640p" \
    --batch_size 25 \
    --device "cpu"

python scripts/precompute_dataset.py \
    --dataset_name "laion/relaion2B-multi-aesthetic" \
    --split "train[:1000000]" \
    --image_field "URL" \
    --caption_fields "TEXT" \
    --vae_path "REPA-E/e2e-qwenimage-vae" \
    --resolution_buckets "[(256,256), (336,192), (192,336), (288,224), (224,288)]" \
    --output_dir "./precomputed_dataset/relaion-multi-aesthetic-1@256p" \
    --batch_size 80 \
    --device "cpu"