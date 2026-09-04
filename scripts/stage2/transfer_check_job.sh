#!/bin/bash
# 2.1 post-arm transfer check: sample the fixed prompt suite at 640p from both
# final checkpoints (EMA weights). Qualitative resolution-extrapolation read.
W=${W:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
export HF_HOME=$W/cache/hf
export TOKENIZERS_PARALLELISM=false

cd "$W/repo"
for arm in new old; do
    EXTRA=""
    if [ "$arm" = "new" ]; then EXTRA="--rope_centered"; fi
    python scripts/stage2/transfer_check.py \
        --ckpt "$W/runs/stage2/s2-rope-$arm/checkpoint_step_006000/ema_weights.pt" \
        --text_encoder_path "$W/models/Qwen3-0.6B" \
        --vae_path "$W/models/e2e-qwenimage-vae" \
        --eval_dataset_path "$W/precomputed_dataset/light-eval@256p" \
        --scale 2.5 \
        --out_dir "$W/runs/stage2/transfer/s2-rope-$arm-640p" \
        $EXTRA
done
echo TRANSFER-DONE
