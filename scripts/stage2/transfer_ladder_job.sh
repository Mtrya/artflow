#!/bin/bash
# 2.1 tie-break ladder: sample the fixed prompt suite at 480p/384p/320p
# (1.875x/1.5x/1.25x latent scale) from both final checkpoints (EMA weights).
# User-amended rule (2026-09-05): complexity ties, so descend 480p -> 384p -> 320p
# until one arm shows clearly fewer artifacts; if still tied at 320p, new RoPE
# wins on the Qwen-Image adoption prior.
W=${W:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
export HF_HOME=$W/cache/hf
export TOKENIZERS_PARALLELISM=false

cd "$W/repo"
for arm in new old; do
    EXTRA=""
    if [ "$arm" = "new" ]; then EXTRA="--rope_centered"; fi
    for spec in 1.875:480p 1.5:384p 1.25:320p; do
        scale="${spec%%:*}"; tag="${spec##*:}"
        python scripts/stage2/transfer_check.py \
            --ckpt "$W/runs/stage2/s2-rope-$arm/checkpoint_step_006000/ema_weights.pt" \
            --text_encoder_path "$W/models/Qwen3-0.6B" \
            --vae_path "$W/models/e2e-qwenimage-vae" \
            --eval_dataset_path "$W/precomputed_dataset/light-eval@256p" \
            --scale "$scale" \
            --out_dir "$W/runs/stage2/transfer/s2-rope-$arm-$tag" \
            $EXTRA
    done
done
echo TRANSFER-LADDER-DONE
