#!/bin/bash
# Night batch 2 (2026-09-05 19:00+ trough): post-2.2a arms, all mod=layer
# (D11). Uses the env-driven run_arm.sh on GPFS.
#   2.2b pair (16K, ~11-12h each): s2-wide (h1152 d24), s2-deep (h1024 d30)
#   2.3a exit screen (4K, ~3h): s2-exit-k8, s2-exit-k16
#        (k=28 baseline = s2-mod-layer's 4K/8K curves, reused — identical config)
#   2.4 mix sanity (8K, ~6h): s2-mix-old (MIX=old), s2-mix-new (MIX=new, h1024 d24)
WS=可上网GPU资源
PRJ=自动化科研
GRP=4090-cuda13.2-2
IMG=artflow-base:torch29-cu128
CMD='bash /inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow/repo/scripts/stage2/run_arm.sh'
BASE=(--workspace "$WS" --project "$PRJ" --group "$GRP" -q 1,10,100 --image "$IMG" --nodes 1)

submit() {  # submit <jobname> <KEY=VALUE...>
    local name=$1; shift
    local args=("${BASE[@]}" -n "$name")
    for kv in "$@"; do args+=(--env "$kv"); done
    args+=(-c "$CMD")
    inspire job create "${args[@]}"
}

submit s2-wide RUN_NAME=s2-wide HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000
submit s2-deep RUN_NAME=s2-deep HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=30 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000
submit s2-exit-k8 RUN_NAME=s2-exit-k8 HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=4000 EXIT=8
submit s2-exit-k16 RUN_NAME=s2-exit-k16 HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=4000 EXIT=16
submit s2-mix-old RUN_NAME=s2-mix-old HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=8000 MIX=old
submit s2-mix-new RUN_NAME=s2-mix-new HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=8000 MIX=new

echo NIGHT2-SUBMITTED
