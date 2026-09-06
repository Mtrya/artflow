#!/bin/bash
# Re-launch s2-exit-k20/k24 at HIGH priority (4) resuming from ckpt 6000.
# Priority policy (user, 2026-09-06): stage 2-4 experiments run medium-high;
# only the stage-5 hero run sits at LOW (preemptible idle-fill).
WS=可上网GPU资源
PRJ=自动化科研
GRP=4090-cuda13.2-2
IMG=artflow-base:torch29-cu128
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow
CMD='bash '$W'/repo/scripts/stage2/run_arm.sh'
BASE=(--workspace "$WS" --project "$PRJ" --group "$GRP" -q 1,10,100 --image "$IMG" --nodes 1 --priority 4)

submit() {
    local name=$1; shift
    local args=("${BASE[@]}" -n "$name")
    for kv in "$@"; do args+=(--env "$kv"); done
    args+=(-c "$CMD")
    inspire job create "${args[@]}"
}

submit s2-exit-k20 RUN_NAME=s2-exit-k20 HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000 EXIT=20 RESUME=$W/runs/stage2/s2-exit-k20/checkpoint_step_006000 RESUME_FULL=1
submit s2-exit-k24 RUN_NAME=s2-exit-k24 HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000 EXIT=24 RESUME=$W/runs/stage2/s2-exit-k24/checkpoint_step_006000 RESUME_FULL=1

echo EXIT-K2024-RESUBMITTED-HIGH
