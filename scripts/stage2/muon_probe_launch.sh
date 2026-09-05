#!/bin/bash
# 2.5 Muon LR probe batch: 3 arms x 3K steps at the 2.2 winner arch
# (h1152 d24 all-single mod=layer, centered-grid RoPE). AdamW baseline for the
# probe comparison = s2-wide (same arch, 3K marks) - zero extra cost.
# Usage: launch when cards free (user preference). Confirm arm comes later at
# the best probed LR vs s2-wide's 16K (16K muon run).
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

SHAPE="HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=3000"
submit s2-muon-lr01 RUN_NAME=s2-muon-lr01 $SHAPE OPTIMIZER=muon MUON_LR=0.01
submit s2-muon-lr02 RUN_NAME=s2-muon-lr02 $SHAPE OPTIMIZER=muon MUON_LR=0.02
submit s2-muon-lr04 RUN_NAME=s2-muon-lr04 $SHAPE OPTIMIZER=muon MUON_LR=0.04

echo MUON-PROBES-SUBMITTED
