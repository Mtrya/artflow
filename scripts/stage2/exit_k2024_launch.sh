#!/bin/bash
# 2.3a-followup: exit k=20 / k=24 at the WINNER arch (h1152 d24 mod=layer),
# 16K steps AdamW, endpoints aligned with s2-wide (k=28, AdamW) for direct
# comparison. Rationale: 2.3a mapped k in {8,16,28} at the old screen shape;
# the plateau near the top is unmapped. Runs at priority 1 (LOW, preemptible)
# per the relaxed budget policy (2026-09-06): idle cards run our work whenever
# free; jobs may be stopped anytime. Early-exit compute savings (truncating
# the frozen encoder at layer k) are NOT in this run's code path — features are
# identical either way, so the quality comparison is valid; the savings
# materialize only if an early-exit encoder forward is implemented later.
WS=可上网GPU资源
PRJ=自动化科研
GRP=4090-cuda13.2-2
IMG=artflow-base:torch29-cu128
CMD='bash /inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow/repo/scripts/stage2/run_arm.sh'
BASE=(--workspace "$WS" --project "$PRJ" --group "$GRP" -q 1,10,100 --image "$IMG" --nodes 1 --priority 1)

submit() {  # submit <jobname> <KEY=VALUE...>
    local name=$1; shift
    local args=("${BASE[@]}" -n "$name")
    for kv in "$@"; do args+=(--env "$kv"); done
    args+=(-c "$CMD")
    inspire job create "${args[@]}"
}

submit s2-exit-k20 RUN_NAME=s2-exit-k20 HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000 EXIT=20
submit s2-exit-k24 RUN_NAME=s2-exit-k24 HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=24 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000 EXIT=24

echo EXIT-K2024-SUBMITTED
