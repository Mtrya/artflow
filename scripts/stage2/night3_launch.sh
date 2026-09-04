#!/bin/bash
# Night batch 3 (next trough after 2.2b/2.3a/2.4 verdicts): all mod=layer (D11).
#   2.2c iso-FLOP pair: s2-big (h1152 d33, 16K), s2-small (h1024 d25, iso-FLOP
#       steps = 16K x 664.264M/399.196M = 26623)
#   2.2d stream screens (8K): s2-hybrid (d2x8+s14), s2-double (d2x15)
#       (2.2d all-single screen = s2-deep's trajectory — same shape, 8K mark free;
#        if deep wins 2.2b its 16K run doubles as the 2.2d confirm)
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

submit s2-big RUN_NAME=s2-big HIDDEN_SIZE=1152 SINGLE_STREAM_DEPTH=33 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=16000
submit s2-small RUN_NAME=s2-small HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=25 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=0 DOUBLE_MOD=none MAX_STEPS=26623
submit s2-hybrid RUN_NAME=s2-hybrid HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=14 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=8 DOUBLE_MOD=layer MAX_STEPS=8000
submit s2-double RUN_NAME=s2-double HIDDEN_SIZE=1024 SINGLE_STREAM_DEPTH=0 SINGLE_MOD=layer DOUBLE_STREAM_DEPTH=15 DOUBLE_MOD=layer MAX_STEPS=8000

echo NIGHT3-SUBMITTED
