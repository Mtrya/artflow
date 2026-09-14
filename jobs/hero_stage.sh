#!/bin/bash
# Hero run: one resolution stage per invocation, chained across jobs.
#
#   bash jobs/hero_stage.sh 256p <TOTAL_STEPS>
#
# TOTAL_STEPS is the shared optimizer-step count of the whole run. Every
# stage is launched with max_steps = TOTAL_STEPS so the cosine schedule and
# the caption curriculum stay continuous across stage boundaries. This launcher
# does not yet enforce the 75:20:5 stage endpoints: separate stop_at_step control
# and predecessor-checkpoint validation are required before hero launch.
# Wall-time termination is not a substitute for exact stage stopping.
#
# Resume behaviour:
#   - If this stage's run directory already holds a checkpoint, resume it
#     with its sampler sidecar (crash/preemption chaining within a stage).
#   - Otherwise, for 640p/896p, bootstrap from the previous stage's latest
#     checkpoint: copy it into this stage's run directory and drop the
#     sampler sidecars, so weights/optimizer/scheduler/EMA carry over while
#     the new resolution's data order starts fresh.
set -eu
W=${ARTFLOW_ROOT:?Set ARTFLOW_ROOT to the shared-workspace root used on the cluster}
STAGE=${1:?usage: hero_stage.sh <256p|640p|896p> <total_steps>}
TOTAL_STEPS=${2:?total optimizer steps shared by all stages}
if ! [[ $TOTAL_STEPS =~ ^[1-9][0-9]*$ ]]; then
  echo "total_steps must be a positive decimal integer" >&2
  exit 2
fi

case $STAGE in
  256p) ACCUM=1; PREV=; GRID_STEPS="$((TOTAL_STEPS * 75 / 100))" ;;
  640p) ACCUM=5; PREV=256p; START=$((TOTAL_STEPS * 75 / 100));
        GRID_STEPS="$START, $((START + 2000)), $((TOTAL_STEPS * 95 / 100))" ;;
  896p) ACCUM=7; PREV=640p; START=$((TOTAL_STEPS * 95 / 100));
        GRID_STEPS="$START, $((START + 2000)), $TOTAL_STEPS" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

export ARTFLOW_ROOT=$W
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$W/torchinductor-cache
# Per-micro shape log: lets a late OOM be attributed to the exact bucket.
export ARTFLOW_LOG_SHAPES=1
source "$W/jobs/swanlab_login.sh"
cd "$W/repo"

RUN=hero-$STAGE
RUN_DIR=$W/runs/$RUN
mkdir -p "$W/data/logs"

latest_ckpt () {
  ls -d "$1"/checkpoint_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1
}

RESUME_ARGS=()
OWN=$(latest_ckpt "$RUN_DIR")
if [ -n "$OWN" ]; then
  RESUME_ARGS=(--resume "$OWN" --resume_full)
elif [ -n "$PREV" ]; then
  PREV_CKPT=$(latest_ckpt "$W/runs/hero-$PREV")
  if [ -z "$PREV_CKPT" ]; then
    echo "no checkpoint from hero-$PREV to bootstrap from" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR"
  BOOT=$RUN_DIR/$(basename "$PREV_CKPT")
  if [ ! -d "$BOOT" ]; then
    cp -r "$PREV_CKPT" "$BOOT"
    # Fresh sampler for the new resolution: the sidecar encodes the old
    # stage's bucket queues and row order.
    rm -f "$BOOT"/sampler_state_rank_*.pt
  fi
  RESUME_ARGS=(--resume "$BOOT" --resume_full)
fi

cat > /tmp/hero-$STAGE-override.toml <<EOF
[train]
max_steps = $TOTAL_STEPS
gradient_accumulation_steps = $ACCUM
checkpoint_interval = 2000
eval_interval = 10000
ema_decay = 0.9999
ema_update_interval = 1
[optim]
lr_warmup_steps = 5000
muon_lr = 0.02
min_learning_rate = 1.5e-5
[eval]
prompts_file = "assets/eval/hero_monitor_v1.jsonl"
ode_steps = 50
grid_steps = [$GRID_STEPS]
loss_interval = 500
kid_at_end = true
EOF

python3 -m torch.distributed.run --nproc_per_node=8 -m src.train.train \
  --config configs/base.toml \
  --config "$W/configs/hero-$STAGE.toml" \
  --config "$W/configs/ladder-common.toml" \
  --config "$W/configs/ladder-ld533.toml" \
  --config /tmp/hero-$STAGE-override.toml \
  --step_breakdown \
  --run_name "$RUN" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  >> "$W/data/logs/$RUN.log" 2>&1
