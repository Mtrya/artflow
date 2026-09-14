#!/bin/bash
# Hero run: one resolution stage per invocation, chained across jobs.
#
#   bash jobs/hero_stage.sh 256p <TOTAL_STEPS>
#
# TOTAL_STEPS is the shared optimizer-step count of the whole run. Every
# stage is launched with max_steps = TOTAL_STEPS so the cosine schedule and
# the caption curriculum stay continuous across stage boundaries. stop_at_step
# supplies the separate cumulative 75:20:5 endpoints. Checkpoint preflight checks
# the global horizon, expected step, and completed file inventory before launch.
# Wall-time termination is not a substitute for exact stage stopping.
#
# Resume behaviour:
#   - If this stage's run directory already holds a checkpoint, resume it
#     with its sampler sidecar (crash/preemption chaining within a stage).
#   - Otherwise, for 640p/896p, bootstrap from the previous stage's latest
#     endpoint checkpoint directly, with --reset_sampler: training state carries
#     over while the new resolution's data order starts fresh. The predecessor
#     checkpoint is never copied or modified.
set -eu
W=${ARTFLOW_ROOT:?Set ARTFLOW_ROOT to the shared-workspace root used on the cluster}
STAGE=${1:?usage: hero_stage.sh <256p|640p|896p> <total_steps>}
TOTAL_STEPS=${2:?total optimizer steps shared by all stages}
if ! [[ $TOTAL_STEPS =~ ^[1-9][0-9]*$ ]]; then
  echo "total_steps must be a positive decimal integer" >&2
  exit 2
fi

case $STAGE in
  256p) ACCUM=1; PREV=; START=0; END=$((TOTAL_STEPS * 75 / 100)); GRID_STEPS="$END" ;;
  640p) ACCUM=5; PREV=256p; START=$((TOTAL_STEPS * 75 / 100));
        END=$((TOTAL_STEPS * 95 / 100)); GRID_STEPS="$START, $((START + 2000)), $END" ;;
  896p) ACCUM=7; PREV=640p; START=$((TOTAL_STEPS * 95 / 100));
        END=$TOTAL_STEPS; GRID_STEPS="$START, $((START + 2000)), $END" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
if (( END <= START )); then
  echo "total_steps produces an empty $STAGE stage" >&2
  exit 2
fi

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
  ls -d "$1"/checkpoint_step_[0-9]* 2>/dev/null | sort -V | tail -1
}

RESUME_ARGS=()
OWN=$(latest_ckpt "$RUN_DIR")
if [ -n "$OWN" ]; then
  python3 -m src.train.stage_control "$OWN" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --min-step "$START"
  RESUME_ARGS=(--resume "$OWN" --resume_full)
elif [ -n "$PREV" ]; then
  PREV_CKPT=$(latest_ckpt "$W/runs/hero-$PREV")
  if [ -z "$PREV_CKPT" ]; then
    echo "no checkpoint from hero-$PREV to bootstrap from" >&2
    exit 1
  fi
  python3 -m src.train.stage_control "$PREV_CKPT" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --expected-step "$START"
  RESUME_ARGS=(--resume "$PREV_CKPT" --resume_full --reset_sampler)
fi

# A private temp file avoids concurrent invocations overwriting each other's config.
OVERRIDE=$(mktemp /tmp/artflow-hero-override.XXXXXX.toml)
trap 'rm -f "$OVERRIDE"' EXIT
cat > "$OVERRIDE" <<EOF
[train]
max_steps = $TOTAL_STEPS
stop_at_step = $END
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
  --config "$OVERRIDE" \
  --step_breakdown \
  --run_name "$RUN" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  >> "$W/data/logs/$RUN.log" 2>&1
