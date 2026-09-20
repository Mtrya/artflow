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
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=$W/models/torch_home
# Match the measured execution path; optional per-micro logging and profilers
# belong to diagnostic launches, not the costed hero command.
unset ARTFLOW_LOG_SHAPES ARTFLOW_INFRA_METRICS ARTFLOW_TRACE_START ARTFLOW_TRACE_STEPS
source "$W/jobs/swanlab_login.sh"
cd "$W/repo"

RUN=hero-$STAGE
RUN_DIR=$W/runs/$RUN
mkdir -p "$W/data/logs" "$RUN_DIR"
# Hold this descriptor through torchrun so two submissions cannot select the
# same resume point and overwrite the same checkpoint directory concurrently.
# Kernel locks disappear when the holder exits; a leftover file is not a lock.
if ! command -v flock >/dev/null; then
  echo "flock is required for single-writer protection of $RUN_DIR" >&2
  exit 1
fi
exec 9>>"$RUN_DIR/.writer.lock"
if ! flock -n 9; then
  echo "another hero launcher holds the writer lock for $RUN_DIR; not starting" >&2
  exit 1
fi

latest_ckpt () {
  ls -d "$1"/checkpoint_step_[0-9]* 2>/dev/null | sort -V | tail -1
}

RESUME_ARGS=()
OWN=$(latest_ckpt "$RUN_DIR")
if [ -n "$OWN" ]; then
  python3 -m src.pretrain.stage_control "$OWN" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --min-step "$START"
  RESUME_ARGS=(--resume "$OWN" --resume_full)
elif [ -n "$PREV" ]; then
  PREV_CKPT=$(latest_ckpt "$W/runs/hero-$PREV")
  if [ -z "$PREV_CKPT" ]; then
    echo "no checkpoint from hero-$PREV to bootstrap from" >&2
    exit 1
  fi
  python3 -m src.pretrain.stage_control "$PREV_CKPT" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --expected-step "$START"
  RESUME_ARGS=(--resume "$PREV_CKPT" --resume_full --reset_sampler)
fi

# Render the exact versioned shard weights, never regenerate them from live row counts.
# A private directory avoids mutating shared configs or other invocations' inputs.
CONFIG_DIR=$(mktemp -d /tmp/artflow-hero-config.XXXXXX)
INPUT_CONFIG="$CONFIG_DIR/inputs.toml"
OVERRIDE="$CONFIG_DIR/override.toml"
trap 'rm -f "$INPUT_CONFIG" "$OVERRIDE"; rmdir "$CONFIG_DIR"' EXIT
python3 -m scripts.bench.render_hero_stage "$STAGE" --root "$W" --out "$INPUT_CONFIG"
cat > "$OVERRIDE" <<EOF
[train]
max_steps = $TOTAL_STEPS
stop_at_step = $END
gradient_accumulation_steps = $ACCUM
[eval]
grid_steps = [$GRID_STEPS]
EOF

python3 -m torch.distributed.run --nproc_per_node=8 -m src.pretrain.train \
  --config configs/base.toml \
  --config "$INPUT_CONFIG" \
  --config configs/hero.toml \
  --config "$OVERRIDE" \
  --compile_dynamic --compile_autotune --disable_ddp_compile_split \
  --hoist_double_rope --native_flash_varlen --real_rope \
  --muon_compile_square_ns --gpu_health_snapshot --local_cache_clear \
  --run_name "$RUN" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  >> "$W/data/logs/$RUN.log" 2>&1
