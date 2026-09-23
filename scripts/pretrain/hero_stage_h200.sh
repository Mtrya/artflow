#!/bin/bash
# Four-H200 hero: a fresh run at 256p, then exact-state stage continuation.
# The source snapshot and validated stage inputs are explicit immutable paths.
# ARTFLOW_PRETRAIN_PACKAGE=src.train supports the retained, pre-rename NVIDIA
# snapshot used for H200 qualification; current source uses src.pretrain.
set -euo pipefail
W=${ARTFLOW_ROOT:?Set the shared ArtFlow artifact root}
SOURCE=${ARTFLOW_SOURCE:?Set the qualified NVIDIA source snapshot}
INPUTS=${ARTFLOW_H200_INPUTS:?Set the qualified H200 stage-input directory}
PACKAGE=${ARTFLOW_PRETRAIN_PACKAGE:-src.pretrain}
STAGE=${1:?usage: hero_stage_h200.sh <256p|640p|896p>}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOTAL_STEPS=600000
case "$PACKAGE" in
  src.train|src.pretrain) ;;
  *) echo "unsupported pretraining package: $PACKAGE" >&2; exit 2 ;;
esac
case "$STAGE" in
  256p) START=0; END=450000; PREV= ;;
  640p) START=450000; END=570000; PREV=256p ;;
  896p) START=570000; END=600000; PREV=640p ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
test -f "$INPUTS/hero-$STAGE.toml"
test -f "$INPUTS/h200.toml"
test -f "$INPUTS/inputs.sha256"
sha256sum --check --quiet "$INPUTS/inputs.sha256"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
export TORCHINDUCTOR_CACHE_DIR=${ARTFLOW_H200_COMPILER_CACHE:?Set the qualified H200 compiler cache}
export TRITON_CACHE_DIR=${ARTFLOW_H200_TRITON_CACHE:?Set the qualified H200 Triton cache}
export TORCH_HOME="$W/models/torch_home"
unset ARTFLOW_LOG_SHAPES ARTFLOW_INFRA_METRICS ARTFLOW_INFRA_IDENTITIES
unset ARTFLOW_TRACE_START ARTFLOW_TRACE_STEPS
export SWANLAB_LOG_DIR="$W/runs/swanlog/h200"
if [ "${SWANLAB_MODE:-cloud}" = cloud ]; then
  test -r "$W/secrets/swanlab.netrc"
  mkdir -p "$HOME/.swanlab"
  install -m 600 "$W/secrets/swanlab.netrc" "$HOME/.swanlab/.netrc"
fi
cd "$SOURCE"

RUN="h200-hero-$STAGE"
RUN_DIR="$W/runs/$RUN"
LOG="$W/data/logs/$RUN.log"
mkdir -p "$RUN_DIR" "$W/data/logs"
exec 9>>"$RUN_DIR/.writer.lock"
if ! flock -n 9; then
  echo "another H200 launcher holds the writer lock for $RUN_DIR" >&2
  exit 1
fi

# Pick only complete checkpoints from this H200 run. Never fall back to a
# checkpoint from the 4090 or Ascend hero, even if this directory is empty.
latest_checkpoint() {
  python3 - "$1" "$TOTAL_STEPS" "$PACKAGE" <<'PY'
import importlib
from pathlib import Path
import re
import sys
root, horizon, package = sys.argv[1:]
validate = importlib.import_module(package + '.stage_control').validate_checkpoint
paths = sorted((p for p in Path(root).glob('checkpoint_step_*')
                if p.is_dir() and re.fullmatch(r'checkpoint_step_\d+', p.name)),
               key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
for path in paths:
    try:
        validate(path, max_steps=int(horizon), require_record=True,
                 scheduler_count=2, use_ema=True, world_size=4)
    except (ValueError, OSError) as exc:
        print(f'Skipping unusable checkpoint {path.name}: {exc}', file=sys.stderr)
        continue
    print(path)
    break
else:
    if paths:
        raise SystemExit('Existing H200 checkpoints are unusable; refusing a fresh restart')
PY
}

RESUME_ARGS=()
OWN=$(latest_checkpoint "$RUN_DIR")
if [ -n "$OWN" ]; then
  python3 -m "$PACKAGE.stage_control" "$OWN" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --min-step "$START" --world-size 4
  RESUME_ARGS=(--resume "$OWN" --resume_full)
elif [ -n "$PREV" ]; then
  PREVIOUS=$(latest_checkpoint "$W/runs/h200-hero-$PREV")
  if [ -z "$PREVIOUS" ]; then
    echo "missing completed H200 predecessor for $STAGE" >&2
    exit 1
  fi
  python3 -m "$PACKAGE.stage_control" "$PREVIOUS" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --expected-step "$START" --world-size 4
  RESUME_ARGS=(--resume "$PREVIOUS" --resume_full --reset_sampler)
fi

CONFIG_DIR=$(mktemp -d /tmp/artflow-h200-config.XXXXXX)
OVERRIDE="$CONFIG_DIR/stage.toml"
cleanup() {
  rc=$?
  if (( rc != 0 )) && [ -f "$LOG" ]; then
    echo "H200_TRAIN_FAILED exit=$rc" >&2
    tail -n 100 "$LOG" >&2
  fi
  rm -f "$OVERRIDE"
  rmdir "$CONFIG_DIR"
  exit "$rc"
}
trap cleanup EXIT
if [ "$STAGE" = 256p ]; then
  GRID_STEPS="$END"
else
  GRID_STEPS="$START, $((START + 2000)), $END"
fi
cat > "$OVERRIDE" <<EOF
[train]
max_steps = $TOTAL_STEPS
stop_at_step = $END
[eval]
grid_steps = [$GRID_STEPS]
[paths]
output_dir = "$W/runs"
EOF
echo "H200_LAUNCH stage=$STAGE start=$START end=$END resume=${OWN:-${PREVIOUS:-fresh}}"
TRAIN_ENTRY=(-m "$PACKAGE.train")
if [ "${SWANLAB_MODE:-cloud}" = offline ]; then
  TRAIN_ENTRY=("$SCRIPT_DIR/swanlab_offline_train.py" "$PACKAGE.train")
fi
python3 -m torch.distributed.run --nproc_per_node=4 "${TRAIN_ENTRY[@]}" \
  --config configs/base.toml --config "$INPUTS/hero-$STAGE.toml" \
  --config configs/hero.toml --config "$INPUTS/h200.toml" --config "$OVERRIDE" \
  --compile_dynamic --compile_autotune --disable_ddp_compile_split \
  --hoist_double_rope --native_flash_varlen --real_rope \
  --muon_compile_square_ns --gpu_health_snapshot --local_cache_clear \
  --run_name "$RUN" "${RESUME_ARGS[@]}" >> "$LOG" 2>&1
