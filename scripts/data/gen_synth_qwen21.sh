#!/bin/bash
# Render the synthetic validation batch with Qwen-Image 2.1 (GPU job entry).
#
# The runtime is installed from the wheelhouse staged by
# fetch_synth_qwen21.sh, because no PyPI release carries QwenImage21Pipeline
# yet.  One worker per GPU then walks every WORKERS-th row of the grid; a
# worker skips rows whose picture already exists, so a job that dies can be
# started again and carries on where it stopped.
#
#   WORKERS=8 bash scripts/data/gen_synth_qwen21.sh
#
# Env: GRID, OUT, WORKERS, STEPS, LIMIT, REPO, ARTFLOW_ROOT.
set -u

W=${ARTFLOW_ROOT:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
REPO=${REPO:-$W/repo-synth-0929}
PY=${PY:-python}
GRID=${GRID:-$W/data/meta/synth_qwen21/grid.jsonl}
OUT=${OUT:-$W/data/synth/qwen21/images}
MODEL=${MODEL:-$W/models/Qwen-Image-2.1}
WORKERS=${WORKERS:-8}
STEPS=${STEPS:-40}
LIMIT=${LIMIT:-0}
WHEELS=$W/wheels-qwen21

cd "$REPO" || exit 1
mkdir -p "$OUT"

echo "[gen] $(date -Is) installing runtime from $WHEELS"
WHEELS=$WHEELS PY=$PY bash "$REPO/scripts/data/install_qwen21_runtime.sh" || exit 1
[ -f "$GRID" ] || { echo "[gen] grid missing: $GRID"; exit 2; }
wc -l "$GRID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

LIMIT_ARG=""
[ "$LIMIT" != "0" ] && LIMIT_ARG="--limit $LIMIT"

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
  echo "[gen] worker $i start $(date -Is)"
  CUDA_VISIBLE_DEVICES=$i $PY -m scripts.data.synth_generate \
    --prompts "$GRID" --out-dir "$OUT" --recipe qwen-image-2.1 \
    --model "$MODEL" --shard "$i" --shards "$WORKERS" $LIMIT_ARG \
    > "$OUT/worker_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[gen] worker $i FAILED (see $OUT/worker_$i.log)"
    fail=1
  fi
done
find "$OUT" -maxdepth 1 -name '*.jpg' | wc -l
echo "[gen] done fail=$fail $(date -Is)"
echo GEN-SYNTH-DONE
exit $fail
