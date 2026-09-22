#!/bin/bash
# Hero stage launcher for Ascend 910B: 16 ranks HCCL, one stage per invocation.
#
#   bash hero_stage_ascend.sh <256p|640p|896p> [total_steps=600000]
#
# Persistent layout on sj-ssd3 ($W), so platform auto-restarts re-stage fast:
#   pylibs/         pip --target environment (.$CODE_VER.done marker)
#   repo-ascend/    code bundle (src+scripts+configs+bucket_plans)
#   models/         Qwen3-0.6B + e2e-qwenimage-vae
#   precomputed_dataset/  pulled by ascend_download.sh
#   runs/hero-<stage>/    checkpoints + grids
#   logs/hero-<stage>.log training stdout (watchdog target)
#
# Resume behaviour mirrors jobs/hero_stage_4090.sh: own-stage checkpoint ->
# --resume_full; otherwise bootstrap 640p/896p from the previous stage's
# endpoint with --reset_sampler. The platform restarts the job on
# preemption/failure; this script re-validates and resumes.
#
# OOM watchdog: the 2026-09-22 headroom54g run showed an NPU OOM can leave
# torchrun hanging (dead ranks block survivors in collectives). A sidecar
# loop kills the whole process group as soon as the OOM signature appears in
# the log so the platform can restart the job instead of burning the
# allocation.
set -eu
STAGE=${1:?usage: hero_stage_ascend.sh <256p|640p|896p> [total_steps]}
TOTAL_STEPS=${2:-600000}
CODE_VER=${CODE_VER:-0922n}
W=/inspire/sj-ssd3/project/cq-scientific-cooperation-zone/ky26021/artflow
B=https://hf-mirror.com/datasets/kaupane/artflow-code-bridge/resolve/main

case $STAGE in
  256p) ACCUM=1; PREV=;  START=0;                     END=$((TOTAL_STEPS * 75 / 100)) ;;
  640p) ACCUM=4; PREV=256p; START=$((TOTAL_STEPS * 75 / 100)); END=$((TOTAL_STEPS * 95 / 100)) ;;
  896p) ACCUM=5; PREV=640p; START=$((TOTAL_STEPS * 95 / 100)); END=$TOTAL_STEPS ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

mkdir -p $W/pylibs $W/models $W/logs $W/runs
cd /tmp

dl() {  # dl <url> <dst>: retry up to 4 times, fail loudly
  # speed guard: the HF CDN throttles sj by time window — a stalled (0-byte
  # but open) connection otherwise hangs curl forever and burns the node.
  for i in 1 2 3 4; do
    curl -sL --retry 3 --speed-time 60 --speed-limit 10240 --max-time 900 -o "$2" "$1" && [ -s "$2" ] && return 0
    echo "DL_RETRY $2 (attempt $i)"; sleep 5
  done
  echo "DL_FAIL $2"; exit 1
}

# --- persistent python environment ---------------------------------------
# pip installs are timeboxed: pylibs is already populated (marker only re-runs
# per new CODE_VER, and "already satisfied" scans take ~3min), so a hang means
# the node can't reach the nexus mirror — die fast and let the platform retry
# the job on a healthier node (ascend-hostprobe2 hung here for 75min).
# NOTE: no global pipefail here (latest_ckpt relies on ls|sort|tail masking
# empty globs), so check PIPESTATUS explicitly after each timeboxed install.
pip_tb() {  # pip_tb <secs> <tag> <pip-args...>
  local secs=$1 tag=$2; shift 2
  timeout $secs pip3 install --target=$W/pylibs -q --progress-bar off "$@" 2>&1 | tail -1
  local st=${PIPESTATUS[0]}
  if [ $st -ne 0 ]; then echo "PIP_FAIL $tag st=$st"; exit 1; fi
}
if [ ! -f $W/pylibs/.$CODE_VER.done ]; then
  pip3 config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple/
  pip3 config set global.trusted-host nexus.sii.shaipower.online
  pip_tb 1800 torch torch==2.9.0 torchvision==0.24.0
  pip_tb 600 torch-npu --no-deps torch-npu==2.9.0
  pip_tb 900 deps "accelerate==1.14.0" "transformers==5.16.1" "datasets==5.0.1" "diffusers==0.40.0" "swanlab==0.10.0" safetensors numpy pillow tqdm scipy
  pip_tb 600 metrics --no-deps torchmetrics torch-fidelity lightning-utilities
  touch $W/pylibs/.$CODE_VER.done
fi
export PYTHONPATH=$W/pylibs${PYTHONPATH:+:$PYTHONPATH}
python3 -c "import torch, torch_npu; print('RUNTIME', torch.__version__, torch_npu.__version__, torch.npu.is_available())"

# --- code bundle ----------------------------------------------------------
if [ ! -f $W/repo-ascend/.code_ver ] || [ "$(cat $W/repo-ascend/.code_ver)" != "$CODE_VER" ]; then
  dl $B/artflow-code-$CODE_VER.tar.gz /tmp/code.tar.gz && echo DL_CODE_OK
  rm -rf $W/repo-ascend
  mkdir -p $W/repo-ascend && tar xzf /tmp/code.tar.gz -C $W/repo-ascend
  echo "$CODE_VER" > $W/repo-ascend/.code_ver
fi
cd $W/repo-ascend

# --- models (persistent) --------------------------------------------------
if [ ! -d $W/models/Qwen3-0.6B ]; then
  dl $B/qwen3-0.6b.tar.gz /tmp/qwen3.tar.gz && tar xzf /tmp/qwen3.tar.gz -C $W/models && echo QWEN_OK
fi
if [ ! -d $W/models/e2e-qwenimage-vae ]; then
  dl $B/e2e-qwenimage-vae.tar.gz /tmp/vae.tar.gz && tar xzf /tmp/vae.tar.gz -C $W/models && echo VAE_OK
fi

export ARTFLOW_ROOT=$W
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
if [ -n "${ALLOC_CONF:-}" ]; then export PYTORCH_NPU_ALLOC_CONF="$ALLOC_CONF"; else unset PYTORCH_NPU_ALLOC_CONF || true; fi

# torchair GE backend. The version-matched torchair is the one BUNDLED in
# torch_npu (torch_npu.dynamo.torchair, aliased as top-level `torchair` once
# torch_npu is imported — bench3); the standalone 7.2.0 build in
# pylibs-torchair mismatches this stack's dynamo hooks (srcb7
# BackendCompilerFailed) and must NOT shadow it. We only need pkg_resources
# (pylibs-pkgres, setuptools<81) for torchair's `import pkg_resources`.
COMPILE_ARGS="--no-compile"
if [ "${TORCHAIR:-0}" = "1" ]; then
  export PYTHONPATH=$W/pylibs-pkgres${PYTHONPATH:+:$PYTHONPATH}
  COMPILE_ARGS="--compile --compile_blocks --compile_backend torchair"
fi

# Host-dispatch levers for NPU eager mode (AI core util ~35% is host-bound):
# --foreach_updates batches EMA + grad-scale into two fused calls instead of
# ~400 tiny per-parameter ops (identical math); --npu_fused_adamw is one
# kernel for the whole aux AdamW update (opt-in until numerics smoke-tested).
TRAIN_EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ "${FOREACH:-1}" = "1" ]; then TRAIN_EXTRA_ARGS="--foreach_updates $TRAIN_EXTRA_ARGS"; fi
if [ "${FUSED_ADAM:-0}" = "1" ]; then TRAIN_EXTRA_ARGS="--npu_fused_adamw $TRAIN_EXTRA_ARGS"; fi

# swanlab cloud if a netrc was injected at submit time and the zone can reach
# swanlab.cn (verified reachable 2026-09-22 via ascend-netprobe).
if [ -n "${SWANLAB_NETRC_B64:-}" ]; then
  mkdir -p "$HOME/.swanlab"
  echo "$SWANLAB_NETRC_B64" | base64 -d > "$HOME/.swanlab/.netrc"
  chmod 600 "$HOME/.swanlab/.netrc"
fi
if [ -f "$HOME/.swanlab/.netrc" ] && curl -s --max-time 10 -o /dev/null https://swanlab.cn; then
  echo SWANLAB_CLOUD
else
  export SWANLAB_MODE=disabled
  echo SWANLAB_DISABLED
fi

RUN=${RUN_NAME:-hero-$STAGE}
RUN_DIR=$W/runs/$RUN
mkdir -p "$RUN_DIR"
exec 9>>"$RUN_DIR/.writer.lock"
if ! flock -n 9; then
  echo "another hero launcher holds the writer lock for $RUN_DIR; not starting"
  exit 1
fi

latest_ckpt () {
  ls -d "$1"/checkpoint_step_[0-9]* 2>/dev/null | sort -V | tail -1
}

RESUME_ARGS=()
OWN=$(latest_ckpt "$RUN_DIR")
if [ -n "$OWN" ]; then
  python3 -m src.pretrain.stage_control "$OWN" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --min-step "$START" --world-size 16
  RESUME_ARGS=(--resume "$OWN" --resume_full)
elif [ -n "$PREV" ]; then
  PREV_CKPT=$(latest_ckpt "$W/runs/hero-$PREV")
  if [ -z "$PREV_CKPT" ]; then
    echo "no checkpoint from hero-$PREV to bootstrap from" >&2
    exit 1
  fi
  python3 -m src.pretrain.stage_control "$PREV_CKPT" --max-steps "$TOTAL_STEPS" \
    --stop-at-step "$END" --expected-step "$START" --world-size 16
  RESUME_ARGS=(--resume "$PREV_CKPT" --resume_full --reset_sampler)
fi

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
ema_decay_warmup = true
checkpoint_interval = ${CKPT_INTERVAL:-2000}
eval_interval = ${EVAL_INTERVAL:-2500}
[optim]
lr_warmup_steps = ${LR_WARMUP:-20000}
[data]
bucket_plan = "$W/repo-ascend/bucket_plans/hero/${PLAN_DIR:-ascend-0922}/hero-$STAGE-k20.json"
[eval]
loss_interval = ${LOSS_INTERVAL:-500}
[telemetry]
cache_clear_interval = ${CACHE_CLEAR:-0}
EOF

LOG=$W/logs/$RUN.log
# Stale torch shm segments: after a kill -9 (OOM watchdog) the dead attempt's
# /dev/shm/torch_* files leak, and the next attempt on the same node dies
# within minutes on ENOSPC (observed in smoke6 attempts 2+). Clean them at
# launcher start; no live training process exists at this point.
rm -f /dev/shm/torch_* 2>/dev/null || true
df -h /dev/shm | tee -a "$LOG"
echo "LAUNCH $RUN accum=$ACCUM start=$START end=$END ${RESUME_ARGS[*]:-fresh}" | tee -a "$LOG"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: staging + config render OK, skipping torchrun" | tee -a "$LOG"
  cat "$INPUT_CONFIG" "$OVERRIDE"
  exit 0
fi

# /dev/shm is 64MB on these pods: DataLoader worker->main transfers must use
# memfd (file_descriptor) instead of shm files, and the fd strategy wants a
# higher open-file limit. Root cause of the ~4-min crash coin-flip seen in
# hostprobe5 attempts 1-2 and hero2 attempt 1 (torch_* ENOSPC -> bus error).
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Byte cursor into $LOG before training starts: on crash, only the region
# appended after this point is mined for the first real traceback (the raw
# tail is useless — 15 surviving ranks drown it in tbe teardown noise).
CRASH_CURSOR=$(stat -c%s "$LOG" 2>/dev/null || echo 0)

setsid python3 -m torch.distributed.run --nproc_per_node=16 --master_port=29511 \
  --log-dir "$W/logs/elastic" \
  -m src.pretrain.train \
  --config configs/base.toml \
  --config "$INPUT_CONFIG" \
  --config configs/hero.toml \
  --config "$OVERRIDE" \
  --dataloader_sharing_strategy file_descriptor \
  $COMPILE_ARGS \
  $TRAIN_EXTRA_ARGS \
  --run_name "$RUN" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  >> "$LOG" 2>&1 &
TPID=$!

# Watchdog: kill the whole process group at the first OOM signature so a
# partial-rank death cannot wedge the job (headroom54g hung this way). Only
# bytes appended AFTER this launcher started are scanned — the log persists
# across restarts, and a stale OOM line from a previous attempt must not
# kill the fresh run (ascend-hero-smoke hit exactly that kill loop).
LOG_CURSOR=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
DONE_SINCE=0
while kill -0 $TPID 2>/dev/null; do
  sleep 60
  NEW_SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo "$LOG_CURSOR")
  if [ "$NEW_SIZE" -gt "$LOG_CURSOR" ]; then
    if tail -c +$((LOG_CURSOR + 1)) "$LOG" | grep -q "NPU out of memory"; then
      echo "WATCHDOG_OOM_KILL" | tee -a "$LOG"
      kill -9 -$TPID 2>/dev/null || true
      wait $TPID 2>/dev/null || true
      rm -f /dev/shm/torch_* 2>/dev/null || true
      exit 1
    fi
    # Shutdown hang guard (ascend-hostprobe5): torchrun wedges in the tbe
    # multiprocess cleanup AFTER training prints its stage endpoint, so the
    # job would sit on 16 idle cards until the platform times out. Once the
    # endpoint marker is seen, give the process 3 minutes to exit cleanly,
    # then kill the group and report success — checkpoints are already
    # flushed (the marker follows the final save).
    if [ $DONE_SINCE -eq 0 ] && tail -c +$((LOG_CURSOR + 1)) "$LOG" | grep -q "reached stage endpoint"; then
      DONE_SINCE=$(date +%s)
      echo "WATCHDOG_ENDPOINT_SEEN" | tee -a "$LOG"
    fi
    LOG_CURSOR=$NEW_SIZE
  fi
  if [ $DONE_SINCE -ne 0 ] && [ $(( $(date +%s) - DONE_SINCE )) -ge 180 ]; then
    echo "WATCHDOG_ENDPOINT_KILL (clean-exit grace expired)" | tee -a "$LOG"
    kill -9 -$TPID 2>/dev/null || true
    wait $TPID 2>/dev/null || true
    exit 0
  fi
done
wait $TPID && RC=0 || RC=$?
echo "TRAIN_EXIT $RC" | tee -a "$LOG"
# On failure, echo the tail of the training log into job stdout — the
# platform log is the only thing reachable without a dedicated logtail job,
# and a crash loop otherwise gives no traceback (ascend-hostprobe3).
if [ "$RC" != "0" ]; then
  # Mine the region appended during THIS attempt for the first real Python
  # traceback — chronologically almost always the crashing rank, printed
  # before the teardown noise that fills the raw tail.
  NEWLOG=$(mktemp)
  tail -c +$((CRASH_CURSOR + 1)) "$LOG" | tr '\r' '\n' > "$NEWLOG" 2>/dev/null || true
  FIRST_TB=$(grep -n "Traceback (most recent call last)" "$NEWLOG" | head -1 | cut -d: -f1)
  if [ -n "$FIRST_TB" ]; then
    echo "----- TRAIN_FIRST_TRACEBACK begin -----"
    sed -n "${FIRST_TB},$((FIRST_TB + 60))p" "$NEWLOG"
    echo "----- TRAIN_FIRST_TRACEBACK end -----"
  fi
  echo "----- TRAIN_FATAL_LINES begin -----"
  grep -n -i "bus error\|no space left\|RuntimeError\|ValueError\|out of memory\|AssertionError\|KeyError\|FileNotFoundError" "$NEWLOG" | head -15
  echo "----- TRAIN_FATAL_LINES end -----"
  rm -f "$NEWLOG"
  # torchrun writes the failing rank's full traceback as JSON under --log-dir;
  # dump the freshest error files (path: <log-dir>/<run_id>/<attempt>/<rank>/error.json).
  if [ -d "$W/logs/elastic" ]; then
    echo "----- ELASTIC_ERROR_FILES begin -----"
    find "$W/logs/elastic" -name error.json -mmin -30 2>/dev/null | head -4 | while read -r f; do
      echo "--- $f"
      head -c 5000 "$f"; echo
    done
    echo "----- ELASTIC_ERROR_FILES end -----"
  fi
  echo "----- TRAIN_LOG_TAIL begin -----"
  tail -c 4000 "$LOG" | tr '\r' '\n' | tail -40
  echo "----- TRAIN_LOG_TAIL end -----"
fi
exit $RC
