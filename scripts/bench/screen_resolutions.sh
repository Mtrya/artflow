#!/bin/bash
# Screen micro-batch sizes for a subset of the resolutions in the 256p bucket
# plan (bucket_plans/256p-k10-13src.json).
#
# Two rounds per resolution, because the cost of isolating a bucket is set by
# how rare that bucket is, and the candidate sizes that pay off differ by
# bucket.  Measured draws per emitted micro-batch (micro-batch 16, res1):
#
#     bucket bound   26    47    90   118   165   223   376   621   952  2048
#     draws/batch   610   386   578   814  1166  1968  9252 20577 40998 414114
#
# A run at micro-batch 64 draws four times as much per micro-batch, so the
# rare buckets make a sweep over large candidates cost hours of sampling for a
# size they will not use anyway - the long buckets are already flat between
# micro-batch 4 and 16.
#
#   round a: candidates 64 and 128, and only the buckets holding at least 5% of
#            a resolution's draw mass.  These are the cheap, common buckets,
#            where micro-batch 16 is still twice as fast as micro-batch 4.
#   round b: candidates 16 and 32, over every bucket that holds at least 0.1%
#            of the draw mass - which stops at the 952-token bucket.  The
#            2048-token bucket holds 391 captions in the whole corpus and falls
#            through to the declared fallback.
#
# Usage: bash scripts/bench/screen_resolutions.sh "<resolution ids>" <workers>
set -u
RESOLUTIONS=${1:?resolution ids required, e.g. "1 2 3"}
WORKERS=${2:?worker count required}

W=${ARTFLOW_ROOT:?Set ARTFLOW_ROOT to the shared-workspace root used on the cluster}
cd $W/repo || exit 1
D=$W/precomputed_dataset
MIX="$D/d1@256p:0.1208 $D/d2-wikiart@256p:0.1538 $D/d2-museum@256p:0.0073 $D/d3-human@256p:0.0612 $D/d3-people@256p:0.0596 $D/d3-pexels@256p:0.110 $D/d3-synth@256p:0.085 $D/d4-vintage@256p:0.0757 $D/d4-zimage@256p:0.0177 $D/d4-megalith@256p:0.0024 $D/d4-inat@256p:0.0008 $D/d4-pd12m@256p:0.0636 $D/d4-relaion@256p:0.2423"
ROOT=$W/bucket_plans/screen-full
mkdir -p "$ROOT/a" "$ROOT/b" "$W/screen_runs/full"
date -Is
nvidia-smi --query-gpu=index,name,memory.total --format=csv

screen() {
  local round=$1 res=$2 card=$3 candidates=$4 min_share=$5
  local dir=$ROOT/$round
  CUDA_VISIBLE_DEVICES=$card python3 -m scripts.bench.batch_size_screen \
    --plan "$W/bucket_plans/256p-k10-13src.json" \
    --mix "$MIX" \
    --resolutions "$res" \
    --batch-sizes $candidates \
    --steps 30 --warmup 10 --repeats 1 \
    --vae "$W/models/e2e-qwenimage-vae" \
    --text-encoder "$W/models/Qwen3-0.6B" \
    --base-config configs/base.toml \
    --min-caption-share "$min_share" \
    --timeout 1800 \
    --fallback-batch-size 16 \
    --out "$dir/plan-res$res.json" \
    --report "$dir/report-res$res.md" \
    --measurements "$dir/measurements-res$res.json" \
    --cache "$dir/cache-res$res.json" \
    --run-dir "$W/screen_runs/full/$round/res$res" \
    >> "$dir/log-res$res.txt" 2>&1
  echo "  round $round res$res rc=$? $(date -Is)" >> "$dir/log-res$res.txt"
}

index=0
for r in $RESOLUTIONS; do
  if [ "$index" -ge "$WORKERS" ]; then
    echo "more resolutions than workers; $r not started"
    break
  fi
  (
    screen a "$r" "$index" "64 128" 0.05
    screen b "$r" "$index" "16 32" 0.001
  ) &
  index=$((index + 1))
done
wait
echo "SCREEN DONE $(date -Is)"
for r in $RESOLUTIONS; do
  echo "--- res$r round a ---"
  tail -2 "$ROOT/a/log-res$r.txt" 2>/dev/null
  echo "--- res$r round b ---"
  tail -2 "$ROOT/b/log-res$r.txt" 2>/dev/null
done
