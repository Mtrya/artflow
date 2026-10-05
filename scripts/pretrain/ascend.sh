#!/usr/bin/env bash
# Platform setup only. The launcher reads every recipe value from one TOML.
set -euo pipefail
: "${ARTFLOW_ROOT:?Set ARTFLOW_ROOT to the prepared Ascend storage root}"

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$ARTFLOW_ROOT/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$ARTFLOW_ROOT/cache/hf"
export TORCH_HOME="$ARTFLOW_ROOT/models/torch_home"
export SWANLAB_LOG_DIR="$ARTFLOW_ROOT/runs/swanlog/pretrain"

# This is the designated platform secret file documented in INSPIRE.md.
# Never recover credentials from historical job commands or another container.
if [[ -r "$ARTFLOW_ROOT/cache/swanlab.netrc" ]]; then
    mkdir -p "$HOME/.swanlab"
    chmod 700 "$HOME/.swanlab"
    install -m 600 "$ARTFLOW_ROOT/cache/swanlab.netrc" "$HOME/.swanlab/.netrc"
fi
if [[ -z "${SWANLAB_MODE:-}" ]]; then
    if [[ -n "${SWANLAB_API_KEY:-}" || -r "$HOME/.swanlab/.netrc" ]]; then
        export SWANLAB_MODE=online
    else
        export SWANLAB_MODE=offline
    fi
fi
printf 'SwanLab mode: %s; local metrics and checkpoints remain on shared storage.\n' "$SWANLAB_MODE"
exec python -m scripts.pretrain.launch "$@"
