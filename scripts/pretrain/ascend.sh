#!/usr/bin/env bash
# Platform setup uses the same explicit storage root as the launcher.
set -euo pipefail
if [[ "${1:-}" != --storage-root || -z "${2:-}" ]]; then
    echo "Usage: $0 --storage-root PATH --config CONFIG --stage STAGE --nproc_per_node N" >&2
    exit 2
fi
storage_dir=$(realpath -- "$2")
shift 2
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$storage_dir/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$storage_dir/cache/hf"
export TORCH_HOME="$storage_dir/models/torch_home"
export SWANLAB_LOG_DIR="$storage_dir/runs/swanlog/pretrain"
export SWANLAB_MODE=online
exec python -m scripts.pretrain.launch --storage-root "$storage_dir" "$@"
