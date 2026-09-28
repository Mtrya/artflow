#!/bin/bash
# Stage everything the Qwen-Image-2.1 synthetic run needs, on the CPU workspace.
#
# Three artefacts land on the shared disk:
#   - a wheelhouse with diffusers main (the only place QwenImage21Pipeline
#     exists today; no PyPI release carries it), transformers 5.17 and their
#     dependencies, so the GPU job can install without outbound network;
#   - the diffusers checkout the wheel was built from, kept for provenance;
#   - the model weights, flattened so from_pretrained reads the directory.
#
# Safe to re-run: the clone is only made once, pip reuses the wheelhouse, and
# the snapshot download skips files already present.
set -u

W=${ARTFLOW_ROOT:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
PY=$W/venv-harvest/bin/python
WHEELS=$W/wheels-qwen21
SRC=$W/src
MODEL=$W/models/Qwen-Image-2.1
DIFFUSERS=$SRC/diffusers

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1

mkdir -p "$WHEELS" "$SRC"
echo "== start $(date -Is)"

# 1. diffusers main -> wheel
if [ ! -d "$DIFFUSERS" ]; then
  git clone --depth 1 https://github.com/huggingface/diffusers "$DIFFUSERS" || exit 1
fi
git -C "$DIFFUSERS" log -1 --format='diffusers %H %cs %s' || exit 1
"$PY" -m pip install -q --upgrade setuptools wheel 2>&1 | tail -2
"$PY" -m pip wheel --no-deps --no-build-isolation -w "$WHEELS" "$DIFFUSERS" 2>&1 | tail -3

# 2. dependency wheelhouse
"$PY" -m pip download -d "$WHEELS" \
  "huggingface-hub>=1.32.0,<2.0" "transformers==5.17.0" "tokenizers>=0.23.1,<0.24" \
  "accelerate>=1.1.0" "safetensors>=0.8.0" "regex>=2025.10.22" \
  typer pyyaml tqdm numpy packaging filelock requests importlib_metadata \
  "Pillow>=10.0.1" ftfy jinja2 2>&1 | tail -3
ls "$WHEELS"

# 3. model weights
"$PY" - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download(
    "Qwen/Qwen-Image-2.1", local_dir="$MODEL", max_workers=8,
    ignore_patterns=["*.webp", "*.png", "*.pdf", "*.jpg"],
)
print("model at", path)
EOF
rc=$?

du -sh "$MODEL" "$WHEELS" 2>/dev/null
echo "== done rc=$rc $(date -Is)"
echo SYNTH-FETCH-DONE
