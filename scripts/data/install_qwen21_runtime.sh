#!/bin/bash
# Install the Qwen-Image 2.1 runtime from the staged wheelhouse.
#
# Everything in the wheelhouse is installed except torch, triton and the
# nvidia/cuda wheels: the base image already carries a torch built for its own
# driver, and those wheels exist only because pip resolves accelerator
# dependencies against the CPU-side platform.  Dependencies are installed
# exactly as staged (--no-deps), which is why the wheelhouse must be complete.
set -eu

W=${ARTFLOW_ROOT:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow}
WHEELS=${WHEELS:-$W/wheels-qwen21}
PY=${PY:-python}

# numpy is excluded for the same reason as torch: the image's scipy and sklearn
# are compiled against the numpy it ships with, and replacing numpy under them
# turns every later `import scipy` into an ABI failure.  transformers and
# diffusers both accept the numpy already there.
names=$(ls "$WHEELS"/*.whl | xargs -n1 basename \
  | grep -vE '^(torch-|triton-|numpy-|nvidia_|cuda_|setuptools-)' \
  | sed -E 's/-[0-9][^-]*(-[^-]*)*\.whl$//' | sort -u | tr '\n' ' ')
echo "[runtime] installing $(echo "$names" | wc -w) wheels"
# --upgrade matters: the base image already carries older diffusers and
# transformers, and without it pip reports "already satisfied" and installs
# nothing, leaving a runtime that cannot load the model.
$PY -m pip install -q --upgrade --no-index --no-deps --find-links "$WHEELS" $names 2>&1 | tail -5
$PY - <<'EOF'
import numpy, torch, scipy.sparse  # scipy after numpy: the ABI check that matters
import diffusers, transformers, huggingface_hub
print("numpy", numpy.__version__, "| torch", torch.__version__,
      "| diffusers", diffusers.__version__, "| QwenImage21Pipeline",
      hasattr(diffusers, "QwenImage21Pipeline"))
print("transformers", transformers.__version__, "| hub", huggingface_hub.__version__)
if not hasattr(diffusers, "QwenImage21Pipeline"):
    raise SystemExit("diffusers without QwenImage21Pipeline")
EOF
