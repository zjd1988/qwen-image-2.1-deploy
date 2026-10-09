#!/bin/bash
# Reference-image edit via vLLM-Omni /v1/images/edits (multipart/form-data).
# Usage: bash scripts/omni_edit.sh <image> <prompt> [size] [output] [seed]
# NOTE: on 24 GB cards keep output at ~512px and downscale large input photos
# first (VAE encode of a big input can OOM). If you get HTTP 500 CUDA OOM,
# retry — allocator state differs after prior requests (see README).
set -e
cd "$(dirname "$0")/.."

IMAGE=${1:?usage: omni_edit.sh <image> <prompt> [size] [output] [seed]}
PROMPT=${2:?usage: omni_edit.sh <image> <prompt> [size] [output] [seed]}
SIZE=${3:-512x512}
OUTPUT=${4:-outputs/edit_omni.png}
SEED=${5:-42}

curl -sf -X POST http://localhost:8091/v1/images/edits \
  -F "model=models/Qwen-Image-2.1" \
  -F "image=@${IMAGE}" \
  -F "prompt=${PROMPT}" \
  -F "size=${SIZE}" \
  -F "num_inference_steps=40" \
  -F "true_cfg_scale=1.0" \
  -F "seed=${SEED}" \
  | .venv-omni/bin/python -c '
import base64, json, sys
resp = json.load(sys.stdin)
img = base64.b64decode(resp["data"][0]["b64_json"])
open(sys.argv[1], "wb").write(img)
print(f"[saved] {sys.argv[1]} ({len(img)} bytes)")
' "$OUTPUT"
