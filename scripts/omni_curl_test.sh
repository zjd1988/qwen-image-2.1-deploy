#!/bin/bash
# Smoke-test the vLLM-Omni server: one T2I request, decode base64 to PNG.
# NOTE: always send num_inference_steps (server default is 50, not 40) and
# true_cfg_scale 1.0 (default 4.0 would double DiT compute).
# On 24 GB GPUs the text encoder (17.5 GB) stays resident, so 1024px OOMs;
# 512x512 is the verified size for this hardware (see README).
set -e
cd "$(dirname "$0")/.."
curl -sf -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen-Image-2.1",
       "prompt": "A ceramic teapot on a wooden table",
       "size": "512x512",
       "num_inference_steps": 40,
       "true_cfg_scale": 1.0,
       "seed": 42}' \
  | .venv-omni/bin/python -c '
import base64, json, sys
resp = json.load(sys.stdin)
img = base64.b64decode(resp["data"][0]["b64_json"])
open("outputs/verify_omni_t2i.png", "wb").write(img)
print(f"[saved] outputs/verify_omni_t2i.png ({len(img)} bytes)")
'
