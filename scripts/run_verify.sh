#!/bin/bash
# Quick end-to-end verification: 1024x1024, 20 steps, CPU offload (24 GB GPU).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=models/Qwen-Image-2.1

$PY scripts/t2i.py --model "$MODEL" \
    --prompt 'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement' \
    --width 1024 --height 1024 --steps 20 --seed 42 \
    --output outputs/verify_t2i.png
