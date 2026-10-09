#!/bin/bash
# Download Qwen-Image-2.1 weights via the HF mirror (huggingface.co is
# unreachable from this host). ~35 GB.
set -e
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv/bin/hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen-Image-2.1
