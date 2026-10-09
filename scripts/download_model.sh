#!/bin/bash
# Download Qwen-Image-2.1 weights via the HF mirror (huggingface.co is
# unreachable from this host). ~35 GB.
set -e
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -x .venv/bin/hf ]; then
    .venv/bin/hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen-Image-2.1
else
    # fresh machine: no .venv yet, run hf via uv without creating an env
    UV=${UV:-$(command -v uv || echo /data/zhaojd-a/github_codes/uv)}
    $UV tool run --from 'huggingface-hub[hf-transfer]' \
        hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen-Image-2.1
fi
