#!/bin/bash
# Set up vLLM-Omni with Qwen-Image-2.1 support.
# Support is not yet merged upstream; it lives on PR #7759 of vllm-project/vllm-omni
# (see https://recipes.vllm.ai/Qwen/Qwen-Image-2.1). Installs into .venv-omni.
set -e
cd "$(dirname "$0")/.."
# uv resolution: $UV env var > uv on PATH > local fallback
if [ -z "$UV" ]; then
    UV=$(command -v uv || echo /data/zhaojd-a/github_codes/uv)
fi

if [ ! -d vllm-omni/.git ]; then
    git clone https://github.com/vllm-project/vllm-omni.git
fi
cd vllm-omni
git fetch origin pull/7759/head:qwen-image-2.1
git checkout qwen-image-2.1
cd ..

$UV venv --python 3.12 .venv-omni
VIRTUAL_ENV=.venv-omni $UV pip install --python .venv-omni/bin/python vllm==0.29.0 --torch-backend=auto
VIRTUAL_ENV=.venv-omni $UV pip install --python .venv-omni/bin/python -e vllm-omni
