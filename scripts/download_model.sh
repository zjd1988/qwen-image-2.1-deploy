#!/bin/bash
# Download model weights via the HF mirror (huggingface.co is unreachable
# from this host). ~32 GB each.
# Usage: bash scripts/download_model.sh [base|turbo]   (default: base)
# NOTE: HF_HUB_DISABLE_XET=1 is required for xet-backed repos (e.g. Turbo):
# hf_xet bypasses the mirror and hits cas-server.xethub.hf.co directly,
# which fails with 401 from this network.
set -e
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

case "${1:-base}" in
    base)  REPO=Qwen/Qwen-Image-2.1 ;;
    turbo) REPO=Qwen/Qwen-Image-2.1-Turbo ;;
    *) echo "usage: $0 [base|turbo]" >&2; exit 1 ;;
esac
DIR=models/${REPO#Qwen/}

if [ -x .venv/bin/hf ]; then
    .venv/bin/hf download "$REPO" --local-dir "$DIR" --exclude "assets/*"
else
    # fresh machine: no .venv yet, run hf via uv without creating an env
    UV=${UV:-$(command -v uv || echo /data/zhaojd-a/github_codes/uv)}
    $UV tool run --from 'huggingface-hub' \
        hf download "$REPO" --local-dir "$DIR" --exclude "assets/*"
fi
