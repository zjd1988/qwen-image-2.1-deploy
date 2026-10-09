#!/bin/bash
# Serve Qwen-Image-2.1 via vLLM-Omni on 2x RTX 3090 (24 GB each).
# bf16 peak is ~34 GB, over one card. TP=2 shards the DiT (~7 GB/GPU);
# the 17.5 GB text encoder is module-offloaded to CPU (runs once per request).
# CUDA graph decode off: offload hooks force eager anyway.
set -e
cd "$(dirname "$0")/.."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# expandable_segments is REQUIRED for /v1/images/edits: without it the
# allocator fragments and the input-image VAE encode (~400 MB) 500s with OOM
# even when a few hundred MB are free. T2I works either way.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-omni/bin/vllm serve models/Qwen-Image-2.1 --omni --port 8091 \
    --diffusion-offload-config '{"mode": "layer", "components": ["dit"], "pin_memory": false}' \
    --no-enable-cuda-graph-decode \
    --step-execution --max-num-seqs 4
