#!/usr/bin/env python3
"""Benchmark cache-dit DBCache acceleration vs vanilla Diffusers.

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_cache_dit.py [--turbo]
"""
import argparse
import time

import torch
from diffusers import QwenImage21Pipeline

PROMPT = "一家咖啡馆门口的霓虹灯招牌，写着'缓存加速测试'，夜晚街道，湿润路面反射灯光，写实摄影"


def run(pipe, label, output, seed=42, **extra):
    t0 = time.time()
    image = pipe(
        prompt=PROMPT,
        width=1024,
        height=1024,
        generator=torch.Generator("cuda").manual_seed(seed),
        **extra,
    ).images[0]
    dt = time.time() - t0
    image.save(output)
    print(f"[{label}] {dt:.1f}s -> {output}", flush=True)
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen-Image-2.1")
    p.add_argument("--turbo", action="store_true",
                   help="use Turbo checkpoint (fixed 8-step schedule)")
    args = p.parse_args()

    model = "models/Qwen-Image-2.1-Turbo" if args.turbo else args.model
    tag = "turbo" if args.turbo else "base"
    extra = {"use_kv_cache": True} if args.turbo else {}
    if not args.turbo:
        extra["num_inference_steps"] = 40

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    run(pipe, f"{tag}/vanilla", f"outputs/bench_{tag}_vanilla.png", **extra)

    import cache_dit
    cache_dit.enable_cache(pipe)  # default: DBCache
    run(pipe, f"{tag}/dbcache-warmup", f"outputs/bench_{tag}_warmup.png", **extra)
    run(pipe, f"{tag}/dbcache", f"outputs/bench_{tag}_dbcache.png", **extra)


if __name__ == "__main__":
    main()
