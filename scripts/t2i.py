#!/usr/bin/env python3
"""Qwen-Image-2.1 text-to-image inference.

Defaults target a single 24 GB GPU (RTX 3090/4090): bf16 weights with
model CPU offload. Use --no-offload when the GPU can hold the whole model
(~35 GB in bf16).
"""
import argparse
import time

import torch
from diffusers import QwenImage21Pipeline

ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="outputs/t2i.png")
    p.add_argument("--model", default="models/Qwen-Image-2.1",
                   help="local model dir or HF model id")
    p.add_argument("--ratio", choices=sorted(ASPECT_RATIOS), default=None,
                   help="use a native 2K aspect ratio; overrides --width/--height")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-offload", action="store_true")
    args = p.parse_args()

    width, height = ASPECT_RATIOS.get(args.ratio, (args.width, args.height)) \
        if args.ratio else (args.width, args.height)

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.no_offload:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    image = pipe(
        prompt=args.prompt,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    print(f"[gen] {width}x{height}, {args.steps} steps, {time.time() - t0:.1f}s",
          flush=True)

    image.save(args.output)
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
