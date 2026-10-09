#!/usr/bin/env python3
"""Qwen-Image-2.1 image editing (1..N reference images)."""
import argparse
import time

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", action="append", required=True,
                   help="input image path; repeat for multi-reference editing (up to 10)")
    p.add_argument("--output", default="outputs/edit.png")
    p.add_argument("--model", default="models/Qwen-Image-2.1")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-offload", action="store_true")
    args = p.parse_args()

    images = [Image.open(path) for path in args.image]

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.no_offload:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    kwargs = {}
    if args.width and args.height:
        kwargs.update(width=args.width, height=args.height)

    t0 = time.time()
    image = pipe(
        prompt=args.prompt,
        image=images if len(images) > 1 else images[0],
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        **kwargs,
    ).images[0]
    print(f"[gen] {args.steps} steps, {time.time() - t0:.1f}s", flush=True)

    image.save(args.output)
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
