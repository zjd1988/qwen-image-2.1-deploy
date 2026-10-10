#!/usr/bin/env python3
"""Rebuild Qwen-Image-2.1-Turbo's single-file text encoder from the base
model's 4 shards, avoiding a 17.5 GB re-download.

Turbo's text_encoder/model.safetensors is byte-for-byte the base Qwen3-VL 8B
weights repacked into one file (verified 2026-10-10 against the first 1.2 GB
of the official download stream). This script needs a small piece of the
official file to copy the exact header (tensor order and offsets):

  bash scripts/download_model.sh turbo   # let it fetch ~100 MB, Ctrl-C
  .venv/bin/python scripts/merge_te_shards.py
  # then verify the result matches the partial download byte-for-byte:
  #   cmp -n $(stat -c%s text_encoder/model.safetensors.incomplete) \
  #       text_encoder/model.safetensors text_encoder/model.safetensors.incomplete

Streams tensor bytes shard-by-shard; RAM usage stays under ~100 MB.
"""
import json
import struct
import sys
from pathlib import Path

BASE = Path("models/Qwen-Image-2.1/text_encoder")
TURBO = Path("models/Qwen-Image-2.1-Turbo/text_encoder")


def read_header(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return json.loads(f.read(n)), 8 + n


def main():
    partial = TURBO / "model.safetensors.incomplete"
    header_src = TURBO / "model.safetensors"
    if partial.exists():
        header_src = partial
    if not header_src.exists():
        sys.exit(f"need a piece of the official TE file for its header: "
                 f"start scripts/download_model.sh turbo, wait for ~100MB of "
                 f"text_encoder/model.safetensors.incomplete, then stop it")
    with open(header_src, "rb") as f:
        remote, _ = read_header(f)

    wmap = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    shard_hdr = {}
    for shard in sorted(set(wmap.values())):
        with open(BASE / shard, "rb") as f:
            shard_hdr[shard] = read_header(f)

    info = {}
    for key, shard in wmap.items():
        hdr, data_start = shard_hdr[shard]
        b0, b1 = hdr[key]["data_offsets"]
        info[key] = (shard, hdr[key]["dtype"], hdr[key]["shape"],
                     data_start + b0, b1 - b0)

    for key, meta in remote.items():
        if key == "__metadata__":
            continue
        shard, dt, shape, off, ln = info[key]
        rb0, rb1 = meta["data_offsets"]
        assert meta["dtype"] == dt and meta["shape"] == shape and rb1 - rb0 == ln, \
            f"tensor metadata mismatch: {key}"

    order = sorted((k for k in remote if k != "__metadata__"),
                   key=lambda k: remote[k]["data_offsets"][0])
    total = sum(info[k][4] for k in order)
    print(f"merging {len(order)} tensors, {total / 1e9:.2f} GB")

    out = TURBO / "model.safetensors.merged"
    written = 0
    with open(out, "wb") as fo:
        # header: 8-byte length + JSON, copied verbatim from the official file
        with open(header_src, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            fo.write(struct.pack("<Q", n))
            fo.write(f.read(n))
        cur_shard = None
        sf = None
        for i, key in enumerate(order):
            shard, dt, shape, off, ln = info[key]
            if shard != cur_shard:
                if sf:
                    sf.close()
                sf = open(BASE / shard, "rb")
                cur_shard = shard
            sf.seek(off)
            remain = ln
            while remain:
                chunk = sf.read(min(remain, 64 * 1024 * 1024))
                fo.write(chunk)
                remain -= len(chunk)
            written += ln
            if i % 100 == 0:
                print(f"  {i}/{len(order)}  {written / 1e9:.2f} GB", flush=True)
        if sf:
            sf.close()
    print(f"[saved] {out} ({written / 1e9:.2f} GB)")
    print("next: byte-verify against the partial official download (see docstring),")
    print(f"then: mv {out} {TURBO / 'model.safetensors'}")


if __name__ == "__main__":
    main()
