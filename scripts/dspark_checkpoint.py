#!/usr/bin/env python3
"""Assemble the minimal local checkpoint that llama.cpp's DSpark converter needs, without the 19.5 GB weights file.

The converter (convert_hf_to_gguf.py --dspark) reads the draft from mtp.safetensors and two tensors of the target
model: the unembedding (to fold the Markov head into markov_w2) and embed_norm. Those two are fetched with HTTP range
requests from the target's safetensors file, using the offsets in its JSON header, and written as a small
safetensors file with a matching index.

    dspark_checkpoint.py OUT_DIR [--repo thinkingmachines/Inkling-2-Mini] [--revision main]
"""
import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "chat_template.jinja", "processor_config.json", "tiktoken/tokenizer.model",
         "mtp.safetensors"]
TARGET_FILE = "model-00001-of-00001.safetensors"
TARGET_TENSORS = ["model.llm.unembed.weight", "model.llm.embed_norm.weight"]


def get(url, start=None, end=None, retries=5):
    headers = {"User-Agent": "inkling-dspark-convert"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                data = resp.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"short range read: {len(data)} of {end - start + 1} bytes")
            return data
        except Exception as err:  # network errors are retried, the last one is raised
            if attempt == retries - 1:
                raise
            print(f"retry {url}: {err}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "inkling-dspark-convert"}), timeout=120) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--repo", default="thinkingmachines/Inkling-2-Mini")
    p.add_argument("--revision", default="main")
    args = p.parse_args()
    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}"

    for name in FILES:
        print(f"fetch {name}")
        download(f"{base}/{name}", args.out / name)

    target = args.out / TARGET_FILE
    if not target.exists():
        url = f"{base}/{TARGET_FILE}"
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        header = json.loads(get(url, 8, 8 + n - 1))
        data_start = 8 + n
        out_header, blobs, offset = {}, [], 0
        for name in TARGET_TENSORS:
            info = header[name]
            begin, end = info["data_offsets"]
            print(f"fetch {name} {info['dtype']} {info['shape']} ({(end - begin) / 2**20:.0f} MiB)")
            blob = get(url, data_start + begin, data_start + end - 1)
            out_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + len(blob)]}
            blobs.append(blob)
            offset += len(blob)
        meta = json.dumps(out_header).encode()
        meta += b" " * (-len(meta) % 8)  # safetensors pads the header to 8 bytes
        tmp = target.with_suffix(".part")
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(meta)))
            f.write(meta)
            for blob in blobs:
                f.write(blob)
        tmp.rename(target)

    index = {"metadata": {}, "weight_map": {name: TARGET_FILE for name in TARGET_TENSORS}}
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"checkpoint ready in {args.out}")


if __name__ == "__main__":
    main()
