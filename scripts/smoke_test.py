#!/usr/bin/env python3
"""Check a running Inkling 2 Mini server end to end and print its speed.

    scripts/smoke_test.py                       text, per-request thinking, vision
    scripts/smoke_test.py --needle 200000       also a needle-in-a-haystack run at ~200K tokens

Standard library only, so it runs on any host with Python 3.9+.
"""
import argparse
import base64
import json
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


def request(url, body=None, key=None, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        raise SystemExit(f"{url}: HTTP {err.code}: {err.read().decode(errors='replace')[:500]}")


def chat(args, content, max_tokens=256, **extra):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, **extra}
    started = time.monotonic()
    out = request(f"{args.url}/v1/chat/completions", body, args.api_key)
    elapsed = time.monotonic() - started
    msg = out["choices"][0]["message"]
    return msg.get("content") or "", msg.get("reasoning_content") or "", out.get("timings", {}), elapsed


def speed(t):
    return (f"prompt {t.get('prompt_n', '?')} tok in {t.get('prompt_ms', 0) / 1000:.2f}s "
            f"({t.get('prompt_per_second', 0):.0f} tok/s), "
            f"gen {t.get('predicted_n', '?')} tok at {t.get('predicted_per_second', 0):.1f} tok/s")


def png(width, height, pixel):
    """Encode an RGB image as PNG; pixel(x, y) returns an (r, g, b) tuple."""
    rows = b"".join(b"\0" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


failures = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def test_text(args):
    content, reasoning, t, _ = chat(args, "Reply with exactly one word: the capital of France.")
    check("text, default effort", "paris" in content.lower(), f"{content.strip()[:80]!r}; {speed(t)}")
    # effort 0 can still open a short thinking block, so only report its length
    print(f"       reasoning by default: {len(reasoning)} chars (expect none or very little)")


def test_thinking(args):
    content, reasoning, t, _ = chat(args, "What is 17 * 23? Answer with the number.", max_tokens=2048,
                                    reasoning_effort="high")
    check("thinking per request", bool(reasoning) and "391" in content,
          f"{len(reasoning)} reasoning chars, answer {content.strip()[:40]!r}; {speed(t)}")


def test_vision(args):
    # left half red, right half blue
    image = png(256, 128, lambda x, y: (220, 30, 30) if x < 128 else (30, 30, 220))
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    content, _, t, _ = chat(args, [
        {"type": "text", "text": "This image has two colored halves. Name the color of the left half and of the right half."},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    low = content.lower()
    check("vision", "red" in low and "blue" in low, f"{content.strip()[:100]!r}; {speed(t)}")


def test_needle(args):
    rng = random.Random(0)
    code = f"{rng.randrange(10**7, 10**8)}"
    words = ["the", "river", "stone", "quiet", "market", "lantern", "north", "garden", "copper", "window",
             "evening", "harbor", "paper", "silver", "morning", "bridge", "orchard", "letter", "candle", "road"]
    # ~1.3 tokens per filler word with this vocabulary; the server reports the real count
    n_words = int(args.needle / 1.3)
    filler = [rng.choice(words) for _ in range(n_words)]
    filler.insert(int(n_words * args.depth), f". The secret access code is {code}. ")
    prompt = " ".join(filler) + "\n\nWhat is the secret access code mentioned in the text above? Reply with the number only."
    content, _, t, elapsed = chat(args, prompt, max_tokens=32)
    check(f"needle at {t.get('prompt_n', '?')} tokens, depth {args.depth:.0%}", code in content,
          f"got {content.strip()[:40]!r}, want {code}; {speed(t)}; wall {elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key")
    p.add_argument("--needle", type=int, metavar="TOKENS", help="also run a needle test at about this many tokens")
    p.add_argument("--depth", type=float, default=0.5, help="needle position, 0 = start, 1 = end (default 0.5)")
    args = p.parse_args()

    props = request(f"{args.url}/props", key=args.api_key)
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    vision = props.get("modalities", {}).get("vision")
    print(f"server: {props.get('model_alias') or props.get('model_path')}, n_ctx {n_ctx}, vision {vision}")

    test_text(args)
    test_thinking(args)
    if vision:
        test_vision(args)
    else:
        print("[SKIP] vision: the server has no vision projector (VISION=off)")
    if args.needle:
        if n_ctx and args.needle > n_ctx * 0.95:
            raise SystemExit(f"--needle {args.needle} does not fit in n_ctx {n_ctx}")
        test_needle(args)

    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
