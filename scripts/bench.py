#!/usr/bin/env python3
"""Measure a running server: fitted context, time to first token and prefill speed at several prompt lengths,
generation speed, and needle recall. Writes a JSON file that scripts/compare.py puts side by side.

    scripts/bench.py --out results/base.json
    scripts/bench.py --lengths 1000,32000,128000 --out results/exp.json

Standard library only. Each prompt is unique (random filler), so the server's prompt cache cannot help.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test import request  # noqa: E402

WORDS = ["the", "river", "stone", "quiet", "market", "lantern", "north", "garden", "copper", "window",
         "evening", "harbor", "paper", "silver", "morning", "bridge", "orchard", "letter", "candle", "road"]


def filler(rng, n_tokens):
    # ~1.3 tokens per word with this vocabulary; the server reports the real count
    return " ".join(rng.choice(WORDS) for _ in range(int(n_tokens / 1.3)))


def complete(args, prompt, n_predict, **extra):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n_predict,
            "temperature": 0, "cache_prompt": False, **extra}
    started = time.monotonic()
    out = request(f"{args.url}/v1/chat/completions", body, args.api_key)
    wall = time.monotonic() - started
    t = out.get("timings", {})
    content = out["choices"][0]["message"].get("content") or ""
    return t, wall, content


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key")
    p.add_argument("--lengths", default="1000,8000,32000,64000,128000",
                   help="prompt lengths in tokens; lengths beyond 95%% of n_ctx are skipped")
    p.add_argument("--max-ctx-fraction", type=float, default=0.9,
                   help="also measure one prompt at this fraction of n_ctx (0 to skip)")
    p.add_argument("--gen", type=int, default=256, help="tokens to generate for the decode measurement")
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    props = request(f"{args.url}/props", key=args.api_key)
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or 0
    result = {"label": args.label or args.out.stem, "n_ctx": n_ctx, "model": props.get("model_path"),
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "prefill": [], "decode": {}, "needle": None}
    print(f"n_ctx {n_ctx}")

    lengths = [int(x) for x in args.lengths.split(",") if x]
    if args.max_ctx_fraction > 0 and n_ctx:
        lengths.append(int(n_ctx * args.max_ctx_fraction))
    rng = random.Random(1)

    for n in sorted(set(lengths)):
        if n_ctx and n > n_ctx * 0.95:
            print(f"skip {n}: does not fit in n_ctx {n_ctx}")
            continue
        t, wall, _ = complete(args, filler(rng, n) + "\nSay OK.", 1)
        row = {"tokens": t.get("prompt_n"), "ttft_s": round(t.get("prompt_ms", 0) / 1000, 3),
               "prefill_tok_s": round(t.get("prompt_per_second", 0), 1), "wall_s": round(wall, 2)}
        result["prefill"].append(row)
        print(f"prefill {row['tokens']:>8} tok: TTFT {row['ttft_s']:>8.2f} s, {row['prefill_tok_s']:>8.1f} tok/s")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))  # keep partial results if a long run fails

    # decode: code (DSpark drafts well) and prose (it drafts poorly)
    for kind, prompt in [("code", "Write a Python function that parses an ISO 8601 date without libraries, "
                                  "with docstring and tests."),
                         ("prose", "Write a short story about a lighthouse keeper who finds a letter.")]:
        t, _, _ = complete(args, prompt, args.gen)
        result["decode"][kind] = round(t.get("predicted_per_second", 0), 1)
        print(f"decode {kind}: {result['decode'][kind]} tok/s")

    # recall near the end of the context
    if n_ctx:
        code = str(rng.randrange(10**7, 10**8))
        n = int(n_ctx * 0.8)
        words = filler(rng, n).split(" ")
        words.insert(len(words) // 3, f". The secret access code is {code}. ")
        t, _, content = complete(args, " ".join(words) + "\n\nWhat is the secret access code? Reply with the number only.", 16)
        result["needle"] = {"tokens": t.get("prompt_n"), "found": code in content}
        print(f"needle at {t.get('prompt_n')} tokens: {'found' if code in content else 'MISSED'}")

    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
