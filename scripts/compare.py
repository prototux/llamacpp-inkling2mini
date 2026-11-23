#!/usr/bin/env python3
"""Put bench.py results side by side: scripts/compare.py results/base.json results/exp.json"""
import json
import sys


def main():
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    if len(runs) < 2:
        raise SystemExit(__doc__)
    names = [r["label"] for r in runs]
    width = max(12, *(len(n) for n in names))

    def row(title, values, ratio=None):
        cells = "".join(f"{v:>{width}}" for v in values)
        print(f"{title:<28}{cells}{'' if ratio is None else f'{ratio:>{width}}'}")

    row("", names, f"{names[-1]}/{names[0]}")
    row("n_ctx", [r["n_ctx"] for r in runs], f"{runs[-1]['n_ctx'] / max(runs[0]['n_ctx'], 1):.2f}x")

    # prefill rows matched by prompt length bucket (within 10%)
    base = runs[0]["prefill"]
    for b in base:
        vals, last = [], None
        for r in runs:
            m = next((x for x in r["prefill"] if x["tokens"] and abs(x["tokens"] - b["tokens"]) <= 0.1 * b["tokens"]), None)
            vals.append(f"{m['ttft_s']:.2f} s" if m else "-")
            last = m
        speed = f"{b['ttft_s'] / last['ttft_s']:.2f}x faster" if last and last["ttft_s"] else ""
        row(f"TTFT @ {b['tokens']} tok", vals, speed)
    # prompts only the later runs could fit
    for r_i, r in enumerate(runs[1:], 1):
        for x in r["prefill"]:
            if x["tokens"] and not any(abs(x["tokens"] - b["tokens"]) <= 0.1 * b["tokens"] for b in base):
                row(f"TTFT @ {x['tokens']} tok", ["-"] * r_i + [f"{x['ttft_s']:.2f} s"] + ["-"] * (len(runs) - r_i - 1), "new")

    for kind in ("code", "prose"):
        vals = [r["decode"].get(kind, 0) for r in runs]
        row(f"decode {kind} (tok/s)", vals, f"{vals[-1] / vals[0]:.2f}x" if vals[0] else "")
    row("needle (80% of n_ctx)", [("found" if r["needle"]["found"] else "MISSED") if r.get("needle") else "-" for r in runs])


if __name__ == "__main__":
    main()
