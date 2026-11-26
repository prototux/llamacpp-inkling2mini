# llamacpp-inkling2mini

[![build](https://github.com/prototux/llamacpp-inkling2mini/actions/workflows/build.yml/badge.svg)](https://github.com/prototux/llamacpp-inkling2mini/actions/workflows/build.yml)

[llama.cpp](https://github.com/ggml-org/llama.cpp) patched and tuned for
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, 26.4B looped MoE,
3.4B active, vision, 1M-token context) on **a single 24 GB NVIDIA GPU**, packaged as a Docker image with a
ready-to-use Compose file. It runs fully offline once the image and the model files are on the machine.

## Results on an RTX 3090 (24 GB)

| | Stock llama.cpp | This build | Gain |
|---|---:|---:|---:|
| Prompt processing (prefill) | 800 tok/s | **2,000 tok/s** | 2.5x |
| Generation | 50 tok/s | **110 tok/s** | 2.2x |
| Maximum context | 42K tokens | **593K tokens** | 14x |

Measured on an RTX 3090 with nothing else on the GPU (`scripts/bench.py`, 2026-11-25). This build: Unsloth
UD-Q3_K_XL weights, q8_0 KV cache, vision encoder on the CPU, thinking off, default settings of this repository.
Prefill on 7K-30K-token prompts (1,260 tok/s at 118K, 500 tok/s at 493K); generation at a short context with the
default sampling, on the GPU (114 tok/s greedy; 98 tok/s after a 17K-token prompt, ~84 after 54K); a fact hidden at 438K tokens is recalled. "Stock" is the official
`ghcr.io/ggml-org/llama.cpp:server-cuda` image of 2026-11-23.

## What is different from stock llama.cpp

| # | Change | Origin | Effect | Switch |
|---|---|---|---|---|
| 1 | CUDA flash attention with Inkling's relative-position bias (an index into a per-head bias table) | [pwilkin](https://github.com/pwilkin/llama.cpp/tree/codex/inkling2-flash-attention-cuda), upstream [#32740](https://github.com/ggml-org/llama.cpp/pull/32740) | stock llama.cpp materializes a dense `heads x context x batch` bias and skips flash attention: VRAM grows with the context, prompts past ~79K tokens abort ([#32804](https://github.com/ggml-org/llama.cpp/issues/32804)) and the V cache cannot be quantized | - |
| 2 | Fused CUDA kernels for Inkling 2's small "glue" ops (convolutions, clipping, depth mixing, loop gates) | [realugbun/llama.cpp `inkling2-speed`](https://github.com/realugbun/llama.cpp/tree/inkling2-speed) | far fewer kernel launches per token: faster prefill and generation | - |
| 3 | Prefill attention on the tensor cores: the MMA kernel adds the bias to its mask tile | this repo | faster time to first token on long prompts (the bias used to force the CUDA-core tile kernel) | `LLAMA_INKLING_FA_MMA` |
| 4 | Bias bins computed in the attention kernel from token positions | this repo | removes the `context x batch` index table that the CPU filled and uploaded every batch: less VRAM, faster long prefill | `LLAMA_INKLING_POS_BIAS` |
| 5 | q8_0 KV cache read in place by the prefill kernels | this repo | no F16 copy of the whole cache per batch: ~4 KiB/token of VRAM back for context | `LLAMA_INKLING_FA_Q8_0` |
| 6 | Optional sharing of the full-attention KV between the 3 passes of the loop core | this repo | ~2x context, **off**: long answers degrade into repetition | `LLAMA_INKLING_SHARE_LOOP_KV` |
| 7 | DSpark speculative decoding | upstream [#32218](https://github.com/ggml-org/llama.cpp/pull/32218) | **off**: ~50% acceptance on code, measured slower than plain decoding on a 3090 before the fixes of row 8 | `DSPARK` |
| 8 | Speculative-decoding fixes: the target's rollback snapshots no longer break the fused convolution kernels (~1.2-1.5K extra kernel launches per verify step before), verification batches on the vec attention kernel, adaptive drafting from [#32773](https://github.com/ggml-org/llama.cpp/pull/32773), a shorter draft block option | this repo | DSpark verify steps should get much cheaper; not yet measured, so DSpark stays off by default | `LLAMA_INKLING_FA_VEC_VERIFY`, `LLAMA_SPEC_ADAPT`, `LLAMA_DSPARK_BLOCK` |
| 9 | Fewer kernels per generated token: the MoE router norm written so the existing fused top-k kernel covers it, attention sinks passed to flash attention directly, trimmed one-token graphs | this repo | ~300 fewer kernel launches per token (est. +8-10% generation) | `LLAMA_INKLING_MOE_CLAMP_NORM`, `LLAMA_INKLING_GRAPH_TRIM`, `LLAMA_INKLING_DECODE_SINKS` (on) |
| 10 | Decode attention reads each KV head once for its 3 query heads instead of 3 times | this repo | +5% generation at 90K context, -2% at short context, so `auto` uses it only for caches of 64K cells and more | `LLAMA_INKLING_FA_VEC_GQA` (off; `auto` or `1`) |
| 11 | Attention without a mask tensor: causal and sliding-window masking derived from token positions in the kernels | this repo | identical output, less VRAM at long context, +1% generation | `LLAMA_INKLING_NO_MASK` (on) |
| 12 | Multi-stream decode: the independent op chains of each layer (Q, K, V, bias, sinks, gate) run on side streams inside the CUDA graph, fusions kept | this repo | measured slightly slower than the default on a 3090 | `GGML_CUDA_GRAPH_OPT=2` (off) |
| 13 | With thinking off, the prompt opens the answer block | this repo | Inkling 2 thinks at length even at effort 0 (`none`): the thinking used the token budget and long answers came out cut short | `REASONING_EFFORT` |
| 14 | Decode fusions: the loop controllers, input norms, depth mix and their quantization ("glue"), the attention bias table, the expert scales, a layer's attention projections, the gated output-projection input and the Q/K/V epilogue (clip, rotation, cache writes) each in one kernel, plus a fused SITU for every layer; every one bit-exact against the unfused graph, checked on the GPU with `LLAMA_INKLING_MK_CHECK` | this repo | ~2,390 -> ~1,600 kernel launches per token: 90 -> ~106 tok/s on a 3090 | `LLAMA_INKLING_MK_GLUE`, `_MK_RELTAB`, `_MK_MOESCALE`, `_MK_PROJ`, `_MK_WOQ`, `_MK_QKV` (on) |
| 15 | Q8_0/Q6_K one-token products with the row length known at compile time, so all weight loads of a thread are issued at once; same operations in the same order, bit-identical | this repo | +2% generation | `LLAMA_INKLING_GEMV` (on) |

And the configuration the image runs by default:

- **every layer on the GPU**: moving experts to the CPU corrupts the looped layers in current llama.cpp
  ([#32710](https://github.com/ggml-org/llama.cpp/issues/32710), [#32778](https://github.com/ggml-org/llama.cpp/issues/32778));
- **q8_0 K and V cache**: ~17 KiB per token, only 8 of the model's 48 attention slots keep a full-length cache;
- **context sized automatically** by `--fit` to fill the GPU, keeping 768 MiB free;
- **vision encoder on the CPU**: images encode a little slower, the VRAM goes to context;
- **thinking off by default**, enabled per request with `"reasoning_effort"`.

The reasoning behind each choice, the memory arithmetic and the kernel details are in [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

Requirements: Linux, an NVIDIA GPU with 24 GB (tuned on an RTX 3090; the image is built for sm_86 and sm_89),
NVIDIA driver >= 570, Docker with Compose v2.24+ and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
On Debian 13, `sudo host/setup-debian13.sh` installs and configures the toolkit and checks the driver.

```bash
git clone https://github.com/prototux/llamacpp-inkling2mini && cd llamacpp-inkling2mini
mkdir -p models
docker compose --profile download run --rm download    # ~14 GB, checked against SHA-256 (or see "Model files")
docker compose up -d
docker compose logs -f inkling                         # ready at "server is listening"
```

The API is on `http://127.0.0.1:8080` (OpenAI-compatible under `/v1`), the llama.cpp web UI on the same address.
The log line `n_ctx = ...` shows the context `--fit` chose; to make restarts reproducible, copy `.env.example` to
`.env` and set `CTX` to that value.

To update: `git pull && docker compose pull --policy always && docker compose up -d`. The compose file sets
`pull_policy: missing` so that a machine without network starts from its local image, which also makes a plain
`docker compose pull` skip an image that is already present.

## Model files

Put these files in `./models` (or the directory `MODELS_DIR` points to). Download them once, on any machine;
after that nothing needs the network.

| File (name in `./models`) | Size | Download | SHA-256 |
|---|---:|---|---|
| `Inkling-2-Mini-UD-Q3_K_XL.gguf` | 13.6 GB | [unsloth/Inkling-2-Mini-GGUF `Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file) | `e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639` |
| `mmproj-Inkling-2-Mini-Q8_0.gguf` (vision) | 346 MB | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/mmproj-Inkling-2-Mini-Q8_0.gguf) | `233287272ec5ac39b4168677aba54a4c024f202410f5629fac1830e1059f8abb` |

The weights file is published as `.gguf_file` in Unsloth's `Mainline_Support` folder (the folder for upstream
llama.cpp's tensor layout; the files at the repository root use Unsloth's own layout and do **not** load here).
Save it as `.gguf`:

```bash
cd models
curl -L -o Inkling-2-Mini-UD-Q3_K_XL.gguf \
  https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file
curl -L -O https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/mmproj-Inkling-2-Mini-Q8_0.gguf
sha256sum Inkling-2-Mini-UD-Q3_K_XL.gguf mmproj-Inkling-2-Mini-Q8_0.gguf   # compare with the table
```

Other Unsloth quants from the same `Mainline_Support` folder work too (set `MODEL_FILE`): UD-IQ4_XS (15.7 GB) and
UD-Q4_K_XL (17.6 GB) are a little more accurate, with less room for context.

**Air-gapped machine**: on a connected machine, `docker pull ghcr.io/prototux/llamacpp-inkling2mini:latest` and
`docker save ghcr.io/prototux/llamacpp-inkling2mini:latest | gzip > inkling-image.tar.gz`; copy it with this
repository and `./models`, then `docker load < inkling-image.tar.gz` and `docker compose up -d`.

## Using it

```bash
# chat, thinking off (the default)
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello"}]}'

# thinking on for this request: none, minimal, low, medium, high, xhigh, max
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "reasoning_effort": "high",
  "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}]}'

# image input
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 photo.png)"'"}}]}]}'
```

The thinking text comes back in `message.reasoning_content`. A numeric effort (0 to 0.99) goes in
`"chat_template_kwargs": {"reasoning_effort": 0.5}`.

## Configuration

Copy `.env.example` to `.env`; every setting is optional. The main ones:

| Setting | Default | Effect |
|---|---|---|
| `CTX` | empty (auto) | fixed context size; empty lets `--fit` pick the largest that fits |
| `FIT_MARGIN_MIB` | 768 (1536 with `VISION=gpu`) | VRAM `--fit` keeps free; raise it if a desktop shares the GPU |
| `KV_TYPE` | `q8_0` | `q4_0`: ~2x context, worse long-context recall; `f16`: ~half the context |
| `VISION` | `cpu` | `cpu`: encoder off the GPU; `gpu`: faster images, ~50K tokens less; `off` |
| `UBATCH` | 512 | prefill micro-batch: larger = faster prompt processing, more VRAM |
| `REASONING_EFFORT` | `none` | default thinking effort |
| `BACKEND_SAMPLING` | `on` | sample on the GPU; `off` samples on the host (~8% slower generation) |
| `BIND_ADDR` / `API_KEY` | `127.0.0.1` / empty | set both before exposing the port |
| `DSPARK` | `off` | speculative decoding, see "Known limitations" |
| `EXTRA_ARGS` | empty | any other `llama-server` flags |

### Kernel switches

Changes 3 to 6, 8 to 12, 14 and 15 each have a switch (`1` = on, `0` = off). If `scripts/verify-kernels.sh` reports a failure on your GPU, turn
the matching switch off and restart (`docker compose up -d`); no rebuild is needed.

| Switch | Default |
|---|---|
| `LLAMA_INKLING_FA_MMA` | 1 |
| `LLAMA_INKLING_POS_BIAS` | 1 |
| `LLAMA_INKLING_FA_Q8_0` | 1 |
| `LLAMA_INKLING_SHARE_LOOP_KV` | 0 |
| `LLAMA_INKLING_FA_VEC_VERIFY` | 1 |
| `LLAMA_INKLING_MOE_CLAMP_NORM`, `LLAMA_INKLING_GRAPH_TRIM`, `LLAMA_INKLING_DECODE_SINKS` | 1 |
| `LLAMA_INKLING_NO_MASK`, `LLAMA_INKLING_MK_GLUE`, `LLAMA_INKLING_MK_RELTAB`, `LLAMA_INKLING_MK_MOESCALE`, `LLAMA_INKLING_MK_PROJ`, `LLAMA_INKLING_MK_WOQ`, `LLAMA_INKLING_MK_QKV`, `LLAMA_INKLING_GEMV` | 1 (set by `docker/start-server.sh`) |
| `LLAMA_INKLING_FA_VEC_GQA` | 0 (`auto` for long contexts) |
| `GGML_CUDA_GRAPH_OPT` | unset (`2` to enable) |

## Checking an installation

All scripts run on the GPU host, from the repository root.

| Script | What it does |
|---|---|
| `scripts/verify-kernels.sh` | compares every CUDA attention path and the fused kernels against the CPU reference, and the decode fusions bit for bit against the unfused graph (no model needed) |
| `scripts/smoke_test.py [--needle N]` | text, per-request thinking, vision, and optionally recall of a fact hidden in an N-token prompt |
| `scripts/bench.py --out results/x.json` | context size, time to first token at several prompt lengths, generation speed, recall |
| `scripts/compare.py a.json b.json` | two benchmark runs side by side |
| `scripts/quality.sh` | perplexity with and without a setting (by default the loop KV sharing), with the server stopped |
| `scripts/mk-bench.sh` | generation speed per kernel-switch configuration, at short context and after a long prompt |
| `scripts/compare-quality.sh`, `scripts/bisect-loop.sh` | token probabilities and long greedy answers per configuration: finds a setting that changes the output |
| `scripts/check-thinking.sh` | thinking off by default, on per request |

## Binaries

Every push to `main` publishes a [release](https://github.com/prototux/llamacpp-inkling2mini/releases) with the
Linux x86_64 binaries from the image (`llama-server`, `llama-perplexity`, `llama-inkling-pass-kv`, the kernel tests
and their libraries). They need NVIDIA driver >= 570 and the CUDA 12.8 runtime libraries (cudart, cuBLAS), which
are not bundled. Run them with `LD_LIBRARY_PATH=./lib ./llama-server ...` and the flags from
[docker/start-server.sh](docker/start-server.sh).

## Building from source

```bash
docker build -f docker/Dockerfile -t llamacpp-inkling2mini .                       # RTX 30xx (sm_86)
docker build -f docker/Dockerfile --build-arg CUDA_ARCH="86;89;120" --build-arg BUILD_JOBS=8 -t llamacpp-inkling2mini .
IMAGE=llamacpp-inkling2mini docker compose up -d
```

The build clones upstream llama.cpp at the commit pinned in the Dockerfile (`LLAMA_CPP_COMMIT`) and applies
[patches/](patches/) in order. Each nvcc job needs 1 to 2 GB of RAM; `BUILD_JOBS` sets how many run at once.

## Known limitations

- **One GPU, tested on an RTX 3090.** Multi-GPU splits of this model have open upstream issues.
- **DSpark speculative decoding is off.** With the draft model converted for this build (`scripts/convert-dspark.sh`),
  acceptance was ~50% on code and decoding dropped from 75 to ~63 tok/s on a 3090. Row 8 fixes the main cause found
  (unfused kernels in verification steps); to try it: convert the draft, set `DSPARK=on` and compare with
  `scripts/bench.py`. Adding `LLAMA_SPEC_ADAPT=0.33` may help further.
- **Requests with tools are ~8% slower**: llama.cpp samples on the host for any request with a grammar, and a
  tool-calling client (Open WebUI with tools, for example) sends one with every request. Being fixed.
- **Generation slows with context depth**: ~109 tok/s on a short prompt, ~83 after a 54K-token prompt; the decode
  attention kernel reaches only ~40% of the memory bandwidth at depth. `LLAMA_INKLING_KV_TYPE_FULL=q4_0` (q4_0 cache
  for the full-attention slots) is included but produces garbage output; leave it unset.
- **Loop KV sharing (`LLAMA_INKLING_SHARE_LOOP_KV=1`) makes long answers degrade into repetition**: short tests
  look fine, but a 3000-token greedy Terraform answer diverges from the default after ~60 characters and ends in
  a loop (`scripts/bisect-loop.sh`). Leave it off.
- **With thinking off the model sometimes declines to write a long file in full** ("I can't produce a fully runnable
  configuration...") and gives a skeleton, most often at temperature 0. This is the model, not this build: every
  configuration does it, all optimizations off included. `"reasoning_effort": "low"` (a few hundred thinking
  tokens) gave complete Terraform files in all runs; `REASONING_EFFORT=low` makes it the default.
- Tool calls with thinking off can fail with "unexpected empty grammar stack" in some setups
  ([#32818](https://github.com/ggml-org/llama.cpp/issues/32818)); `"reasoning_effort": "minimal"` avoids it.
- `--reasoning-budget` re-arms on every thinking block ([#32836](https://github.com/ggml-org/llama.cpp/issues/32836));
  bound long thinking with `max_tokens` instead.
- Inkling 2 Mini was released on 2026-11-20 and its llama.cpp support is days old: expect upstream changes.

## Credits and licenses

- Model: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines Lab, Apache 2.0.
  GGUF quants by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), vision projector by
  [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) by the ggml authors; Inkling 2 support upstream by its
  contributors; flash-attention bias work by pwilkin; fused kernels by realugbun; DSpark support from
  [#32218](https://github.com/ggml-org/llama.cpp/pull/32218).
- Everything in this repository is MIT ([LICENSE](LICENSE)); the patches modify MIT-licensed llama.cpp code.
