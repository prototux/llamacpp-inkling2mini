# Design notes

How this build gets Inkling 2 Mini to 575K tokens of context with vision on one 24 GB GPU, and why.

## The model, as far as memory is concerned

Inkling 2 Mini has 24 physical layers. Layers 6 to 17 form a loop core that runs 3 times per token, so the model
executes 48 layer applications and llama.cpp keeps 48 cache slots. Only layers 5, 11, 17 and 23 use full attention;
11 and 17 sit in the loop, which makes **8 full-attention slots**. The other 40 slots use a 1,024-token sliding
window. There is no RoPE: a per-head relative-position bias (1,024 distance bins plus one "far" bin, and attention
sinks) carries the positions, and short convolutions keep a few tokens of state per layer.

Per token of context, the KV cache holds 8 slots x 2 (K, V) x 8 KV heads x 128 = 16,384 values:

| KV type | Per token | 500K tokens |
|---|---:|---:|
| f16 | 32 KiB | 15.3 GiB |
| q8_0 | 17 KiB | 8.1 GiB |
| q4_0 | 9 KiB | 4.3 GiB |

The 40 sliding-window slots take ~110 MiB in total and the convolution state ~4 MB, whatever the context.

## Why llama.cpp

vLLM and SGLang support Inkling 2 only on Hopper, Blackwell and DGX Spark (NVFP4 kernels). llama.cpp runs it on
any CUDA GPU, but stock llama.cpp turns the relative-position bias into a dense `heads x context x batch` tensor
and then falls back to non-flash attention. On a 3090 that caps the context around 42K tokens, aborts long prompts
([#32804](https://github.com/ggml-org/llama.cpp/issues/32804)) and rules out a quantized V cache.

## The patches

Applied in order on top of upstream llama.cpp (see [../patches/README.md](../patches/README.md)):

1. **Indexed-bias flash attention** (pwilkin). The attention kernels receive the small bias table
   `[bins, heads, queries]` and an index per (query, key) instead of a dense bias. Flash attention works again,
   and so do quantized caches.
2. **DSpark** speculative decoding (llama.cpp #32218). Included but off by default, see below.
3. **Fused glue kernels** (realugbun). Inkling 2 runs dozens of tiny element-wise ops per layer application, 48
   times per token: generation is launch-bound. Fusing the op chains removes thousands of launches per token.
4. **Prefill and memory work in this repository:**
   - *Tensor-core prefill.* With 24 query heads over 8 KV heads (GQA ratio 3), each CUDA block of the MMA kernel
     handles one query head, so the bias of that head can be added to the mask tile while it is loaded into shared
     memory. The tensor-core math is unchanged. Before, the bias forced prefill onto the CUDA-core tile kernel.
   - *Position mode.* The kernels compute the bin as `clamp(q_pos - k_pos, 0, far)` from a position per cache
     cell and per query, instead of reading an index table of `context x batch` int32 that the CPU filled on one
     thread and uploaded every batch (about 1 GiB per batch at 500K tokens).
   - *q8_0 in place.* The prefill kernels used to convert the whole q8_0 cache to an F16 copy for every batch
     (4 KiB per token reserved in the compute buffer). q8_0 data is a run of 34-byte blocks of 32 values, and every
     row and head offset is a whole number of blocks, so the kernels now dequantize 8-value chunks straight into
     shared memory.
5. **Loop KV sharing** (opt-in). Passes 2 and 3 of layers 11 and 17 attend to the cache of pass 1, halving the
   full-attention KV. The model was trained with a cache per pass, so this trades quality for context; the
   `llama-inkling-pass-kv` tool measures how close the passes' K/V are on real text.

Each change of item 4 and 5 has an environment switch (README, "Kernel switches") and is covered by
`test-flash-attn-indexed-bias`, which compares every CUDA path with an exact expected result.

## Runtime choices

- **All layers on the GPU, no expert offload.** Offloading experts of looped layers to the CPU corrupts the output
  in current llama.cpp ([#32710](https://github.com/ggml-org/llama.cpp/issues/32710),
  [#32778](https://github.com/ggml-org/llama.cpp/issues/32778)). With `-ngl` set, `--fit` shrinks the context
  instead of moving layers.
- **UD-Q3_K_XL** (13.6 GB, 96.3% top-token agreement with the full model per Unsloth) leaves the most room for
  context; UD-IQ4_XS and UD-Q4_K_XL are the alternatives.
- **q8_0 KV**: close to lossless. q4_0 nearly doubles the context but costs long-context recall.
- **Vision encoder on the CPU** (`--no-mmproj-offload`): ~1 GiB of VRAM (weights + encoder buffers) goes to
  context; images take longer to encode.
- **Thinking off by default** with `--reasoning-effort none` (the template's own default is 0.9). `--reasoning off`
  is not used because it would also block thinking for requests that ask for it.
- **Sampling** follows the model's recommendation: temperature 1.0, top-p 1.0, top-k 0, min-p 0.

## Speculative decoding (DSpark): why it is off

Unsloth's draft GGUF predates the DSpark code in this build ("markov_w1.weight has wrong shape");
`scripts/convert-dspark.sh` converts the draft from the official checkpoint instead (BF16: the converter quantizes
the Markov head together with its 200K-vocabulary projection otherwise). With it, acceptance on code is ~50% at a
draft length of 3, and decoding goes from 75 to ~63 tok/s on a 3090: one verification step of 4 tokens costs ~3x a
single-token step, while plain decoding uses only ~15% of the GPU's memory bandwidth. The cause is under
investigation; until it is fixed, plain decoding is faster.
