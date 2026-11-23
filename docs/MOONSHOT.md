# Inkling 2 Mini on one RTX 3090: moonshot branch

Branch `moonshot`, on top of `inkling-3090-exp`. Written on a machine without a GPU: the CUDA code here is
**uncompiled**. Every change sits behind its own environment switch so that it can be A/B'd and disabled alone.

## Where the time goes (derived from the code and the user's measurements, not profiled)

Model facts (HF config + GGUF header of `Inkling-2-Mini-UD-Q3_K_XL`): hidden 2304, 24 heads x 128, 8 KV heads
(GQA 3), 128 experts / 8 used, expert FFN 1152, loop core layers 6-17 x 3 passes = 48 layer applications per
token, full attention on layers 5, 11, 17, 23 (8 cache slots), SWA 1024 elsewhere, relative-position bias with
16 x 1024 tables (+1 far bin), per-query sinks, route scale 8, router norm eps 1e-20. In the UD-Q3_K_XL file the
core attention weights are Q8_0 (~27 MB per layer), the core experts NVFP4 with per-expert scales (~36 MB per
layer application for 8 experts), the output head Q5_K (320 MB).

- Weights read per decoded token: ~3 GB (48 x ~63 MB + head) = ~3.3 ms at 936 GB/s. Measured decode: 13.3 ms
  (75 tok/s) at short context. Decode is bound by the number of kernels per token (estimated 40-60 per layer
  application after the existing fusions, 2000-3000 per token, ~2-5 us each), not by bandwidth.
- At deep context the KV read dominates: 8 full-attention slots x 2176 B/token (q8_0) = 17 KiB per context
  token, 8.7 GB per decoded token at 500K (9 ms). The decode flash-attention kernel (`fattn-vec.cuh`, one CUDA
  block per *Q head*) reads each KV head 3 times (GQA ratio 3), so the real traffic is up to 3x that unless L2
  catches the reuse (6 MB L2 against three 1 GB streams: unlikely).
- Prefill at depth: the f16 attention mask is `n_kv x n_ubatch` per full-attention cache: 1 GiB of compute buffer
  at 1M context / ubatch 512, filled on the host (`std::copy` per token row) and uploaded every ubatch.

## Ideas, ranked

| # | Idea | Expected gain | Lossless | Effort | Status |
|---|------|---------------|----------|--------|--------|
| M1 | Decode FA vec kernel processes the 3 Q heads of a KV head in one block: K/V read once per KV head | deep-context decode: KV traffic /3 (500K: ~9 ms -> ~3 ms per token, ~1.4x at 500K; more at 1M) | same math, split-K differs: not bitwise | medium, CUDA | implemented, uncompiled, `LLAMA_INKLING_FA_VEC_GQA=1` |
| M2 | Maskless causal attention: in position mode the kernels derive causal/SWA/empty masking from the positions they already have, the f16 mask is not built, filled, uploaded or stored | -1 GiB compute buffer at 1M (~60K more context); deep prefill: no per-ubatch 1 GiB host fill + upload (est. 10-30% TTFT at 500K+) | bitwise | medium, ggml API + CPU + 3 CUDA kernels + kv-cache | implemented, CPU-tested, CUDA uncompiled, `LLAMA_INKLING_NO_MASK=1` |
| M3 | Router norm written as CLAMP instead of SCALE_BIAS so the existing CUDA top-k MoE fusion covers sum/norm/scale | -4 kernels per layer application, -192 per token (~1 ms, ~5-7% decode) | identical for routing sums > 1.7e-13 (always) | tiny, graph only | implemented, `LLAMA_INKLING_MOE_CLAMP_NORM=0` disables |
| M4 | Decode graph trims: real sinks instead of a zero fill + padded/shifted table (saves PAD, SUB, FILL per layer), one shared zero-sink tensor for prefill, lm_head over the unpadded vocabulary rows (no CONT) | -3 kernels per layer application for decode (-144 per token, ~3-4%) | sinks path: equivalent, not bitwise; the rest bitwise | small, graph only | implemented, `LLAMA_INKLING_DECODE_SINKS=0`, `LLAMA_INKLING_GRAPH_TRIM=0` disable |
| M5 | Separate KV type for the full-attention cache (q4_0) while SWA stays q8_0; the tile/MMA prefill kernels read q4_0 in place (no F16 copy) | full-attention KV 17 -> 9 KiB/token: ~2x context, deep decode KV traffic /2 | lossy (perplexity test) | small + CUDA loader | implemented, CUDA uncompiled, `LLAMA_INKLING_KV_TYPE_FULL=q4_0` |
| M6 | `LLAMA_GRAPH_STATS=1`: node histogram per batch size in the log | diagnostic | - | tiny | implemented |
| - | Self-speculative decoding with 1-pass drafts of the loop | none | - | - | rejected: a 1-pass draft costs ~0.5 decode steps; with verify at >=1.2 steps the break-even acceptance is ~0.9; DSpark's draft is 10x cheaper and already loses |
| - | Fused Q/K/V/gate projection weights (one mmvq per layer instead of 4-6) | -2-4 kernels per layer application (~2%) | bitwise | medium (GGUF transform + loader) | deferred: types differ across layers (Q8_0 core, Q4_K-Q6_K tail), small gain |
| - | GQA-aware MMA prefill kernel (ncols2 = 3) with the per-head bias | prefill attention K/V loads /3 | yes | high | not done: prefill attention is compute-bound, ncols2 must divide the tensor-core tile |
| - | Persistent megakernel / multi-stream CUDA graph | decode up to 2x | yes | very high | not attempted blind |
| - | Bigger ubatch (2048) for prefill, `--backend-sampling` | prefill MoE GEMMs 4x more rows per expert; no 800 KB logits D2H per token | yes | config | in the test plan |

## Decisions

- M4 sinks: for a one-token graph the per-query sink logit is passed to flash attention as the `sinks` tensor
  instead of being folded into the bias table (`table - sink`, zero sink, padded far bin). Same softmax,
  different rounding path.
- M2 encodes "no mask" as `mask == NULL` on a position-bias flash attention plus op param 6 (`causal window`:
  0 none, > 0 masked when `q_pos - k_pos >= window`). The graph uses it only when every stream of the ubatch is
  exclusively one sequence (O(1) check: the sequence's cell count equals the used cells), the attention is causal
  and the device supports the op; the masked graph is built otherwise, so multi-sequence batches keep working.
- M1 keeps `launch_fattn`'s grid logic: `ncols2 = 4` (>= GQA 3) gives one block per KV head; the kernel maps
  its columns to the Q heads of that KV head, columns past the GQA ratio are idle.

## Switches

| Switch | Default | Effect |
|---|---|---|
| `LLAMA_INKLING_MOE_CLAMP_NORM` | on | router norm as CLAMP (fusable); `=0` restores SCALE_BIAS |
| `LLAMA_INKLING_GRAPH_TRIM` | on | shared zero sinks, lm_head over the real vocabulary; `=0` restores the old graph |
| `LLAMA_INKLING_DECODE_SINKS` | on | one-token graphs pass the real sink logits to flash attention; `=0` folds them into the table |
| `LLAMA_INKLING_NO_MASK` | off | `=1`: single-sequence ubatches attend without a mask tensor (position-derived masking) |
| `LLAMA_INKLING_FA_VEC_GQA` | off | `=1`: decode flash attention reads each KV head once for its 3 Q heads |
| `LLAMA_INKLING_KV_TYPE_FULL[_K\|_V]` | unset | cell type of the full-attention cache, e.g. `q4_0` (K and V the same type for the in-place prefill loader) |
| `LLAMA_GRAPH_STATS` | off | `=1`: log the node count by op of each graph shape |

Existing switches from `inkling-3090-exp` keep working: `LLAMA_INKLING_FA_MMA`, `LLAMA_INKLING_POS_BIAS`,
`LLAMA_INKLING_FA_Q8_0`, `LLAMA_INKLING_FA_VEC_VERIFY`, `LLAMA_INKLING_SHARE_LOOP_KV`. `LLAMA_INKLING_NO_MASK`
needs `LLAMA_INKLING_POS_BIAS` (the default) and applies only when the device accepts the op.

## 3090 test plan

All commands through the compose service; `E` stands for the environment of the run, e.g.
`docker compose run --rm --no-deps -e LLAMA_INKLING_NO_MASK=1 --entrypoint /app/<binary> inkling <args>`.

1. Kernel correctness (CUDA against the CPU reference; the maskless, GQA-columns and q4_0 in-place cases are new):
   - `E=LLAMA_INKLING_FA_VEC_GQA=1 /app/test-flash-attn-indexed-bias` (all cases must print `max error` < 0.001 and exit 0;
     run it once more with `LLAMA_INKLING_FA_VEC_GQA=0` and once with `LLAMA_INKLING_FA_Q8_0=0`).
   - `/app/test-backend-ops -o FLASH_ATTN_EXT -b CUDA0` and `/app/test-backend-ops -o MUL_MAT -b CUDA0` (regression only).
2. Greedy identity (lossless switches). Same prompt, `--temp 0 --seed 1 -n 256`, compare the token streams:
   - baseline: all new switches off (`LLAMA_INKLING_MOE_CLAMP_NORM=0 LLAMA_INKLING_GRAPH_TRIM=0 LLAMA_INKLING_DECODE_SINKS=0`);
   - defaults (M3 + M4 on): expected identical tokens; a rare flip can only come from M4's sink rounding path,
     `LLAMA_INKLING_DECODE_SINKS=0` isolates it;
   - `LLAMA_INKLING_NO_MASK=1`: expected identical (bit-identical attention);
   - `LLAMA_INKLING_FA_VEC_GQA=1`: expected identical or a rare late flip (split-K order).
   Use `llama-perplexity -f <text> -c 4096 --chunks 8` for a numeric check on top: equal to 4 digits for M2, M3,
   M4; M1 within noise.
3. Decode speed at short context: `scripts/bench.py` (or `llama-bench -p 0 -n 128 -fa 1`) for: baseline
   (switches off), defaults, `LLAMA_INKLING_NO_MASK=1`, `LLAMA_INKLING_FA_VEC_GQA=1`. With `LLAMA_GRAPH_STATS=1` the
   server log shows the node count per token: expect ~190 fewer SUM_ROWS/SCALE/DIV/SCALE_BIAS nodes from M3 and
   ~100 fewer SUB/FILL/CONT nodes from M4 in the `n_tokens = 1` line.
4. Deep context (the M1 and M2 targets): fill 300K-500K tokens of context, then measure decode tok/s with
   `LLAMA_INKLING_FA_VEC_GQA=0` and `=1`; measure prefill tok/s and TTFT on the same prompt with `LLAMA_INKLING_NO_MASK=0`
   and `=1`, and read the compute buffer size in the startup log (`CUDA0 compute buffer size`): M2 should drop it by
   `2 * n_kv * n_ubatch` bytes per full-attention cache.
5. Context (M2, M5): `--fit` with `LLAMA_INKLING_NO_MASK=1`, and with `LLAMA_INKLING_KV_TYPE_FULL=q4_0`; pin the logged
   `n_ctx`. For q4_0 run `llama-perplexity -c 8192` against the q8_0 cache and decide on the loss.
6. Configuration experiments without code: `-ub 2048 -b 2048` for prefill (4x more rows per expert GEMM), and
   `--backend-sampling` for decode (no 800 KB logits copy per token).

## Branch `megakernel`: multi-stream decode (stage 1)

Decode after the fusions is ~2500 kernels per token in one serial CUDA-graph chain (~13 ms); the weights alone
would take ~3.3 ms. A real megakernel (one kernel per layer application) needs attention, 6 projections, convs,
router, 8 NVFP4/IQ expert GEMVs and the loop gates in one blind-written kernel: not attempted. Stage 1 attacks the
same floor generically: run the independent chains of the graph on several CUDA streams inside the CUDA graph, so
the launch latency and the tails of small kernels overlap, with every fusion kept.

`GGML_CUDA_GRAPH_OPT=2` (ggml-cuda.cu, `ggml_cuda_stream_plan_build`, off by default):

- dependencies between compute nodes by root tensor (views resolved), incl. write-after-read/write-after-write on
  in-place writes (KV cache `cpy`, conv states);
- a node continues its predecessor's chain when that predecessor is still the tail of its stream; a MUL_MAT /
  MUL_MAT_ID sibling (R, K, V, sinks, gate next to Q; conv_proj) takes a free side stream (`GGML_CUDA_GRAPH_OPT_STREAMS`,
  default 3, max 7); everything else joins a predecessor's stream. Chains stay on one stream so the realugbun and
  ggml fusions still match; fused groups wait on the cross-stream predecessors of their whole same-stream run and
  record the events of all their nodes, so a fusion can never break an ordering;
- one event per compute node; a window opens at the first side-stream node (fork event) and closes at the first
  main-stream node after which every side chain is consumed (for Inkling: the flash attention of each layer),
  where the main stream waits on all side streams; every tensor of a window is pinned until its end
  (`add_alloc_dep`) because ggml-alloc plans for the sequential order;
- only graphs with <= 512 rows per node (decode, DSpark verification) and only with CUDA graphs enabled;
  `GGML_CUDA_GRAPH_OPT_DEBUG=1` logs nodes/streams/windows/waits per graph.

Expected (estimate, not measured): the attention projections+convs section of each layer application has ~15
kernels in 4-6 chains with a critical path of ~5, plus the table/bias chain; ~20-25% of the kernels overlap:
roughly -1.5 to -2.5 ms per token (75 -> 85-95 tok/s). Lossless: the same kernels, the same order per stream.

Tests: `test-sched-streams` (new: a decode-like 6-block graph with sibling projections, in-place cache writes and
joins, through `ggml_backend_sched` against the CPU, twice to cover the CUDA-graph replay), and
`test-llama-archs -a 'inkling2.*'` on the GPU (CUDA logits vs CPU), both with `GGML_CUDA_GRAPH_OPT=2`
(`GGML_CUDA_GRAPH_OPT_STREAMS=1..7`). Then greedy identity and `bench.py` decode with `=2` vs unset.

Not done / next: RMS_NORM siblings as fork points (the three loop controllers; needs the fused CTRL_DOWN group to
tolerate a stream change, i.e. waits over the whole lookahead run rather than the same-stream run); overlapping
the expert GEMVs of a layer with the next layer's work is impossible (true data dependency); a per-layer
megakernel remains the only way below ~2500 launches.
