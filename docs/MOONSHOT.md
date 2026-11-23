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

## Branch `megakernel`, iteration 2: the fused projection kernel (stage "proj")

### q4_0 full-attention cache (LLAMA_INKLING_KV_TYPE_FULL=q4_0) produced garbage on the 3090
Audit of the in-place q4_0 loader (`fattn_q4_0_ptr`, `fattn_load_chunk` in fattn-common.cuh) against ggml's
`dequantize_q4_0`: value i < 16 is the low nibble of `qs[i]`, value i + 16 the high nibble of `qs[i]`, both minus 8,
times `d`; a chunk of 8 values never straddles the nibble halves (chunk starts are multiples of 8), block and row
strides are whole 18-byte blocks (`fattn_kv_traits<4>`). No defect found by reading; the vec decode path is
upstream's q4_0 code. The previous test could not have caught a loader bug: its K rows had one non-zero value at
position 0 and constant V rows, both exactly representable and both insensitive to the nibble half read. New
cases `test_inkling_dims` in test-flash-attn-indexed-bias: 24 heads x 128, 8 KV heads, 2048 cells, random K/V,
expected values from ggml's own dequantization of the quantized cache (`ggml_get_type_traits()->to_float`),
f16/q8_0/q4_0, 1/4/128/512 queries (vec, vec-verify, MMA), masked, maskless causal and maskless window 1024.
On the 3090 this run localizes the fault: q4_0 at 128/512 queries = the in-place loader (MMA/tile), at 1 query =
the vec kernel; `LLAMA_INKLING_FA_Q8_0=0` switches the prefill shapes to the F16-copy path for comparison. Please
also report the exact environment of the garbage run (`LLAMA_INKLING_FA_VEC_GQA`, `LLAMA_INKLING_NO_MASK`).

### Stage "proj": `LLAMA_INKLING_MK_PROJ=1` (default off)
Decode runs ~2500 kernels per token; every quantized MUL_MAT of one token is two of them (q8_1 quantization of the
input, then mmvq), and a layer application has 5 of those in the attention projections (Q, R, K, V, gate) plus wo.
- `src/models/inkling2.cpp`: the projections are built and expanded consecutively, quantized ones first, then the
  BF16 sinks/conv taps, then the convolutions (the graph was interleaving K's convolution between the K and V
  projections). Same ops, same results; the order changes which fusions can match.
- `ggml/src/ggml-cuda/mmvq.cu`: `mul_mat_vec_q_multi<type, small_k>`: grid.y selects the matrix, grid.x its row
  block; the body is mul_mat_vec_q's ncols_dst == 1 body (same thread-to-block mapping, accumulation order,
  shared-memory and warp reductions), so every output is bit-identical to the unfused kernel. Host side
  `ggml_cuda_mul_mat_vec_q_multi`: each distinct input quantized once with `quantize_row_q8_1_cuda` (identical
  bytes), one launch per (type, small_k) group. Types: Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL
  (every attention type of UD-Q3_K_XL; BF16 stays on mmvf).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_mk_proj_match` in try_fuse: a run of >= 2 consecutive one-token
  MUL_MAT nodes with contiguous quantized weights that the unfused path would also send to mmvq, no run member
  reading another's output, no compute-buffer weights. Views between them are allowed.
- Launches saved per token (estimate): pass-1/non-loop layers 5 mmvq + 5 quantize -> 1 + 1 (-8), loop layers
  -> 1 + 3 (-6); tail layers with mixed types a little less: ~300 launches per token, roughly -1.2 to -1.6 ms
  (75 -> ~83-85 tok/s). Lossless (bit-identical).
- `LLAMA_INKLING_MK_DEBUG=1`: the kernel checks its arguments per block and prints block/matrix on a violation.

Self-test `test-inkling-megakernel` (CMake target of the same name): seven projection sets at the real shapes
(core Q8_0 with one and with three inputs, the layer-18/20/22 type mixes, two and eight products), reference
(`MK_PROJ=0`) vs fused (`MK_PROJ=1`) on the same backend, CUDA graphs disabled so both really run, bit-identity
required, wall time per evaluation printed for both (50 repetitions). On FAIL it prints max abs/rel error and the
first mismatching product, element and values.

### Next stages (designed, not implemented)
- "wo + gate": the attention output projection fused with the sigmoid gate multiply (the gate is already a
  separate mmvq in the run): a `mul_mat_vec_q` fusion argument, like the existing gate/bias fusion.
- "router": sigmoid + bias + top-8 + norm + scale is one fused kernel already; the remaining launches are the two
  mul_mat_id (gate_up, down) with their q8_1 quantization (2) and the SITU and the weighted reduction (fused): a
  mul_mat_id variant that quantizes its own input saves 2 launches per layer application.
- "glue": rs state gather + the three norms + the controllers' two kernels per controller (8): one kernel per
  layer application that produces the three normalized inputs and the q8_1 bytes for the projection kernel,
  which then needs no quantize launch at all.
- The whole layer application as one cooperative kernel (grid sync between phases, 82 SMs) once the pieces above
  are each verified on the 3090.

## Branch `megakernel`, iteration 3: self-quantizing products (stages proj, wo, moe)

The common piece: `mul_mat_vec_q_selfq<type, small_k>` (mmvq.cu). A block quantizes the input of its product to q8_1
in shared memory with the arithmetic of `quantize_q8_1` (one lane per value, warp max and warp sum over 32 lanes,
`d = amax/127`, `q = roundf(x/d)`, `ds = (d, d*sum)`), optionally after `sigmoid(g)*x` with the expression of the
fused SIGMOID+MUL kernel, then runs the ncols_dst == 1 body of `mul_mat_vec_q` on it. No quantize launch, no pool
allocation, bit-identical to quantize + mmvq. Types: Q4_0, Q5_0, Q8_0, Q2_K..Q6_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
IQ4_XS, IQ4_NL, NVFP4, MXFP4 (all attention and expert types of UD-Q3_K_XL); inputs up to 3072 values.

| Stage | Switch | Fuses | Launches saved per layer application | Per token |
|---|---|---|---|---|
| proj | `LLAMA_INKLING_MK_PROJ=1` (`=2`: the iteration-2 variant with separate quantize launches) | the run of one-token quantized MUL_MATs (Q, R, K, V, gate): one kernel per weight type, inputs quantized in-kernel | pass-1 layers 10 -> 1 (-9), loop layers 10 -> 1 (-9), tail layers one per type group | ~430 |
| wo | `LLAMA_INKLING_MK_WO=1` | SIGMOID(gate), MUL(attn, .), MUL_MAT(wo, .) as one self-quantizing product (the gate run in `ggml_cuda_try_fuse` before the generic UNARY+MUL fusion) | 3 -> 1 (-2) | ~96 |
| moe | `LLAMA_INKLING_MK_MOE=1` | a one-token MUL_MAT_ID (gate_up: the token for every expert slot; down: one input per slot) quantizes its slots' inputs in the kernel; taken in `ggml_cuda_mul_mat_id` under exactly the conditions of the mmvq path | 2 x (2 -> 1) (-2) | ~96 |

Estimate for all three on: ~620 launches fewer per token out of ~2500, at ~2-5 us each: -1.5 to -3 ms per token,
75 -> ~85-95 tok/s. Lossless by construction (same bytes into the same dot-product code).

Rules kept (from the 3090 failures of iterations 1-2): no arithmetic on a possibly-null pointer before its null check
(`g != nullptr` guards the gate reads); no container of `ggml_cuda_pool_alloc` (the self-quantizing kernels allocate
nothing; `=2` keeps the coordinator's single-allocation fix).

Self-test `test-inkling-megakernel` (target of the same name), now with CUDA graphs enabled: every run builds a
fresh graph object, its first evaluation is captured and the 50 timed ones replay it, as llama-server does; the
timed numbers are therefore replay times. Cases: proj x8 (core one/three inputs, layer 18/20/22 type mixes, two
products, eight products with eight distinct inputs and with shared inputs = pool stress), proj2 (four inputs,
the separate-quantize variant), wo x2 (Q8_0, Q6_K), moe gate_up x5 (NVFP4, IQ3_XXS, IQ3_S, IQ2_S, Q3_K; 16
experts, 8 used) and down x2 (NVFP4, IQ4_NL; one input per slot). Each prints `PASS/FAIL ref <us> fused <us>` and
on FAIL max abs/rel error and the first mismatching output/element/values.

Commands (compose service):
```
docker compose run --rm --no-deps --entrypoint /app/test-inkling-megakernel inkling
docker compose run --rm --no-deps -e LLAMA_INKLING_MK_DEBUG=1 -e CUDA_LAUNCH_BLOCKING=1 --entrypoint /app/test-inkling-megakernel inkling
docker compose run --rm --no-deps -e LLAMA_INKLING_MK_PROJ=1 -e LLAMA_INKLING_MK_WO=1 -e LLAMA_INKLING_MK_MOE=1 --entrypoint /app/test-llama-archs inkling -a 'inkling2.*'
# then greedy identity and bench.py decode with the three switches, one at a time and together
```

Next: stage "glue" (rs gather + the three norms + the four controllers as one kernel per layer application that also
emits the q8_1 bytes for the projection kernel; needs the realugbun kernels refactored into device functions to stay
bit-identical), then the layer application as one cooperative kernel.
