# megakernel test build (rev 8)

The GitHub repository (prototux/llamacpp-inkling2mini, build-2) plus two patches that are not published yet:

| Patch | Content | Switches (off unless noted) |
|---|---|---|
| `0007-inkling2-moonshot-decode-attention-maskless-q4-kv.patch` | decode attention reading each KV head once (M1), attention without a mask tensor (M2), router norm covered by the fused MoE kernel (M3, on), trimmed one-token graphs (M4, on), q4_0 full-attention cache (M5), graph statistics (M6) | `LLAMA_INKLING_FA_VEC_GQA`, `LLAMA_INKLING_NO_MASK`, `LLAMA_INKLING_MOE_CLAMP_NORM`, `LLAMA_INKLING_GRAPH_TRIM`, `LLAMA_INKLING_DECODE_SINKS`, `LLAMA_INKLING_KV_TYPE_FULL`, `LLAMA_GRAPH_STATS` |
| `0008-cuda-multi-stream-decode-plan.patch` | multi-stream decode plan (side streams inside the CUDA graph, fusions kept); fix for `NO_MASK` crashing the vec kernel (rev 2); **megakernel stage 1**: all attention projections of a layer in one kernel per weight type, bit-identical (rev 3); flash-attention tests at the real Inkling shapes for f16/q8_0/q4_0 (rev 3) | `GGML_CUDA_GRAPH_OPT=2`, `LLAMA_INKLING_MK_PROJ=1`, `LLAMA_INKLING_MK_DEBUG=1` |

Rev 4 fixes the CUDA pool abort ("GGML_ASSERT(ptr == ... pool_used)") with `LLAMA_INKLING_MK_PROJ=1`. Rev 2 fixed "illegal memory access" with `LLAMA_INKLING_NO_MASK=1`. Rev 3 added megakernel stage 1 and its self-test
(`test-inkling-megakernel`), and attention tests that should pinpoint the `LLAMA_INKLING_KV_TYPE_FULL=q4_0` garbage:
the q4_0 lines of the bias tests say which kernel is wrong (1 query = vec kernel, 128/512 = tile/MMA loader).
Design and estimates: [docs/MOONSHOT.md](docs/MOONSHOT.md).

Rev 6: MK_WO and MK_MOE were slower than unfused on the 3090 (78 -> 64 tok/s, every GPU block re-quantized the
whole input); they now quantize once above 64 rows and the self-test flags any fused case slower than its
reference as `PERF FAIL`. With MK_PROJ, the K/V norms move before the projections so every layer fuses all 5.

Rev 7: the glue stage (`LLAMA_INKLING_MK_GLUE=1`): the loop controllers, the input norms and their q8_1 quantization
of a layer in 1-2 kernels (loop-pass attention 14 launches -> 2), the q8_1 consumed directly by the projections and
the MoE (so MK_MOE becomes a gain). Estimated ~490 fewer launches per token. Not yet run on a GPU.

Rev 8: `scripts/mk-diagnose.sh` runs the two diagnostics that should explain the open problems: `MK_CHECK`
compares every fused stage with the unfused computation bit for bit inside the real server (finds the kernel behind
the repetition loop, or clears them all), `MK_PROFILE` times each decode step per kernel class (explains why glue gave
no speedup). New stages: `MK_GLUE=2` (controllers as one cooperative kernel) and the depth mix folded into the glue.

Rev 9 (measured on the 3090, CUDA graphs on, `scripts/mk-bench.sh`): the repetition loop was `LLAMA_INKLING_SHARE_LOOP_KV`,
not a megakernel stage. The glue is now bit-exact (MK_CHECK: 0 mismatching outputs), and new bit-exact stages
`MK_RELTAB`, `MK_MOESCALE`, `GEMV` (a faster Q8_0/Q6_K one-token kernel) and a faster `MK_PROJ` bring decode to
~103 tok/s short / ~80 at 54K, from ~90 / ~73 with everything off. Recommended: `LLAMA_INKLING_MK_GLUE=1
LLAMA_INKLING_MK_RELTAB=1 LLAMA_INKLING_MK_MOESCALE=1 LLAMA_INKLING_GEMV=1 LLAMA_INKLING_MK_PROJ=1 LLAMA_INKLING_NO_MASK=1`.
`MK_WO`, `MK_MOE`, `MK_GLUE=2` and `GGML_CUDA_GRAPH_OPT=2` measured neutral or slower; see `.env.example`.

## Install from ghcr.io (any machine)

```bash
git clone -b megakernel https://github.com/prototux/llamacpp-inkling2mini && cd llamacpp-inkling2mini
ln -s /path/to/your/models models
docker compose pull --policy always     # ghcr.io/prototux/llamacpp-inkling2mini:megakernel, built by CI (pull_policy: missing skips a plain pull)
SKIP_BUILD=1 scripts/test-megakernel.sh # tests the pulled image; or `docker compose up -d` to just run it
```

## Run from the tarball

```bash
tar xzf llamacpp-inkling2mini-megakernel.tar.gz && cd llamacpp-inkling2mini-megakernel
ln -s ../inkling-models models          # or wherever your GGUFs are (MODEL_FILE / MMPROJ_FILE in .env)
docker compose down                     # in your other checkouts too: the tests need the whole GPU
scripts/test-megakernel.sh              # ~1-2 h; results/megakernel-<date>/summary.txt
```

Send back `summary.txt` and the log of any FAIL or DIFFERENT line.
To use it afterwards: `cp .env.example .env`, enable the switches that passed, `docker compose up -d`.

The build applies exactly the patches listed in `patches/series`, in that order, so leftover patch files from an
older test build in `patches/` are ignored.
