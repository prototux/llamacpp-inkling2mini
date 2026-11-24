# Patches

Applied in file-name order with `git apply` on upstream llama.cpp at the commit pinned in
[../docker/Dockerfile](../docker/Dockerfile) (`LLAMA_CPP_COMMIT`, ggml-org/llama.cpp master of 2026-11-23).

| Patch | Origin | Content |
|---|---|---|
| `0001-cuda-indexed-bias-flash-attention.patch` | pwilkin, branch [`codex/inkling2-flash-attention-cuda`](https://github.com/pwilkin/llama.cpp/tree/codex/inkling2-flash-attention-cuda) at `b9ea31a8d2`; its ggml API and CPU reference were merged upstream as [#32740](https://github.com/ggml-org/llama.cpp/pull/32740), so the patch now carries the CUDA/HIP kernels, the Inkling 2 graph and the bias added before the mask's F16 rounding (upstream: in F32), as the CUDA tiles do | indexed relative-position bias in `GGML_OP_FLASH_ATTN_EXT`, CPU reference, CUDA vec/tile kernels, Inkling 2 graph |
| `0002-inkling2-dspark-speculative-decoding.patch` | upstream PR [#32218](https://github.com/ggml-org/llama.cpp/pull/32218) at `e4b1cb5318` | DSpark draft model, converter and speculative decoding |
| `0003-inkling2-fused-cuda-glue-kernels.patch` | realugbun, branch [`inkling2-speed`](https://github.com/realugbun/llama.cpp/tree/inkling2-speed), commits `7ec7714fa` `26739461d` `520a486b0` `0f6ac0c9c` | fused CUDA kernels for the Inkling 2 glue op chains; one conflict with 0001 resolved by keeping the indexed-bias graph |
| `0004-inkling2-attention-prefill-position-bias-q8-kv.patch` | this repository | tensor-core prefill with the bias, position-mode bias, q8_0 K/V read in place, tests |
| `0005-inkling2-loop-kv-sharing-and-drift-tool.patch` | this repository | opt-in cross-pass KV sharing, `llama-inkling-pass-kv` |
| `0006-inkling2-dspark-verify-fixes.patch` | this repository; adaptive drafting ported from [#32773](https://github.com/ggml-org/llama.cpp/pull/32773) by Oxsilaris06 | fusable convolution-state chain with rollback snapshots, vec attention kernel for verification batches, adaptive drafting, `LLAMA_DSPARK_BLOCK`, `LLAMA_SPEC_RS_SEQ` |
| `0009-inkling2-thinking-off-answers-directly.patch` | this repository | with thinking disabled the Inkling parser opens the text block in the prompt (not with tools), and a server default of `--reasoning-effort none` disables thinking; a parser test |

| `0007-inkling2-moonshot-decode-attention-maskless-q4-kv.patch` | this repository | fewer one-token kernels (router norm, sinks, graph trim), decode attention per KV head, attention without a mask tensor, q4_0 full-attention cache (known bug), graph statistics |
| `0008-cuda-multi-stream-decode-plan.patch` | this repository | multi-stream plan for small-batch CUDA graphs (`GGML_CUDA_GRAPH_OPT=2`), fix for the vec kernel with a NULL mask, flash-attention tests at the Inkling 2 shapes |

The build applies the patches listed in `series`, in that order.

To update: rebase the changes on a newer upstream commit, regenerate the patches with `git diff --binary` between
the successive steps, set `LLAMA_CPP_COMMIT`, and check that applying them reproduces the tree you tested
(`git write-tree`).
