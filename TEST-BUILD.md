# megakernel test build (rev 5)

The GitHub repository (prototux/llamacpp-inkling2mini, build-2) plus two patches that are not published yet:

| Patch | Content | Switches (off unless noted) |
|---|---|---|
| `0007-inkling2-moonshot-decode-attention-maskless-q4-kv.patch` | decode attention reading each KV head once (M1), attention without a mask tensor (M2), router norm covered by the fused MoE kernel (M3, on), trimmed one-token graphs (M4, on), q4_0 full-attention cache (M5), graph statistics (M6) | `LLAMA_INKLING_FA_VEC_GQA`, `LLAMA_INKLING_NO_MASK`, `LLAMA_INKLING_MOE_CLAMP_NORM`, `LLAMA_INKLING_GRAPH_TRIM`, `LLAMA_INKLING_DECODE_SINKS`, `LLAMA_INKLING_KV_TYPE_FULL`, `LLAMA_GRAPH_STATS` |
| `0008-cuda-multi-stream-decode-plan.patch` | multi-stream decode plan (side streams inside the CUDA graph, fusions kept); fix for `NO_MASK` crashing the vec kernel (rev 2); **megakernel stage 1**: all attention projections of a layer in one kernel per weight type, bit-identical (rev 3); flash-attention tests at the real Inkling shapes for f16/q8_0/q4_0 (rev 3) | `GGML_CUDA_GRAPH_OPT=2`, `LLAMA_INKLING_MK_PROJ=1`, `LLAMA_INKLING_MK_DEBUG=1` |

Rev 4 fixes the CUDA pool abort ("GGML_ASSERT(ptr == ... pool_used)") with `LLAMA_INKLING_MK_PROJ=1`. Rev 2 fixed "illegal memory access" with `LLAMA_INKLING_NO_MASK=1`. Rev 3 added megakernel stage 1 and its self-test
(`test-inkling-megakernel`), and attention tests that should pinpoint the `LLAMA_INKLING_KV_TYPE_FULL=q4_0` garbage:
the q4_0 lines of the bias tests say which kernel is wrong (1 query = vec kernel, 128/512 = tile/MMA loader).
Design and estimates: [docs/MOONSHOT.md](docs/MOONSHOT.md).

## Install from ghcr.io (any machine)

```bash
git clone -b megakernel https://github.com/prototux/llamacpp-inkling2mini && cd llamacpp-inkling2mini
ln -s /path/to/your/models models
docker compose pull                     # ghcr.io/prototux/llamacpp-inkling2mini:megakernel, built by CI
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
