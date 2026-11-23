#!/usr/bin/env bash
# Test the "megakernel" test build on the GPU host, in one go:
#   1. build the image from docker/Dockerfile (patches 0001-0008)
#   2. GPU correctness: attention kernels, fused kernels, the multi-stream plan, tiny-model CUDA-vs-CPU runs
#   3. greedy identity: the same prompt at temperature 0 with each optimization on must give the same text
#   4. speed: time to first token and decode speed per configuration
#   5. maximum context that --fit picks per memory setting
# Everything lands in results/megakernel-<date>/, with summary.txt at the end. Keeps going after a failure.
#
# usage: scripts/test-megakernel.sh            (~1-2 h, mostly the build and the benchmarks)
#        SKIP_BUILD=1 scripts/test-megakernel.sh
#        STEPS="kernels identity" scripts/test-megakernel.sh      (any of: kernels identity bench context)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

out=results/megakernel-$(date +%Y%m%d-%H%M)
mkdir -p "$out"
summary=$out/summary.txt
steps=${STEPS:-kernels identity bench context}
port=8090                  # the test server; leave the normal one on 8080 stopped (the GPU is needed whole)
ctx=${TEST_CTX:-131072}    # fixed context for identity and speed runs, so every configuration is comparable

note() { echo "$*" | tee -a "$summary"; }
has_step() { [[ " $steps " == *" $1 "* ]]; }

# configurations: name | environment
configs=(
    "baseline|LLAMA_INKLING_MOE_CLAMP_NORM=0 LLAMA_INKLING_GRAPH_TRIM=0 LLAMA_INKLING_DECODE_SINKS=0"
    "defaults|"
    "no-mask|LLAMA_INKLING_NO_MASK=1"
    "vec-gqa|LLAMA_INKLING_FA_VEC_GQA=1"
    "multi-stream|GGML_CUDA_GRAPH_OPT=2"
    "mk-proj|LLAMA_INKLING_MK_PROJ=1"
    "mk-proj-nographs|LLAMA_INKLING_MK_PROJ=1 GGML_CUDA_DISABLE_GRAPHS=1"
    "mk-wo|LLAMA_INKLING_MK_WO=1"
    "mk-moe|LLAMA_INKLING_MK_MOE=1"
    "all-on|LLAMA_INKLING_NO_MASK=1 LLAMA_INKLING_FA_VEC_GQA=1 GGML_CUDA_GRAPH_OPT=2 LLAMA_INKLING_MK_WO=1 LLAMA_INKLING_MK_MOE=1"
)

env_flags() { local f=() w; for w in $1; do f+=(-e "$w"); done; printf '%s\n' "${f[@]}"; }

run_tool() {  # run_tool <log> <env words> <binary> [args...]
    local log=$1 envw=$2 bin=$3; shift 3
    mapfile -t ef < <(env_flags "$envw")
    docker compose run --rm --no-deps "${ef[@]}" --entrypoint "$bin" inkling "$@" > "$log" 2>&1
}

start_server() {  # start_server <env words>; waits until healthy
    local envw="$*"
    mapfile -t ef < <(env_flags "$envw")
    docker rm -f inkling-test >/dev/null 2>&1
    docker compose run -d --rm --name inkling-test --no-deps -p "127.0.0.1:$port:8080" "${ef[@]}" inkling >/dev/null || return 1
    for _ in $(seq 1 180); do
        curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
        docker inspect inkling-test >/dev/null 2>&1 || return 1
        sleep 5
    done
    return 1
}
stop_server() { docker rm -f inkling-test >/dev/null 2>&1; }

note "megakernel test build, $(date)"
note "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)"
docker compose down >/dev/null 2>&1

if [[ -z ${SKIP_BUILD:-} ]]; then
    note "== build (first time 30-60 min)"
    if ! docker compose build inkling > "$out/build.log" 2>&1; then
        note "BUILD FAILED, see $out/build.log"; exit 1
    fi
fi

if has_step kernels; then
    note "== kernels (CUDA vs exact reference / CPU)"
    for v in "default|" "vec-gqa|LLAMA_INKLING_FA_VEC_GQA=1" "f16-copy|LLAMA_INKLING_FA_Q8_0=0" "no-mma|LLAMA_INKLING_FA_MMA=0"; do
        name=${v%%|*}; envw=${v#*|}
        if run_tool "$out/bias-$name.log" "$envw" /app/test-flash-attn-indexed-bias && grep -q CUDA0 "$out/bias-$name.log"; then note "PASS bias test [$name]"; else note "FAIL bias test [$name] ($out/bias-$name.log)"; fi
    done
    for op in FLASH_ATTN_EXT "INKLING2_.*"; do
        if run_tool "$out/backend-ops-${op%%_*}.log" "" /app/test-backend-ops -b CUDA0 -o "$op"; then note "PASS test-backend-ops $op"; else note "FAIL test-backend-ops $op ($out/backend-ops-${op%%_*}.log)"; fi
    done
    # megakernel stage self-test: fused vs reference on random weights at the real Inkling shapes, with timings
    if run_tool "$out/megakernel.log" "" /app/test-inkling-megakernel; then
        note "$(grep -E '^(proj|proj2|steps|wo|moe)|PERF FAIL|slower than the reference' "$out/megakernel.log" | sed 's/^/  /')"
        if grep -q FAIL "$out/megakernel.log"; then note "FAIL megakernel self-test ($out/megakernel.log)"; else note "PASS megakernel self-test"; fi
    else
        note "FAIL megakernel self-test crashed: rerun with debug ($out/megakernel-debug.log)"
        run_tool "$out/megakernel-debug.log" "LLAMA_INKLING_MK_DEBUG=1 CUDA_LAUNCH_BLOCKING=1" /app/test-inkling-megakernel
    fi
    for s in 3 1 7; do
        if run_tool "$out/streams-$s.log" "GGML_CUDA_GRAPH_OPT=2 GGML_CUDA_GRAPH_OPT_DEBUG=1 GGML_CUDA_GRAPH_OPT_STREAMS=$s" /app/test-sched-streams && grep -q OK "$out/streams-$s.log"; then note "PASS multi-stream plan, $s streams"; else note "FAIL multi-stream plan, $s streams ($out/streams-$s.log)"; fi
    done
    for v in "default|" "multi-stream|GGML_CUDA_GRAPH_OPT=2" "no-mask|LLAMA_INKLING_NO_MASK=1" "mk-proj|LLAMA_INKLING_MK_PROJ=1" "all-on|LLAMA_INKLING_NO_MASK=1 LLAMA_INKLING_FA_VEC_GQA=1 GGML_CUDA_GRAPH_OPT=2 LLAMA_INKLING_MK_PROJ=1"; do
        name=${v%%|*}; envw=${v#*|}
        if run_tool "$out/archs-$name.log" "$envw" /app/test-llama-archs -a 'inkling2.*' && grep -q 'passed' "$out/archs-$name.log"; then note "PASS tiny-model CUDA vs CPU [$name]"; else note "FAIL tiny-model CUDA vs CPU [$name] ($out/archs-$name.log)"; fi
    done
fi

if has_step identity; then
    note "== greedy identity (temperature 0, 256 tokens, CTX=$ctx)"
    prompt='Write a Python function that merges overlapping intervals, with a docstring and three tests.'
    for c in "${configs[@]}"; do
        name=${c%%|*}; envw=${c#*|}
        if ! start_server "CTX=$ctx VISION=off DSPARK=off $envw"; then
            note "FAIL [$name]: server did not start"; docker logs inkling-test > "$out/server-$name.log" 2>&1; stop_server; continue
        fi
        curl -fs "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
            -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": 256, "temperature": 0, "seed": 1, "cache_prompt": False}))' "$prompt")" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])' > "$out/greedy-$name.txt"
        docker logs inkling-test > "$out/server-$name.log" 2>&1
        stop_server
        if [[ $name == baseline ]]; then
            note "baseline: $(wc -c < "$out/greedy-baseline.txt") bytes"
        elif cmp -s "$out/greedy-baseline.txt" "$out/greedy-$name.txt"; then
            note "IDENTICAL [$name]"
        else
            note "DIFFERENT [$name] vs baseline: diff $out/greedy-baseline.txt $out/greedy-$name.txt"
        fi
    done
fi

if has_step bench; then
    note "== speed (CTX=$ctx; prompts of 1K, 8K, 32K, 100K tokens)"
    for c in "${configs[@]}"; do
        name=${c%%|*}; envw=${c#*|}
        if ! start_server "CTX=$ctx VISION=off DSPARK=off $envw"; then note "FAIL [$name]: server did not start"; stop_server; continue; fi
        python3 scripts/bench.py --url "http://127.0.0.1:$port" --label "$name" --lengths 1000,8000,32000,100000 \
            --max-ctx-fraction 0 --out "$out/bench-$name.json" > "$out/bench-$name.log" 2>&1
        stop_server
        note "$name: $(grep -E 'decode code|decode prose' "$out/bench-$name.log" | tr '\n' ' ')"
    done
    python3 scripts/compare.py "$out"/bench-baseline.json "$out"/bench-defaults.json "$out"/bench-multi-stream.json "$out"/bench-mk-wo.json "$out"/bench-mk-moe.json "$out"/bench-all-on.json \
        > "$out/compare.txt" 2>&1 && tee -a "$summary" < "$out/compare.txt"
fi

# MK_PROJ run log: the fused runs of a real server, for the bug under investigation (DSpark draft graphs included)
if has_step identity; then
    if start_server "CTX=$ctx VISION=off DSPARK=${DSPARK:-off} LLAMA_INKLING_MK_PROJ=1 LLAMA_INKLING_MK_DEBUG=1"; then
        curl -fs "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
            -d '{"prompt": "Count from 1 to 20.", "n_predict": 64, "temperature": 0}' > /dev/null
        docker logs inkling-test > "$out/mk-proj-runs.log" 2>&1
        note "MK_PROJ run log: $out/mk-proj-runs.log"
    fi
    stop_server
fi

if has_step context; then
    note "== maximum context picked by --fit (vision on CPU, DSpark off)"
    for v in "default|" "no-mask|LLAMA_INKLING_NO_MASK=1" "q4-full-kv|LLAMA_INKLING_KV_TYPE_FULL=q4_0" "shared-loop-kv|LLAMA_INKLING_SHARE_LOOP_KV=1" "no-mask+shared|LLAMA_INKLING_NO_MASK=1 LLAMA_INKLING_SHARE_LOOP_KV=1"; do
        name=${v%%|*}; envw=${v#*|}
        if start_server "DSPARK=off $envw"; then
            n=$(curl -fs "http://127.0.0.1:$port/props" | python3 -c 'import json,sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])')
            note "n_ctx [$name]: $n"
        else
            note "FAIL [$name]: server did not start (see $out/ctx-$name.log)"
        fi
        docker logs inkling-test > "$out/ctx-$name.log" 2>&1
        stop_server
    done
fi

note "== done: $summary"
echo
echo "Send me $summary (and the logs of any FAIL / DIFFERENT line)."
