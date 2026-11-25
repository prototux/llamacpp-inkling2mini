#!/usr/bin/env bash
# Decode speed of each megakernel stage with CUDA graphs on (the production setting), on the GPU host.
# Per configuration: one server, generation speed at a short context (2 runs of N tokens) and after a long prompt
# (LONG tokens of filler), greedy. The stages mostly remove kernel launches, which CUDA graphs already make cheap,
# so this decides which stages are worth keeping and building on.
#
# usage: scripts/mk-bench.sh
#        N=512 LONG=100000 CONFIGS="off| proj|LLAMA_INKLING_MK_PROJ=1" scripts/mk-bench.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# every run gets the code defaults plus its own settings only, not the switches of .env (compose.yaml reads
# INKLING_ENV_FILE instead of .env for the server's environment; .env still sets MODELS_DIR, IMAGE, ...)
INKLING_ENV_FILE=$(mktemp)
export INKLING_ENV_FILE
# afterwards, restart the server if it was running, with .env (not the empty test file)
was_up=$(docker compose ps -q inkling 2>/dev/null)
trap 'rm -f "$INKLING_ENV_FILE"; docker rm -f inkling-mkbench >/dev/null 2>&1; [[ -z $was_up ]] || env -u INKLING_ENV_FILE docker compose up -d >/dev/null 2>&1' EXIT

n=${N:-512}
long=${LONG:-64000}
ctx=${TEST_CTX:-131072}
port=8095
out=results/mkbench-$(date +%Y%m%d-%H%M)
mkdir -p "$out"

ms="GGML_CUDA_GRAPH_OPT=2,GGML_CUDA_GRAPH_OPT_STREAMS=3"
all="LLAMA_INKLING_MK_PROJ=1,LLAMA_INKLING_MK_WO=1,LLAMA_INKLING_MK_MOE=1"
default_configs=(
    "off|"
    "proj|LLAMA_INKLING_MK_PROJ=1"
    "wo|LLAMA_INKLING_MK_WO=1"
    "moe|LLAMA_INKLING_MK_MOE=1"
    "glue|LLAMA_INKLING_MK_GLUE=1"
    "glue2|LLAMA_INKLING_MK_GLUE=2"
    "proj-wo-moe|$all"
    "all-glue|$all,LLAMA_INKLING_MK_GLUE=1"
    "all-glue2|$all,LLAMA_INKLING_MK_GLUE=2"
    "streams|$ms"
    "streams-all-glue|$ms,$all,LLAMA_INKLING_MK_GLUE=1"
)
if [[ -n ${CONFIGS:-} ]]; then read -r -a configs <<< "$CONFIGS"; else configs=("${default_configs[@]}"); fi

speed() {  # speed <prompt file> <tokens>: generation tok/s and prompt tokens of one greedy request
    python3 - "$1" "$2" "$port" <<'PY'
import json, sys, urllib.request
body = {"messages": [{"role": "user", "content": open(sys.argv[1]).read()}], "max_tokens": int(sys.argv[2]),
        "temperature": 0, "cache_prompt": False, "ignore_eos": True}
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[3]}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
try:
    t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
    print(f"{t['predicted_per_second']:.1f} {t['prompt_n']}")
except Exception as e:
    print(f"error {type(e).__name__}")
PY
}

echo "Write a Python module that implements a small HTTP router with tests." > "$out/short.txt"
python3 -c '
import random, sys
rng = random.Random(1)
words = ["the", "river", "stone", "quiet", "market", "lantern", "north", "garden", "copper", "window"]
print(" ".join(rng.choice(words) for _ in range(int(int(sys.argv[1]) / 1.3))) + "\nSummarize the text above in detail.")' "$long" > "$out/long.txt"

docker compose down >/dev/null 2>&1
printf '%-18s %10s %10s %14s\n' config short-1 short-2 "long (tokens)" | tee "$out/report.txt"
for c in "${configs[@]}"; do
    name=${c%%|*}
    envw=${c#*|}
    ef=()
    for w in ${envw//,/ }; do ef+=(-e "$w"); done
    docker rm -f inkling-mkbench >/dev/null 2>&1
    if ! docker compose run -d --rm --name inkling-mkbench --no-deps -p "127.0.0.1:$port:8080" \
            -e "CTX=$ctx" -e VISION=off -e DSPARK=off "${ef[@]}" inkling >/dev/null; then
        echo "$name: server did not start" | tee -a "$out/report.txt"; continue
    fi
    ready=0
    for _ in $(seq 1 120); do
        if curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
        docker inspect inkling-mkbench >/dev/null 2>&1 || break
        sleep 5
    done
    if [[ $ready -eq 0 ]]; then echo "$name: server not ready" | tee -a "$out/report.txt"; docker logs inkling-mkbench > "$out/$name.log" 2>&1; continue; fi
    s1=$(speed "$out/short.txt" "$n"); s2=$(speed "$out/short.txt" "$n"); l=$(speed "$out/long.txt" "$n")
    printf '%-18s %10s %10s %14s\n' "$name" "${s1%% *}" "${s2%% *}" "${l%% *} (${l##* })" | tee -a "$out/report.txt"
    docker logs inkling-mkbench > "$out/$name.log" 2>&1
    docker rm -f inkling-mkbench >/dev/null 2>&1
done
echo "report and logs in $out"
