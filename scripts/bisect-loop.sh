#!/usr/bin/env bash
# Find which setting makes long generations degrade into repetition.
# Each configuration generates the same long code answer at temperature 0; the report gives, per configuration,
# where its text first differs from the all-off baseline and how repetitive its tail is (loop score: share of
# repeated 8-word sequences in the last 400 words; healthy text is near 0, a loop near 1).
#
# usage: scripts/bisect-loop.sh                 (~2-4 min per configuration)
#        N=3000 scripts/bisect-loop.sh          (longer generations for a slower onset)
#        CONFIGS="mine|$(grep -E '^[A-Z_]+=' .env | paste -sd,)" scripts/bisect-loop.sh   (configs space-separated, variables comma-separated)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# every run gets the code defaults plus its own settings only, not the switches of .env (compose.yaml reads
# INKLING_ENV_FILE instead of .env for the server's environment; .env still sets MODELS_DIR, IMAGE, ...)
INKLING_ENV_FILE=$(mktemp)
export INKLING_ENV_FILE
# afterwards, restart the server if it was running, with .env (not the empty test file)
was_up=$(docker compose ps -q inkling 2>/dev/null)
trap 'rm -f "$INKLING_ENV_FILE"; [[ -z $was_up ]] || env -u INKLING_ENV_FILE docker compose up -d >/dev/null 2>&1' EXIT

n=${N:-1500}
ctx=${TEST_CTX:-32768}
port=8091
out=results/bisect-$(date +%Y%m%d-%H%M)
mkdir -p "$out"

base="LLAMA_INKLING_MOE_CLAMP_NORM=0 LLAMA_INKLING_GRAPH_TRIM=0 LLAMA_INKLING_DECODE_SINKS=0"
default_configs=(
    "baseline|$base"
    "defaults|"
    "share-loop-kv|LLAMA_INKLING_SHARE_LOOP_KV=1"
    "no-mask|LLAMA_INKLING_NO_MASK=1"
    "vec-gqa|LLAMA_INKLING_FA_VEC_GQA=1"
    "multi-stream|GGML_CUDA_GRAPH_OPT=2"
)
if [[ -n ${CONFIGS:-} ]]; then
    read -r -a configs <<< "baseline|${base// /,} ${CONFIGS}"
else
    configs=("${default_configs[@]}")
fi

prompt='Write a complete Terraform configuration for an AWS VPC with two public and two private subnets across two availability zones, an internet gateway, a NAT gateway, route tables, security groups for a web tier and a database tier, an RDS PostgreSQL instance and an autoscaling group of EC2 instances behind an application load balancer. Write every resource in full, no placeholders.'

run() {  # run <name> <env words, comma or space separated>
    local name=$1 envw=${2//,/ } ef=() w
    for w in $envw; do ef+=(-e "$w"); done
    docker rm -f inkling-bisect >/dev/null 2>&1
    if ! docker compose run -d --rm --name inkling-bisect --no-deps -p "127.0.0.1:$port:8080" \
            -e "CTX=$ctx" -e VISION=off -e DSPARK=off "${ef[@]}" inkling >/dev/null; then
        echo "$name: server did not start"; return
    fi
    local ready=0
    for _ in $(seq 1 120); do
        if curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
        docker inspect inkling-bisect >/dev/null 2>&1 || break   # the server exited
        sleep 5
    done
    if [[ $ready -eq 0 ]]; then
        echo "server not ready" > "$out/$name.err"
    else
        # keep the raw answer: an HTTP error or an empty message is reported instead of being read as text
        curl -sS --max-time 3600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
            -o "$out/$name.json" -w '%{http_code}' -d "$(python3 -c '
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": int(sys.argv[2]),
                  "temperature": 0, "seed": 1, "cache_prompt": False}))' "$prompt" "$n")" > "$out/$name.http" 2> "$out/$name.err"
        python3 - "$out/$name" <<'PY'
import json, sys
base = sys.argv[1]
try:
    msg = json.load(open(base + ".json"))["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text:
        text = msg.get("reasoning_content") or ""
    open(base + ".txt", "w").write(text)
    if not text:
        open(base + ".err", "a").write("empty answer\n")
except Exception as e:
    open(base + ".err", "a").write(f"no answer: HTTP {open(base + '.http').read().strip()}: {e}: {open(base + '.json').read()[:300] if __import__('os').path.exists(base + '.json') else ''}\n")
PY
    fi
    docker logs inkling-bisect > "$out/$name.log" 2>&1
    docker rm -f inkling-bisect >/dev/null 2>&1
}

docker compose down >/dev/null 2>&1
for c in "${configs[@]}"; do
    name=${c%%|*}
    echo "== $name"
    run "$name" "${c#*|}"
done

python3 - "$out" <<'EOF' | tee "$out/report.txt"
import os, sys
out = sys.argv[1]
def loop_score(text):
    w = text.split()[-400:]
    grams = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)
base = open(os.path.join(out, "baseline.txt")).read() if os.path.exists(os.path.join(out, "baseline.txt")) else ""
if not base.strip():
    print("WARNING: the baseline produced no text, so the 'first diff' column is meaningless")
print(f"{'config':<20}{'chars':>7}{'first diff vs baseline':>24}{'loop score':>12}")
names = sorted({f.rsplit(".", 1)[0] for f in os.listdir(out) if f.endswith((".txt", ".err", ".log")) and f != "report.txt"})
for name in names:
    err = open(os.path.join(out, name + ".err")).read().strip() if os.path.exists(os.path.join(out, name + ".err")) else ""
    t = open(os.path.join(out, name + ".txt")).read() if os.path.exists(os.path.join(out, name + ".txt")) else ""
    if err or not t.strip():
        log = os.path.join(out, name + ".log")
        tail = open(log).read().strip().splitlines()[-3:] if os.path.exists(log) else []
        print(f"{name:<20} ERROR: {err or 'no text'} | server log: {' / '.join(tail)[:300]}")
        continue
    d = next((i for i, (a, b) in enumerate(zip(base, t)) if a != b), min(len(base), len(t)))
    same = "identical" if t == base else f"char {d}"
    flag = "  <-- LOOP" if loop_score(t) > 0.3 else ""
    print(f"{name:<20}{len(t):>7}{same:>24}{loop_score(t):>12.2f}{flag}")
EOF
echo "texts and logs in $out"
