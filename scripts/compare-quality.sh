#!/usr/bin/env bash
# Find which build or setting changed the model's output distribution or answer quality.
#   1. probabilities: the top-10 token probabilities over the first 128 tokens of one prompt, compared with the
#      first configuration (max absolute difference and mean KL divergence). Greedy checks cannot see a change
#      that keeps the most likely token but reshapes the distribution; sampling at temperature 1 does.
#   2. completeness: the Terraform prompt sampled at the server's settings (temperature 1.0), 3 seeds per
#      configuration: number of resources and how many are syntactically complete (balanced braces).
#
# usage: scripts/compare-quality.sh
#        CONFIGS="mine||LLAMA_INKLING_GRAPH_TRIM=0,LLAMA_INKLING_DECODE_SINKS=0" scripts/compare-quality.sh
# A configuration is name|image|env (image empty = the compose image); configurations space-separated,
# variables comma-separated. Defaults compare published builds and the default-on switches.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

img=ghcr.io/prototux/llamacpp-inkling2mini
default_configs=(
    "build2|$img:sha-7ad4ddf|"
    "latest|$img:sha-2c1e65e|"
    "latest-m3m4-off|$img:sha-2c1e65e|LLAMA_INKLING_MOE_CLAMP_NORM=0,LLAMA_INKLING_GRAPH_TRIM=0,LLAMA_INKLING_DECODE_SINKS=0"
    "latest-trim-off|$img:sha-2c1e65e|LLAMA_INKLING_GRAPH_TRIM=0"
    "latest-sinks-off|$img:sha-2c1e65e|LLAMA_INKLING_DECODE_SINKS=0"
    "latest-clamp-off|$img:sha-2c1e65e|LLAMA_INKLING_MOE_CLAMP_NORM=0"
    "megakernel|$img:megakernel|"
)
if [[ -n ${CONFIGS:-} ]]; then read -r -a configs <<< "$CONFIGS"; else configs=("${default_configs[@]}"); fi

ctx=${TEST_CTX:-32768}
port=8092
seeds=${SEEDS:-1 2 3}
out=results/quality-$(date +%Y%m%d-%H%M)
mkdir -p "$out"
tf='Write a complete Terraform configuration for an AWS VPC with two public and two private subnets across two availability zones, an internet gateway, a NAT gateway, route tables, security groups for a web tier and a database tier, an RDS PostgreSQL instance and an autoscaling group of EC2 instances behind an application load balancer. Write every resource in full, no placeholders.'
probe='The three most important things to know about Terraform state files are'

start() {  # start <image> <env words comma-separated>
    local image=$1 ef=() w
    for w in ${2//,/ }; do ef+=(-e "$w"); done
    docker rm -f inkling-quality >/dev/null 2>&1
    IMAGE=${image:-} docker compose run -d --rm --name inkling-quality --no-deps -p "127.0.0.1:$port:8080" \
        -e "CTX=$ctx" -e VISION=off -e DSPARK=off "${ef[@]}" inkling >/dev/null || return 1
    for _ in $(seq 1 120); do
        curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
        docker inspect inkling-quality >/dev/null 2>&1 || return 1
        sleep 5
    done
    return 1
}

docker compose down >/dev/null 2>&1
for c in "${configs[@]}"; do
    IFS='|' read -r name image envw <<< "$c"
    echo "== $name ($image $envw)"
    [[ -n $image ]] && docker pull -q "$image" >/dev/null
    if ! start "$image" "$envw"; then echo "server did not start" > "$out/$name.err"; docker rm -f inkling-quality >/dev/null 2>&1; continue; fi
    curl -sS "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' -o "$out/$name.probs.json" \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": 128, "temperature": 0, "n_probs": 10, "cache_prompt": False}))' "$probe")"
    for s in $seeds; do
        curl -sS "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -o "$out/$name.tf$s.json" \
            -d "$(python3 -c 'import json,sys; print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": 4096, "seed": int(sys.argv[2]), "cache_prompt": False}))' "$tf" "$s")"
    done
    docker logs inkling-quality > "$out/$name.log" 2>&1
    docker rm -f inkling-quality >/dev/null 2>&1
done

python3 - "$out" <<'EOF' | tee "$out/report.txt"
import json, math, os, re, sys
out = sys.argv[1]
names = []
for f in sorted(os.listdir(out), key=lambda f: os.path.getmtime(os.path.join(out, f))):
    m = re.match(r"(.+)\.probs\.json$", f)
    if m and m.group(1) not in names:
        names.append(m.group(1))

def probs(name):
    try:
        d = json.load(open(os.path.join(out, name + ".probs.json")))
        res = []
        for step in d["completion_probabilities"]:
            tops = step.get("top_probs") or step.get("top_logprobs") or step.get("probs") or []
            res.append({(t.get("token") or t.get("tok_str")): (t["prob"] if "prob" in t else math.exp(t.get("logprob", -99)))
                        for t in tops})
        return res
    except Exception as e:
        return None

def tf_stats(name, path):
    try:
        msg = json.load(open(path))["choices"][0]["message"]
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    except Exception:
        return None
    blocks = [m.start() for m in re.finditer(r'^\s*resource\s+"', text, re.M)]
    complete = 0
    for i, s in enumerate(blocks):
        e = blocks[i + 1] if i + 1 < len(blocks) else len(text)
        body = text[s:e]
        complete += body.count("{") > 0 and body.count("{") == body.count("}")
    return len(blocks), complete, len(text), "..." in text

ref = probs(names[0]) if names else None
print(f"{'config':<20}{'max |dp|':>10}{'mean KL':>10}   terraform runs: resources/complete (chars)")
for n in names:
    p = probs(n)
    if p is None or ref is None:
        dp = kl = float("nan")
    else:
        dps, kls = [], []
        for a, b in zip(ref, p):
            for tok, pa in a.items():
                pb = b.get(tok, 1e-6)
                dps.append(abs(pa - pb))
                if pa > 0:
                    kls.append(pa * math.log(pa / max(pb, 1e-6)))
        dp = max(dps) if dps else float("nan")
        kl = sum(kls) / max(len(ref), 1)
    runs = []
    for f in sorted(os.listdir(out)):
        if f.startswith(n + ".tf") and f.endswith(".json"):
            st = tf_stats(n, os.path.join(out, f))
            runs.append("err" if st is None else f"{st[0]}/{st[1]} ({st[2]}{', ...' if st[3] else ''})")
    print(f"{n:<20}{dp:>10.4f}{kl:>10.4f}   {'  '.join(runs)}")
print(f"reference for the probabilities: {names[0] if names else '-'}")
EOF
echo "raw answers and logs in $out"
