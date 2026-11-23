#!/usr/bin/env bash
# The two megakernel diagnostics in one go, on the GPU host (CUDA graphs off, which both need):
#   1. MK_CHECK: every fused stage (proj, wo, moe, glue) also runs into a shadow buffer next to the unfused nodes,
#      and the two are compared bit for bit after each decode step; a MISMATCH line names the stage, tensor and
#      first differing element. If the loop happens with 0 mismatches, it is not in a fused kernel.
#   2. MK_PROFILE: GPU time per kernel class of one decode step and host wall time, all megakernel stages off and
#      then with the glue stage, to see where the ~11-13 ms per token go and whether launches dominate.
# Results in results/diagnose-<date>/, summary in summary.txt.
#
# usage: scripts/mk-diagnose.sh
#        N=2000 scripts/mk-diagnose.sh       (generate longer in the check run, for a slow-onset loop)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# every run gets the code defaults plus its own settings only, not the switches of .env (compose.yaml reads
# INKLING_ENV_FILE instead of .env for the server's environment; .env still sets MODELS_DIR, IMAGE, ...)
INKLING_ENV_FILE=$(mktemp)
export INKLING_ENV_FILE
trap 'rm -f "$INKLING_ENV_FILE"' EXIT

n=${N:-1200}
port=8093
out=results/diagnose-$(date +%Y%m%d-%H%M)
mkdir -p "$out"
summary=$out/summary.txt
prompt='Write a complete Terraform configuration for an AWS VPC with two public and two private subnets, an internet gateway, a NAT gateway, route tables, security groups, an RDS PostgreSQL instance and an autoscaling group behind a load balancer. Write every resource in full.'

serve() {  # serve <name> <env words>: start, run one long greedy generation, save the log, stop
    local name=$1 ef=() w
    for w in $2; do ef+=(-e "$w"); done
    docker rm -f inkling-diag >/dev/null 2>&1
    docker compose run -d --rm --name inkling-diag --no-deps -p "127.0.0.1:$port:8080" \
        -e CTX=32768 -e VISION=off -e DSPARK=off -e GGML_CUDA_DISABLE_GRAPHS=1 "${ef[@]}" inkling >/dev/null || { echo "$name: server did not start" | tee -a "$summary"; return; }
    local ok=0
    for _ in $(seq 1 120); do
        if curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ok=1; break; fi
        docker inspect inkling-diag >/dev/null 2>&1 || break
        sleep 5
    done
    if [[ $ok -eq 1 ]]; then
        curl -sS --max-time 3600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -o "$out/$name.json" \
            -d "$(python3 -c 'import json,sys; print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": int(sys.argv[2]), "temperature": 0, "cache_prompt": False}))' "$prompt" "$n")"
    else
        echo "$name: server not ready" | tee -a "$summary"
    fi
    docker logs inkling-diag > "$out/$name.log" 2>&1
    docker rm -f inkling-diag >/dev/null 2>&1
}

docker compose down >/dev/null 2>&1
echo "megakernel diagnostics, $(date)" > "$summary"

echo "== 1. MK_CHECK (all stages on, compared against the unfused nodes every step)" | tee -a "$summary"
serve check "LLAMA_INKLING_MK_CHECK=1 LLAMA_INKLING_MK_PROJ=1 LLAMA_INKLING_MK_GLUE=1 LLAMA_INKLING_MK_MOE=1 LLAMA_INKLING_MK_WO=1 LLAMA_INKLING_MK_DEBUG=1"
grep -E 'mk_check: [0-9]+ evaluations' "$out/check.log" | tail -1 | tee -a "$summary"
grep -E 'MISMATCH' "$out/check.log" | head -20 | tee -a "$summary"
grep -c 'mk_glue: ' "$out/check.log" | sed 's/^/glue instances logged: /' | tee -a "$summary"

for cfg in "profile-off|LLAMA_INKLING_MK_PROFILE=1" "profile-glue|LLAMA_INKLING_MK_PROFILE=1 LLAMA_INKLING_MK_GLUE=1" "profile-glue2|LLAMA_INKLING_MK_PROFILE=1 LLAMA_INKLING_MK_GLUE=2 LLAMA_INKLING_MK_MOE=1 LLAMA_INKLING_MK_WO=1"; do
    name=${cfg%%|*}
    echo "== 2. $name" | tee -a "$summary"
    serve "$name" "${cfg#*|}"
    # the last complete profile block of the run
    awk '/mk_profile: evaluation/{blk=""} /mk_profile:/{blk=blk $0 "\n"} END{printf "%s", blk}' "$out/$name.log" | tee -a "$summary"
done

echo "== done: $summary"
echo "Send me $summary (and check.log if it shows MISMATCH lines)."
