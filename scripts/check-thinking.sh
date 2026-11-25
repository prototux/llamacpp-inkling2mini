#!/usr/bin/env bash
# Check that thinking is off by default and comes back per request: one server, three chat requests.
#   default             no reasoning_effort: the answer must be in content, with no reasoning
#   effort high         "reasoning_effort": "high": reasoning first, then the answer
#   default with tools  a tool is offered: the model may think or call it, the request must not fail
#
# usage: scripts/check-thinking.sh
#        IMAGE=ghcr.io/prototux/llamacpp-inkling2mini:latest scripts/check-thinking.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# every run gets the code defaults plus its own settings only, not the switches of .env (compose.yaml reads
# INKLING_ENV_FILE instead of .env for the server's environment; .env still sets MODELS_DIR, IMAGE, ...)
INKLING_ENV_FILE=$(mktemp)
export INKLING_ENV_FILE
# afterwards, restart the server if it was running, with .env (not the empty test file)
was_up=$(docker compose ps -q inkling 2>/dev/null)
trap 'rm -f "$INKLING_ENV_FILE"; docker rm -f inkling-thinking >/dev/null 2>&1; [[ -z $was_up ]] || env -u INKLING_ENV_FILE docker compose up -d >/dev/null 2>&1' EXIT

port=8094
out=results/thinking-$(date +%Y%m%d-%H%M)
mkdir -p "$out"

docker compose down >/dev/null 2>&1
docker rm -f inkling-thinking >/dev/null 2>&1
docker compose run -d --rm --name inkling-thinking --no-deps -p "127.0.0.1:$port:8080" \
    -e CTX=32768 -e VISION=off -e DSPARK=off inkling >/dev/null || { echo "server did not start"; exit 1; }
for _ in $(seq 1 120); do
    curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    docker inspect inkling-thinking >/dev/null 2>&1 || { echo "server exited"; exit 1; }
    sleep 5
done

ask() {  # ask <name> <extra request fields as JSON>
    curl -sS --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
        -o "$out/$1.json" -d "$(python3 -c '
import json, sys
body = {"messages": [{"role": "user", "content": "Write a Terraform resource for an AWS S3 bucket with versioning enabled, then explain it in two sentences."}],
        "max_tokens": 6000, "temperature": 0, "cache_prompt": False}
body.update(json.loads(sys.argv[1]))
print(json.dumps(body))' "$2")"
}
ask default '{}'
ask effort-high '{"reasoning_effort": "high"}'
ask tools '{"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}'
docker logs inkling-thinking > "$out/server.log" 2>&1

python3 - "$out" <<'EOF' | tee "$out/report.txt"
import json, os, sys
out = sys.argv[1]
want = {"default": "content only", "effort-high": "reasoning, then content", "tools": "no error"}
fails = 0
print(f"{'request':<14}{'content':>9}{'reasoning':>11}{'finish':>9}   expected")
for name, expect in want.items():
    try:
        d = json.load(open(os.path.join(out, name + ".json")))
        ch = d["choices"][0]
        msg = ch["message"]
        c, r = len(msg.get("content") or ""), len(msg.get("reasoning_content") or "")
        ok = {"default": c > 0 and r == 0, "effort-high": c > 0 and r > 0, "tools": True}[name]
        print(f"{name:<14}{c:>9}{r:>11}{ch.get('finish_reason', '-'):>9}   {expect}{'' if ok else '  <-- FAIL'}")
    except Exception as e:
        ok = False
        print(f"{name:<14} ERROR {e}: {open(os.path.join(out, name + '.json')).read()[:200] if os.path.exists(os.path.join(out, name + '.json')) else ''}")
    fails += not ok
print("PASS" if fails == 0 else f"{fails} FAILED")
EOF
echo "answers and server log in $out"
