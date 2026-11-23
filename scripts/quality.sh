#!/usr/bin/env bash
# Price the quality cost of a setting before turning it on: perplexity on wikitext-2 for each setting, same model,
# same context. Stop the server first (docker compose down), it needs the whole GPU.
#
# usage: scripts/quality.sh                                    loop KV sharing off vs on (the default comparison)
#        scripts/quality.sh "q8_0" "-ctk q8_0 -ctv q8_0" "q8_0/q4_0" "-ctk q8_0 -ctv q4_0"
#                                                             any label / setting pairs; VAR=value words are env
# env:   CHUNKS (default 20), CTX (default 4096), MODELS_DIR (default ./models)
set -euo pipefail
cd "$(dirname "$0")/.."

models_dir=${MODELS_DIR:-./models}
model=${MODEL_FILE:-Inkling-2-Mini-UD-Q3_K_XL.gguf}
chunks=${CHUNKS:-20}
ctx=${CTX:-4096}

text=$models_dir/wiki.test.raw
if [[ ! -f $text ]]; then
    echo "== fetching wikitext-2 (4.7 MB)"
    tmp=$(mktemp -d)
    curl -fL -o "$tmp/w.zip" https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    python3 -c "import zipfile,sys; open(sys.argv[2],'wb').write(zipfile.ZipFile(sys.argv[1]).read('wikitext-2-raw/wiki.test.raw'))" "$tmp/w.zip" "$text"
    rm -rf "$tmp"
fi

if [[ $# -eq 0 ]]; then
    set -- "loop KV per pass (default)" "LLAMA_INKLING_SHARE_LOOP_KV=0" "loop KV shared" "LLAMA_INKLING_SHARE_LOOP_KV=1"
fi

mkdir -p results
out=results/quality-$(date +%Y%m%d-%H%M).log
while [[ $# -ge 2 ]]; do
    label=$1 setting=$2
    shift 2
    env_args=() flags=()
    for word in $setting; do
        if [[ $word == *=* && $word != -* ]]; then env_args+=(-e "$word"); else flags+=("$word"); fi
    done
    echo "== $label"
    docker compose run --rm --no-deps "${env_args[@]}" --entrypoint /app/llama-perplexity inkling \
        -m "/models/$model" -f /models/wiki.test.raw -c "$ctx" --chunks "$chunks" -ngl 999 -fa on \
        -ctk q8_0 -ctv q8_0 "${flags[@]}" 2>&1 \
        | grep -E 'Final estimate|llama_kv_cache: size|SHARE_LOOP_KV' | sed "s/^/[$label] /" | tee -a "$out"
done
echo "results in $out (lower perplexity is better)"
