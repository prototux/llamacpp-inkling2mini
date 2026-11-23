#!/usr/bin/env bash
# Optional: build the DSpark draft model for speculative decoding (DSPARK=on).
#
# Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf predates the DSpark code in this build and fails with
# "tensor 'markov_w1.weight' has wrong shape". This converts the draft from the official checkpoint with the
# converter of the patched llama.cpp, in a 5 GiB-capped Python container. It downloads ~1.6 GB: the draft and two
# tensors of the target model, not its 19.5 GB weights file.
#
# usage: scripts/convert-dspark.sh [models-dir]        (default ./models)
# env:   OUTTYPE=bf16 (default) or q8_0 (q8_0 quantizes the Markov head and lowers the acceptance rate)
set -euo pipefail
cd "$(dirname "$0")/.."

models_dir=$(realpath "${1:-${MODELS_DIR:-./models}}")
outtype=${OUTTYPE:-bf16}
out=dspark-Inkling-2-Mini-${outtype^^}-pr32218.gguf
commit=$(sed -n 's/^ARG LLAMA_CPP_COMMIT=//p' docker/Dockerfile)

if [[ -f $models_dir/$out ]]; then
    echo "$models_dir/$out already exists; delete it to convert again"
    exit 0
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
echo "== llama.cpp $commit + patches"
git -C "$work" init -q src
git -C "$work/src" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$commit"
git -C "$work/src" checkout -q FETCH_HEAD
for p in patches/*.patch; do
    git -C "$work/src" apply --whitespace=nowarn "$(realpath "$p")"
done

echo "== convert ($outtype)"
docker run --rm --memory=5g --memory-swap=5g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$work/src:/src:ro" -v "$(pwd)/scripts:/scripts:ro" -v "$models_dir:/models" \
    docker.io/library/python:3.12-slim bash -c "
    set -euo pipefail
    python -m venv /tmp/venv && . /tmp/venv/bin/activate
    pip install -q --no-cache-dir -r /src/requirements/requirements-convert_hf_to_gguf.txt
    pip install -q --no-cache-dir /src/gguf-py
    python /scripts/dspark_checkpoint.py /models/.dspark-checkpoint
    python /src/convert_hf_to_gguf.py /models/.dspark-checkpoint --dspark --outtype $outtype --outfile /models/$out.part
    mv /models/$out.part /models/$out
"
rm -rf "$models_dir/.dspark-checkpoint"
echo "converted: $models_dir/$out (set DSPARK=on in .env to use it)"
