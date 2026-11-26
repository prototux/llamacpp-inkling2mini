#!/usr/bin/env bash
# Container entrypoint: turns the environment (see .env.example) into llama-server flags.
# Arguments given to the container are appended, e.g. `docker run ... image --verbose`.
set -euo pipefail

models=${MODELS:-/models}
model=$models/${MODEL_FILE:-Inkling-2-Mini-UD-Q3_K_XL.gguf}
[[ -f $model ]] || { echo "error: model $model not found: see README, 'Model files'" >&2; exit 1; }

args=(
    --host 0.0.0.0 --port 8080
    --alias "${MODEL_ALIAS:-inkling-2-mini}"
    -m "$model"
    --jinja
    # every layer on the GPU: expert offload to the CPU corrupts the looped layers (llama.cpp #32710, #32778)
    -ngl 999
    # one slot gets the whole context; more slots split it
    -np "${PARALLEL:-1}"
    -b "${BATCH:-2048}" -ub "${UBATCH:-512}"
    # thinking off unless a request sets "reasoning_effort" (--reasoning off would also block it per request)
    --reasoning-effort "${REASONING_EFFORT:-none}"
    # recommended sampling for Inkling 2 Mini; requests can override it
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
    -fa on -ctk "${KV_TYPE:-q8_0}" -ctv "${KV_TYPE:-q8_0}"
    # each checkpoint is a sliding-window + conv state snapshot in host RAM (~130 MiB)
    --ctx-checkpoints 8
    --metrics
)

# vision: cpu keeps the encoder out of VRAM (more context, slower image encoding), gpu is fast, off disables it
vision=${VISION:-cpu}
case $vision in
    gpu) margin=1536 ;;
    cpu|off) margin=768 ;;
    *) echo "error: VISION must be gpu, cpu or off" >&2; exit 1 ;;
esac
if [[ $vision != off ]]; then
    mmproj=$models/${MMPROJ_FILE:-mmproj-Inkling-2-Mini-Q8_0.gguf}
    [[ -f $mmproj ]] || { echo "error: vision projector $mmproj not found (or set VISION=off)" >&2; exit 1; }
    args+=(--mmproj "$mmproj")
    if [[ $vision == cpu ]]; then
        args+=(--no-mmproj-offload)
    fi
fi

if [[ -n ${CTX:-} ]]; then
    args+=(-c "$CTX" --fit off)
else
    # largest context that leaves FIT_MARGIN_MIB free; -ngl is set, so --fit shrinks the context instead of
    # moving layers to the CPU
    args+=(--fit on --fit-ctx "${FIT_MIN_CTX:-65536}" --fit-target "${FIT_MARGIN_MIB:-$margin}")
fi

# DSpark speculative decoding: off by default, it measured slower than plain decoding on an RTX 3090
if [[ ${DSPARK:-off} == on ]]; then
    draft=$models/${DSPARK_FILE:-dspark-Inkling-2-Mini-BF16-pr32218.gguf}
    [[ -f $draft ]] || { echo "error: DSPARK=on but $draft is missing: run scripts/convert-dspark.sh" >&2; exit 1; }
    args+=(
        --model-draft "$draft" --spec-type draft-dspark --spec-draft-n-max "${DSPARK_DRAFT_MAX:-3}"
        --n-gpu-layers-draft 99
        # the draft only proposes tokens, its cache precision changes the acceptance rate, never the output
        -ctkd q4_0 -ctvd q4_0
    )
fi

# sampling inside the GPU graph: no copy of the 200K logits to the host per token, +7-9% generation with the default
# sampling settings. llama.cpp falls back to host sampling for a request with a grammar (tools, JSON schema) or a
# reasoning budget
if [[ ${BACKEND_SAMPLING:-on} == on ]]; then
    args+=(--backend-sampling)
fi

if [[ -n ${API_KEY:-} ]]; then
    args+=(--api-key "$API_KEY")
fi

# EXTRA_ARGS: more llama-server flags, space-separated
# shellcheck disable=SC2206
args+=(${EXTRA_ARGS:-} "$@")

echo "llama-server ${args[*]}" | sed 's/--api-key [^ ]*/--api-key ***/'
exec /app/llama-server "${args[@]}"
