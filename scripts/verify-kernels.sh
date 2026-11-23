#!/usr/bin/env bash
# Check the CUDA kernels of this build against the CPU reference on your GPU, without loading the model:
#   - test-flash-attn-indexed-bias: Inkling's relative-position bias in every attention path (decode, small
#     batches, prefill; F16 and q8_0 caches; index and position modes) against an exact expectation
#   - test-backend-ops: flash attention and the fused Inkling 2 kernels, CUDA vs CPU
# usage: scripts/verify-kernels.sh      (uses the image of compose.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
    docker compose run --rm --no-deps --entrypoint "$1" inkling "${@:2}"
}

status=0
echo "== relative-position bias, all kernel paths"
log=$(mktemp)
if run /app/test-flash-attn-indexed-bias | tee "$log" | grep CUDA0 | tail -30 && grep -q CUDA0 "$log"; then
    :
else
    echo "FAILED: test-flash-attn-indexed-bias (or no CUDA device was tested)" >&2
    status=1
fi
rm -f "$log"

echo
echo "== test-backend-ops: flash attention and Inkling 2 fused ops, CUDA vs CPU"
for op in FLASH_ATTN_EXT "INKLING2_.*"; do
    if ! run /app/test-backend-ops -b CUDA0 -o "$op" 2>&1 | grep -E 'OK|FAIL|passed|not supported' | tail -5; then
        echo "FAILED: test-backend-ops -o $op" >&2
        status=1
    fi
done

if [[ $status -eq 0 ]]; then
    echo "kernels OK"
else
    echo "kernel checks FAILED: see the README, 'Kernel switches'"
fi
exit $status
