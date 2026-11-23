#!/bin/sh
# Download the model files into /models, resuming partial downloads and checking each file's SHA-256.
# Runs in the compose "download" service (busybox sh); the same URLs are listed in the README for manual or
# offline installs.
set -eu

dir=/models
hf=https://huggingface.co

# name | url | sha256
files="
Inkling-2-Mini-UD-Q3_K_XL.gguf|$hf/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file|e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639
mmproj-Inkling-2-Mini-Q8_0.gguf|$hf/ggml-org/Inkling-2-Mini-GGUF/resolve/main/mmproj-Inkling-2-Mini-Q8_0.gguf|233287272ec5ac39b4168677aba54a4c024f202410f5629fac1830e1059f8abb
"

echo "$files" | while IFS='|' read -r name url sha; do
    [ -n "$name" ] || continue
    out=$dir/$name
    if [ -f "$out" ]; then
        echo "ok      $name"
        continue
    fi
    echo "fetch   $name"
    curl -fL --retry 5 --retry-delay 5 -C - -o "$out.part" "$url"
    echo "verify  $name"
    if ! echo "$sha  $out.part" | sha256sum -c -s; then
        echo "error: $name does not match its SHA-256, removed the partial file" >&2
        rm -f "$out.part"
        exit 1
    fi
    mv "$out.part" "$out"
done
echo "models ready in $dir"
