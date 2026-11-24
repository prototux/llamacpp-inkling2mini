# syntax=docker/dockerfile:1
#
# llama.cpp for Inkling 2 Mini on NVIDIA GPUs: upstream llama.cpp at a pinned commit + the patches in patches/.
#
#   docker build -f docker/Dockerfile -t llamacpp-inkling2mini .
#   docker build -f docker/Dockerfile --build-arg CUDA_ARCH="86;89" --build-arg BUILD_JOBS=8 .
#
# The image needs NVIDIA driver >= 570 on the host (CUDA 12.8 runtime).

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG NODE_VERSION=24

# ---- llama.cpp sources: pinned upstream commit + our patches, applied in order ----
FROM docker.io/library/alpine:3 AS source
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
# ggml-org/llama.cpp master of 2026-11-22; bump it together with the patches
ARG LLAMA_CPP_COMMIT=c74c71a6f7a0b06df30f979520e5d9a86a4d60ad
RUN apk add --no-cache git
WORKDIR /src
COPY patches/ /patches/
RUN git init -q . \
    && git remote add origin "${LLAMA_CPP_REPO}" \
    && git fetch -q --depth 1 origin "${LLAMA_CPP_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && while read -r p; do [ -n "$p" ] || continue; echo "apply $p"; git apply --whitespace=nowarn "/patches/$p"; done < /patches/series \
    && rm -rf .git

# ---- llama.cpp web UI ----
FROM docker.io/library/node:${NODE_VERSION} AS web
WORKDIR /app/ui/apps/chat
COPY --from=source /src/ui/apps/chat/package.json /src/ui/apps/chat/package-lock.json ./
RUN npm ci
COPY --from=source /src/ui/apps/chat/ ./
COPY --from=source /src/ui/lib/ui/package.json /app/ui/lib/ui/package.json
COPY --from=source /src/ui/lib/ui/src/lib/ /app/ui/lib/ui/src/lib/
COPY --from=source /src/ui/lib/voice/src/ /app/ui/lib/voice/src/
RUN npm run build

# ---- CUDA build ----
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
# 86 = RTX 30xx / A-series Ampere, 89 = RTX 40xx, 120 = RTX 50xx; each extra arch adds build time
ARG CUDA_ARCH=86
# nvcc needs ~1-2 GB of RAM per job for the flash-attention kernels
ARG BUILD_JOBS=4
ARG TARGETS="llama-server llama-perplexity llama-inkling-pass-kv test-flash-attn-indexed-bias test-backend-ops test-sched-streams test-llama-archs"
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake git libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
WORKDIR /app
COPY --from=source /src/ ./
COPY --from=web /app/ui/apps/chat/dist ui/apps/chat/dist
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" -DLLAMA_BUILD_TESTS=ON \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j "${BUILD_JOBS}" --target ${TARGETS} \
    && mkdir -p /out/lib \
    && cp build/bin/* /out/ \
    && find build -name "*.so*" -exec cp -P {} /out/lib \; \
    && cp LICENSE /out/LICENSE.llama.cpp

# ---- runtime ----
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# some hosts mount the driver only under /usr/local/nvidia
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/app/lib
COPY --from=build /out/ /app/
COPY docker/start-server.sh /app/start-server.sh
WORKDIR /app
EXPOSE 8080
VOLUME ["/models"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD ["curl", "-fs", "http://localhost:8080/health"]
ENTRYPOINT ["/app/start-server.sh"]
