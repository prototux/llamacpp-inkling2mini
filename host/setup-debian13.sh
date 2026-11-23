#!/usr/bin/env bash
# Prepare a Debian 13 (trixie) host with an NVIDIA GPU for this setup:
#   1. check the NVIDIA driver; upgrade it from CUDA's offline (local) repository only if it is too old
#   2. install the NVIDIA Container Toolkit (libnvidia-container) and register it with Docker
#   3. run a GPU container to prove it works
#
# usage: sudo host/setup-debian13.sh                              check the driver, install the toolkit
#        sudo host/setup-debian13.sh --cuda-repo cuda-repo-debian13-13-*-local_*_amd64.deb
#                                                                 also upgrade the driver from that local repo
#
# The local repo .deb comes from https://developer.nvidia.com/cuda-downloads
# (Linux > x86_64 > Debian > 13 > deb (local)). It holds the driver and the CUDA toolkit; only the driver is
# installed here, the containers bring their own CUDA runtime.
set -euo pipefail

# the image's CUDA 12.8 runtime needs driver >= 570; any newer driver runs it
min_driver=570
test_image=docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04

cuda_repo=
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-repo) cuda_repo=$2; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "error: unknown argument $1" >&2; exit 1 ;;
    esac
done

log() { printf '\n== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root (sudo)"
# shellcheck source=/dev/null
. /etc/os-release
[[ ${VERSION_CODENAME:-} == trixie ]] || echo "warning: written for Debian 13 (trixie), this is ${PRETTY_NAME:-unknown}"
command -v docker >/dev/null || die "Docker is not installed (https://docs.docker.com/engine/install/debian/)"
docker compose version >/dev/null 2>&1 || die "the Docker Compose plugin is missing (apt-get install docker-compose-plugin)"

driver_version() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
}

log "NVIDIA driver"
current=$(driver_version || true)
if [[ -n $current ]]; then
    echo "installed: $current ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
fi

if [[ -n $current && ${current%%.*} -ge $min_driver && -z $cuda_repo ]]; then
    echo "ok: >= $min_driver, no upgrade needed"
elif [[ -z $cuda_repo ]]; then
    die "driver ${current:-missing} is older than $min_driver: download the Debian 13 'deb (local)' CUDA repo and re-run with --cuda-repo FILE.deb"
else
    [[ -f $cuda_repo ]] || die "$cuda_repo not found"
    log "installing the driver from the offline repository $cuda_repo"
    apt-get update
    apt-get install -y "linux-headers-$(uname -r)" dkms
    dpkg -i "$cuda_repo"
    # the local repo ships its signing key; apt needs it in /usr/share/keyrings
    cp /var/cuda-repo-debian13-*-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    # open kernel modules: NVIDIA's recommended driver flavour for Turing and newer (the 3090 is Ampere)
    apt-get install -y nvidia-open
    echo "driver installed: reboot, then re-run this script without --cuda-repo to finish"
    exit 0
fi

log "NVIDIA Container Toolkit"
if ! command -v nvidia-ctk >/dev/null; then
    apt-get update
    apt-get install -y curl gpg ca-certificates
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
else
    echo "already installed: $(nvidia-ctk --version | head -1)"
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

log "GPU in a container"
docker run --rm --gpus all "$test_image" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo
echo "host ready"
