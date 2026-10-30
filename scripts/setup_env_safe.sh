#!/usr/bin/env bash
# Conservative Lego-RL environment installer.
#
# The main difference from setup_env.sh is ordering:
#   1. install every pinned dependency except flash-attn
#   2. inspect the CUDA version compiled into the installed PyTorch wheel
#   3. select a matching nvcc toolchain
#   4. compile only flash-attn, serially, for the local GPU architecture
#   5. let setup_env.sh clone, patch and install Harbor/Verl/vLLM
#
# This avoids building CUDA extensions before the actual PyTorch CUDA ABI is known.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
BUILD_ROOT="${LEGO_BUILD_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/lego-rl-build}"
TMPDIR="${TMPDIR:-$BUILD_ROOT/tmp}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$BUILD_ROOT/uv-cache}"
MAX_JOBS="${LEGO_MAX_JOBS:-1}"

mkdir -p "$BUILD_ROOT" "$TMPDIR" "$UV_CACHE_DIR"
export TMPDIR UV_CACHE_DIR
export UV_OVERRIDE="$REPO_ROOT/overrides.txt"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

log() {
    printf '\033[1;32m[safe-setup]\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33m[safe-setup WARN]\033[0m %s\n' "$*" >&2
}

die() {
    printf '\033[1;31m[safe-setup ERROR]\033[0m %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_resources() {
    local mem_kb available_kb disk_kb
    mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
    available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    disk_kb="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"

    log "memory: available=$((available_kb / 1024 / 1024)) GiB total=$((mem_kb / 1024 / 1024)) GiB"
    log "disk: available at repo=$((disk_kb / 1024 / 1024)) GiB"
    log "build concurrency: MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"

    if (( available_kb < 32 * 1024 * 1024 )); then
        die "less than 32 GiB RAM is available; refusing to compile"
    fi
    if (( disk_kb < 100 * 1024 * 1024 )); then
        die "less than 100 GiB free disk at $REPO_ROOT; refusing to install"
    fi
    if (( MAX_JOBS > 2 )); then
        warn "LEGO_MAX_JOBS=$MAX_JOBS is higher than the conservative limit; use 1 for the first install"
    fi
}

detect_gpu_archs() {
    local archs
    archs="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
            | sed 's/[[:space:]]//g' \
            | sort -u \
            | paste -sd';' -
    )"
    [[ -n "$archs" ]] || die "could not detect GPU compute capability"
    printf '%s' "$archs"
}

select_cuda_home() {
    local torch_cuda="$1"
    local py="$2"
    local candidate nvcc_bin

    case "$torch_cuda" in
        12.8*)
            candidate="/usr/local/cuda-12.8"
            ;;
        12.9*)
            candidate="/usr/local/cuda-12.9"
            ;;
        13.0*)
            log "PyTorch uses CUDA 13.0; installing a matching pip nvcc toolchain"
            uv pip install --python "$py" \
                "nvidia-cuda-nvcc==13.0.88" \
                "nvidia-cuda-crt==13.0.88" \
                "nvidia-cuda-cccl==13.0.85" \
                "nvidia-nvvm==13.0.88" \
                "nvidia-cuda-runtime==13.0.96"
            candidate="$(
                "$py" -c \
                    'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))'
            )"
            mkdir -p "$candidate/lib"
            if [[ -f "$candidate/lib/libcudart.so.13" && ! -e "$candidate/lib/libcudart.so" ]]; then
                ln -s "$candidate/lib/libcudart.so.13" "$candidate/lib/libcudart.so"
            fi
            ;;
        "")
            die "installed PyTorch has no CUDA runtime (torch.version.cuda is empty)"
            ;;
        *)
            die "unsupported PyTorch CUDA version: $torch_cuda"
            ;;
    esac

    nvcc_bin="$candidate/bin/nvcc"
    [[ -x "$nvcc_bin" ]] || die "matching nvcc not found at $nvcc_bin"
    printf '%s' "$candidate"
}

check_prerequisites() {
    [[ "$(uname -s)" == "Linux" ]] || die "Linux is required"
    [[ "$(uname -m)" == "x86_64" ]] || die "x86_64 is required"
    require_cmd uv
    require_cmd git
    require_cmd nvidia-smi
    require_cmd nvcc
    require_cmd nice
    require_cmd ionice
    [[ -f "$REPO_ROOT/requirements.txt" ]] || die "requirements.txt not found"
    [[ -f "$REPO_ROOT/overrides.txt" ]] || die "overrides.txt not found"
    [[ -f "$REPO_ROOT/scripts/setup_env.sh" ]] || die "setup_env.sh not found"
}

main() {
    local gpu_archs no_flash_req torch_info torch_version torch_cuda cuda_home
    local py="$VENV_PATH/bin/python"

    check_prerequisites
    check_resources

    gpu_archs="$(detect_gpu_archs)"
    export TORCH_CUDA_ARCH_LIST="$gpu_archs"
    log "GPU architectures: $TORCH_CUDA_ARCH_LIST"

    if [[ ! -x "$py" ]]; then
        log "creating virtualenv: $VENV_PATH (Python $PYTHON_VERSION)"
        uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
    else
        log "reusing virtualenv: $VENV_PATH"
    fi

    no_flash_req="$BUILD_ROOT/requirements-no-flash-attn.txt"
    grep -vE '^flash-attn([<=>!~]|$)' "$REPO_ROOT/requirements.txt" > "$no_flash_req"

    log "installing pinned dependencies except flash-attn (this may take a while)"
    uv pip install --python "$py" -r "$no_flash_req"

    torch_info="$(
        "$py" -c \
            'import torch; print(torch.__version__); print(torch.version.cuda or "")'
    )"
    torch_version="$(printf '%s\n' "$torch_info" | sed -n '1p')"
    torch_cuda="$(printf '%s\n' "$torch_info" | sed -n '2p')"
    log "installed torch=$torch_version cuda=$torch_cuda"

    cuda_home="$(select_cuda_home "$torch_cuda" "$py")"
    export CUDA_HOME="$cuda_home"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"

    log "matching CUDA toolkit: $CUDA_HOME"
    "$CUDA_HOME/bin/nvcc" --version | sed -n '1,5p'

    if "$py" -c 'import flash_attn' >/dev/null 2>&1; then
        log "flash-attn is already importable; skipping rebuild"
    else
        log "compiling flash-attn 2.8.3 serially for $TORCH_CUDA_ARCH_LIST"
        ionice -c2 -n7 nice -n 10 uv pip install \
            --python "$py" \
            --no-build-isolation \
            --no-cache \
            --no-deps \
            --force-reinstall \
            "flash-attn==2.8.3"
    fi

    "$py" -c \
        'import flash_attn, torch; print("flash-attn", flash_attn.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda)'

    log "cloning, patching and installing Harbor/Verl/vLLM"
    SKIP_REQUIREMENTS=1 \
    VENV_PATH="$VENV_PATH" \
    MAX_JOBS="$MAX_JOBS" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    CUDA_HOME="$CUDA_HOME" \
    PATH="$PATH" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    LIBRARY_PATH="${LIBRARY_PATH:-}" \
    CPATH="${CPATH:-}" \
    bash "$REPO_ROOT/scripts/setup_env.sh"

    if ! "$py" -c 'import docker' >/dev/null 2>&1; then
        log "installing Docker Python SDK"
        uv pip install --python "$py" docker
    fi

    log "running import and CUDA smoke checks"
    "$py" - <<'PY'
import importlib.util

import docker
import flash_attn
import harbor
import torch
import verl
import verl_patch
import vllm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
print("flash-attn:", flash_attn.__version__)
print("vllm:", vllm.__version__)
print("harbor:", harbor.__file__)
print("verl:", verl.__file__)
print("verl_patch:", importlib.util.find_spec("verl_patch").origin)
print("docker SDK: ok")
PY

    {
        printf 'completed_at=%s\n' "$(date -Is)"
        printf 'python=%s\n' "$("$py" --version 2>&1)"
        printf 'torch=%s\n' "$torch_version"
        printf 'torch_cuda=%s\n' "$torch_cuda"
        printf 'cuda_home=%s\n' "$CUDA_HOME"
        printf 'torch_cuda_arch_list=%s\n' "$TORCH_CUDA_ARCH_LIST"
    } > "$BUILD_ROOT/setup-success.txt"

    log "environment ready"
    log "activate with: source '$VENV_PATH/bin/activate'"
}

main "$@"
