#!/usr/bin/env bash
# Local A800 environment installer for the CUDA 12.8 host.
#
# This follows setup_env_safe.sh:
#   1. install every locked dependency except flash-attn
#   2. inspect the CUDA version used by the installed PyTorch wheel
#   3. select the matching CUDA 12.8 toolkit
#   4. install the prebuilt CPython 3.12 flash-attn SM80 wheel
#   5. call setup_env.sh to clone, patch and install Harbor/Verl/vLLM
#
# This script targets the current 8xA800-SXM4-80GB host (compute capability 8.0).
# It never builds flash-attn from source.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
BUILD_ROOT="${LEGO_BUILD_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/lego-rl-build-a800}"
TMPDIR="${TMPDIR:-$BUILD_ROOT/tmp}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$BUILD_ROOT/uv-cache}"
UV_BIN="${UV_BIN:-/data1/uv/bin/uv}"
WHEELHOUSE="${LEGO_WHEELHOUSE:-$BUILD_ROOT/wheels}"
MAX_JOBS="${LEGO_MAX_JOBS:-16}"
NVCC_THREADS="${NVCC_THREADS:-4}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/data1/cbsu/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl}"

TARGET_GPU_ARCH="${TARGET_GPU_ARCH:-8.0}"
TARGET_CMAKE_CUDA_ARCH="${TARGET_CMAKE_CUDA_ARCH:-${TARGET_GPU_ARCH/./}}"
CUDA_12_8_HOME="${CUDA_12_8_HOME:-/usr/local/cuda-12.8}"

mkdir -p "$BUILD_ROOT" "$TMPDIR" "$UV_CACHE_DIR" "$WHEELHOUSE"
export TMPDIR UV_CACHE_DIR
export UV_OVERRIDE="$REPO_ROOT/overrides.txt"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export MAX_JOBS
export NVCC_THREADS
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export MAKEFLAGS="${MAKEFLAGS:-} -j${MAX_JOBS}"
export TOKENIZERS_PARALLELISM=false

log()  { printf '\033[1;32m[a800-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[a800-setup WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[a800-setup ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

ensure_uv() {
    [[ -x "$UV_BIN" ]] \
        || die "uv not found at $UV_BIN; set UV_BIN to an existing executable (automatic installation is disabled)"
    export PATH="$(dirname "$UV_BIN"):$PATH"
    hash -r
    log "uv: $("$UV_BIN" --version)"
}

check_resources() {
    local mem_kb available_kb disk_kb
    mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
    available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    disk_kb="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"

    log "memory: available=$((available_kb / 1024 / 1024)) GiB total=$((mem_kb / 1024 / 1024)) GiB"
    log "disk: available at repo=$((disk_kb / 1024 / 1024)) GiB"
    log "build concurrency: MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"

    (( available_kb >= 24 * 1024 * 1024 )) \
        || die "less than 24 GiB RAM is available; refusing to install"
    if [[ "$disk_kb" =~ ^[0-9]+$ ]]; then
        (( disk_kb >= 100 * 1024 * 1024 )) \
            || die "less than 100 GiB free disk at $REPO_ROOT; refusing to install"
    fi
}

check_prerequisites() {
    [[ "$(uname -s)" == "Linux" ]] || die "Linux is required"
    [[ "$(uname -m)" == "x86_64" ]] || die "x86_64 is required"
    [[ "$TARGET_GPU_ARCH" =~ ^[0-9]+\.[0-9]+$ ]] \
        || die "TARGET_GPU_ARCH must look like 8.0"
    [[ "$PYTHON_VERSION" == 3.12* ]] \
        || die "the prebuilt flash-attn wheel requires CPython 3.12"
    [[ -f "$FLASH_ATTN_WHEEL" ]] \
        || die "prebuilt flash-attn wheel not found: $FLASH_ATTN_WHEEL"
    [[ "$FLASH_ATTN_WHEEL" == *-cp312-* ]] \
        || die "flash-attn wheel must be a CPython 3.12 wheel: $FLASH_ATTN_WHEEL"
    [[ -x "$CUDA_12_8_HOME/bin/nvcc" ]] \
        || die "CUDA 12.8 nvcc not found: $CUDA_12_8_HOME/bin/nvcc"
    ensure_uv
    require_cmd git
    require_cmd nvidia-smi
    require_cmd gcc
    require_cmd g++
    require_cmd nice
    require_cmd ionice
    [[ -f "$REPO_ROOT/requirements.txt" ]] || die "requirements.txt not found"
    [[ -f "$REPO_ROOT/overrides.txt" ]] || die "overrides.txt not found"
    [[ -f "$REPO_ROOT/scripts/setup_env.sh" ]] || die "setup_env.sh not found"
}

configure_target_arch() {
    local detected_arch

    detected_arch="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
            | sed 's/[[:space:]]//g' \
            | sort -u \
            | paste -sd';' -
    )"
    [[ -n "$detected_arch" ]] || die "could not detect the local GPU compute capability"
    [[ "$detected_arch" != *";"* ]] \
        || die "multiple GPU architectures detected ($detected_arch); set TARGET_GPU_ARCH explicitly"
    [[ "$detected_arch" == "$TARGET_GPU_ARCH" ]] \
        || die "detected GPU architecture $detected_arch does not match TARGET_GPU_ARCH=$TARGET_GPU_ARCH"

    log "detected local GPU architecture: $detected_arch (CMake $TARGET_CMAKE_CUDA_ARCH)"
}

check_compiler() {
    local cc="${CC:-gcc}" cxx="${CXX:-g++}"
    local cc_major cxx_major
    local lib_dir

    cc_major="$("$cc" -dumpversion | awk -F. '{print $1}')"
    cxx_major="$("$cxx" -dumpversion | awk -F. '{print $1}')"
    (( cc_major >= 9 && cxx_major >= 9 )) \
        || die "PyTorch 2.10 requires GCC/G++ >= 9; load a newer GCC module first"

    export CC="$cc" CXX="$cxx"
    export PATH="$(dirname "$cc"):$PATH"
    if [[ "$(dirname "$cc")" == /fs1/software/gcc/*/bin ]]; then
        lib_dir="$(cd "$(dirname "$cc")/.." && pwd)/lib64"
        export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$lib_dir:${LIBRARY_PATH:-}"
        export LDFLAGS="-Wl,-rpath,$lib_dir ${LDFLAGS:-}"
    fi
    log "C compiler: $($cc --version | sed -n '1p')"
    log "C++ compiler: $($cxx --version | sed -n '1p')"
}

nvcc_major_minor() {
    "$1" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | sed -n '1p'
}

select_cuda_home() {
    local torch_cuda="$1"
    local candidate="" nvcc_bin expected actual

    case "$torch_cuda" in
        12.8*)
            candidate="$CUDA_12_8_HOME"
            ;;
        "")
            die "installed PyTorch has no CUDA runtime"
            ;;
        *)
            die "this host provides CUDA 12.8, but PyTorch was built for CUDA $torch_cuda"
            ;;
    esac

    nvcc_bin="$candidate/bin/nvcc"
    [[ -x "$nvcc_bin" ]] || die "matching nvcc not found at $nvcc_bin"
    expected="$(printf '%s' "$torch_cuda" | awk -F. '{print $1 "." $2}')"
    actual="$(nvcc_major_minor "$nvcc_bin")"
    [[ "$actual" == "$expected" ]] \
        || die "selected nvcc $actual does not match PyTorch CUDA $expected"
    SELECTED_CUDA_HOME="$candidate"
}

install_dependencies() {
    local py="$VENV_PATH/bin/python"
    local no_flash_req="$BUILD_ROOT/requirements-no-flash-attn.txt"

    if [[ ! -x "$py" ]]; then
        log "creating virtualenv: $VENV_PATH (Python $PYTHON_VERSION)"
        "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_PATH"
    else
        log "reusing virtualenv: $VENV_PATH"
    fi

    grep -vE '^flash-attn([<=>!~]|$)' "$REPO_ROOT/requirements.txt" > "$no_flash_req"
    log "installing pinned dependencies except flash-attn"
    "$UV_BIN" pip install --python "$py" -r "$no_flash_req"
}

install_flash_attn() {
    local py="$VENV_PATH/bin/python"
    local py_tag

    py_tag="$(
        "$py" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")'
    )"
    [[ "$py_tag" == "cp312" ]] \
        || die "venv uses $py_tag, but the prebuilt flash-attn wheel requires cp312"

    log "installing prebuilt flash-attn wheel: $FLASH_ATTN_WHEEL"
    "$UV_BIN" pip install --python "$py" --no-deps --force-reinstall "$FLASH_ATTN_WHEEL"
}

main() {
    local py="$VENV_PATH/bin/python"
    local torch_info torch_version torch_cuda

    check_prerequisites
    configure_target_arch
    check_compiler
    check_resources

    export TORCH_CUDA_ARCH_LIST="$TARGET_GPU_ARCH"
    export FLASH_ATTN_CUDA_ARCHS="${TARGET_GPU_ARCH/./}"
    export CMAKE_CUDA_ARCHITECTURES="$TARGET_CMAKE_CUDA_ARCH"
    log "target GPU architectures: $TORCH_CUDA_ARCH_LIST"

    install_dependencies

    torch_info="$(
        "$py" -c 'import torch; print(torch.__version__); print(torch.version.cuda or "")'
    )"
    torch_version="$(printf '%s\n' "$torch_info" | sed -n '1p')"
    torch_cuda="$(printf '%s\n' "$torch_info" | sed -n '2p')"
    [[ -n "$torch_cuda" ]] || die "installed PyTorch has no CUDA runtime"
    log "installed torch=$torch_version cuda=$torch_cuda"

    select_cuda_home "$torch_cuda"
    export CUDA_HOME="$SELECTED_CUDA_HOME"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"

    log "matching CUDA toolkit: $CUDA_HOME"
    "$CUDACXX" --version | sed -n '1,5p'

    install_flash_attn

    "$py" -c \
        'import flash_attn, torch; print("flash-attn", flash_attn.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda)'

    log "cloning, patching and installing Harbor/Verl/vLLM"
    SKIP_CLONE="${SKIP_CLONE:-0}" \
    SKIP_REQUIREMENTS=1 \
    VENV_PATH="$VENV_PATH" \
    MAX_JOBS="$MAX_JOBS" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    CUDA_HOME="$CUDA_HOME" \
    CUDACXX="$CUDACXX" \
    CC="$CC" CXX="$CXX" \
    PATH="$PATH" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    LIBRARY_PATH="${LIBRARY_PATH:-}" \
    CPATH="${CPATH:-}" \
    LDFLAGS="${LDFLAGS:-}" \
    CMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" \
    bash "$REPO_ROOT/scripts/setup_env.sh"

    if ! "$py" -c 'import docker' >/dev/null 2>&1; then
        log "installing Docker Python SDK"
        "$UV_BIN" pip install --python "$py" docker
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

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    import vllm

    print("GPU 0:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
    print("vllm:", vllm.__version__)
else:
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        raise SystemExit("vllm is not importable after setup")
    print("vllm spec:", spec.origin)
print("flash-attn:", flash_attn.__version__)
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
        printf 'wheelhouse=%s\n' "$WHEELHOUSE"
    } > "$BUILD_ROOT/setup-success.txt"

    log "environment ready"
    log "activate with: source '$VENV_PATH/bin/activate'"
}

main "$@"
