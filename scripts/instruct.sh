#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/instruct.sh \
    --python-bin /path/to/python \
    --cuda-home /usr/local/cuda \
    [--cuda-arch 90] \
    [--cuda-visible-devices 0] \
    [--device cuda:0] \
    [--mcts-sims 1024] \
    [--v1-cgs 32,64] \
    [--total-games 8] \
    [--nsys-duration-sec 30] \
    [--stable-duration-sec 180] \
    [--build-v0 1]

Notes:
  - This script compiles v0_core (when --build-v0=1) and then calls scripts/instruction.sh.
  - All results are written under results/v1_matrix/<timestamp>.
USAGE
}

PYTHON_BIN=""
CUDA_HOME=""
CUDA_ARCH="90"
CUDA_VISIBLE_DEVICES_VAL="0"
DEVICE="cuda:0"
MCTS_SIMS="1024"
V1_CGS="32,64"
TOTAL_GAMES="8"
NSYS_DURATION_SEC="30"
STABLE_DURATION_SEC="180"
BUILD_V0="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --cuda-home)
      CUDA_HOME="${2:-}"
      shift 2
      ;;
    --cuda-arch)
      CUDA_ARCH="${2:-}"
      shift 2
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VAL="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --mcts-sims)
      MCTS_SIMS="${2:-}"
      shift 2
      ;;
    --v1-cgs)
      V1_CGS="${2:-}"
      shift 2
      ;;
    --total-games)
      TOTAL_GAMES="${2:-}"
      shift 2
      ;;
    --nsys-duration-sec)
      NSYS_DURATION_SEC="${2:-}"
      shift 2
      ;;
    --stable-duration-sec)
      STABLE_DURATION_SEC="${2:-}"
      shift 2
      ;;
    --build-v0)
      BUILD_V0="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" || -z "$CUDA_HOME" ]]; then
  echo "[error] --python-bin and --cuda-home are required."
  usage
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python not executable: $PYTHON_BIN"
  exit 2
fi

if [[ ! -d "$CUDA_HOME" ]]; then
  echo "[error] cuda home not found: $CUDA_HOME"
  exit 2
fi

NVCC="$CUDA_HOME/bin/nvcc"
if [[ ! -x "$NVCC" ]]; then
  echo "[error] nvcc not found: $NVCC"
  exit 2
fi

CONDA_PREFIX_DETECTED="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
PYTHON_INCLUDE="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths().get("include",""))')"
PYTHON_LIBRARY="$("$PYTHON_BIN" -c 'import pathlib,sysconfig; libdir=sysconfig.get_config_var("LIBDIR") or ""; ldlib=sysconfig.get_config_var("LDLIBRARY") or ""; print(str(pathlib.Path(libdir)/ldlib) if libdir and ldlib else "")')"
PYBIND11_DIR="$("$PYTHON_BIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
TORCH_PREFIX="$("$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

echo "[info] root=$ROOT_DIR"
echo "[info] python=$PYTHON_BIN"
echo "[info] conda_prefix=$CONDA_PREFIX_DETECTED"
echo "[info] cuda_home=$CUDA_HOME"
echo "[info] nvcc=$NVCC"
echo "[info] cuda_arch=$CUDA_ARCH"

if [[ "$BUILD_V0" == "1" ]]; then
  echo "[step] rebuild v0_core"
  rm -rf build/v0
  export CUDACXX="$NVCC"
  export PATH="$CUDA_HOME/bin:$PATH"
  cmake -S v0 -B build/v0 \
    -G Ninja \
    -DUSE_CUDA=ON \
    -DBUILD_CUDA_KERNELS=ON \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE" \
    -DPython3_LIBRARY="$PYTHON_LIBRARY" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX_DETECTED;$TORCH_PREFIX"
  cmake --build build/v0 -j"$(nproc)"
fi

echo "[step] run evaluation pipeline"
PYTHON_BIN="$PYTHON_BIN" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VAL" \
DEVICE="$DEVICE" \
MCTS_SIMS="$MCTS_SIMS" \
V1_CGS="$V1_CGS" \
TOTAL_GAMES="$TOTAL_GAMES" \
NSYS_DURATION_SEC="$NSYS_DURATION_SEC" \
STABLE_DURATION_SEC="$STABLE_DURATION_SEC" \
BUILD_V0=0 \
bash scripts/instruction.sh

