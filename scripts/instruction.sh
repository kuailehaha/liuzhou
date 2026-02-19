#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-12345}"
MCTS_SIMS="${MCTS_SIMS:-1024}"
V1_CGS="${V1_CGS:-32,64}"
TOTAL_GAMES="${TOTAL_GAMES:-8}"
NSYS_DURATION_SEC="${NSYS_DURATION_SEC:-30}"
STABLE_DURATION_SEC="${STABLE_DURATION_SEC:-180}"
BUILD_V0="${BUILD_V0:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/build/v0/src:$ROOT_DIR/v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDACXX=/usr/local/cuda/bin/nvcc
export PYTHONPATH=./liuzhou:./liuzhou/v0/build/src:$PYTHONPATH

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/v1_matrix/$RUN_TAG"
mkdir -p "$OUT_DIR"

echo "[info] root=$ROOT_DIR"
echo "[info] python=$PYTHON_BIN"
echo "[info] device=$DEVICE"
echo "[info] sims=$MCTS_SIMS"
echo "[info] out_dir=$OUT_DIR"

if [[ "$BUILD_V0" == "1" ]]; then
  echo "[step] build v0_core (required by v1 pipeline)"
  TORCH_CMAKE_PREFIX="$("$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  PYBIND11_CMAKE_DIR="$("$PYTHON_BIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  
  PY=$CONDA_PREFIX/bin/python
  cmake -S v0 -B v0/build \
    -G Ninja \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPython3_EXECUTABLE=$PY \
    -DPython3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.13 \
    -DPython3_LIBRARY=$CONDA_PREFIX/lib/libpython3.13.so \
    -Dpybind11_DIR="$($PY -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$($PY -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build v0/build -j"$(nproc)"
fi

echo "[step] gpu status"
nvidia-smi --query-gpu=index,name,utilization.gpu,power.draw,memory.used,pstate --format=csv

# echo "[step] python/v0_core sanity"
# "$PYTHON_BIN" - <<'PY'
# import torch
# import v0_core
# print("torch", torch.__version__)
# print("cuda_available", torch.cuda.is_available())
# print("v0_core_ok", hasattr(v0_core, "root_puct_allocate_visits"))
# PY

# echo "[step] semantic AB (256/512)"
# "$PYTHON_BIN" tools/ab_v1_child_value_only.py \
#   --device "$DEVICE" \
#   --seed "$SEED" \
#   --num-states 32 \
#   --state-plies 8 \
#   --mcts-simulations 256 \
#   --self-play-games 8 \
#   --self-play-concurrent-games 8 \
#   --strict \
#   --output-json "$OUT_DIR/ab_v1_child_value_256.json"

# "$PYTHON_BIN" tools/ab_v1_child_value_only.py \
#   --device "$DEVICE" \
#   --seed "$SEED" \
#   --num-states 32 \
#   --state-plies 8 \
#   --mcts-simulations 512 \
#   --self-play-games 8 \
#   --self-play-concurrent-games 8 \
#   --strict \
#   --output-json "$OUT_DIR/ab_v1_child_value_512.json"

# for FG in off on; do
#   echo "[step] validate_v1_claims finalize_graph=$FG"
#   "$PYTHON_BIN" tools/validate_v1_claims.py \
#     --device "$DEVICE" \
#     --seed "$SEED" \
#     --rounds 1 \
#     --v0-workers 1,2,4 \
#     --v1-threads 1,2,4 \
#     --v1-concurrent-games 64 \
#     --v1-sample-moves false \
#     --v1-finalize-graph "$FG" \
#     --v1-child-eval-mode value_only \
#     --total-games "$TOTAL_GAMES" \
#     --v0-mcts-simulations "$MCTS_SIMS" \
#     --v1-mcts-simulations "$MCTS_SIMS" \
#     --v0-batch-leaves 512 \
#     --v0-inference-backend graph \
#     --v0-inference-batch-size 512 \
#     --v0-inference-warmup-iters 5 \
#     --v1-inference-backend py \
#     --v1-inference-batch-size 512 \
#     --v1-inference-warmup-iters 5 \
#     --v0-opening-random-moves 2 \
#     --v0-resign-threshold -0.8 \
#     --v0-resign-min-moves 36 \
#     --v0-resign-consecutive 3 \
#     --with-inference-baseline \
#     --output-json "$OUT_DIR/validate_v1_$FG.json"
# done

# for SAMPLE in false true; do
#   for FG in off on; do
#     echo "[step] sweep_v1_gpu_matrix sample_moves=$SAMPLE finalize_graph=$FG"
#     "$PYTHON_BIN" tools/sweep_v1_gpu_matrix.py \
#       --device "$DEVICE" \
#       --seed "$SEED" \
#       --rounds 1 \
#       --threads 1 \
#       --concurrent-games "$V1_CGS" \
#       --backends py \
#       --total-games "$TOTAL_GAMES" \
#       --mcts-simulations "$MCTS_SIMS" \
#       --child-eval-mode value_only \
#       --sample-moves "$SAMPLE" \
#       --finalize-graph "$FG" \
#       --inference-batch-size 512 \
#       --inference-warmup-iters 5 \
#       --output-json "$OUT_DIR/matrix_py_${SAMPLE}_${FG}.json"
#   done
# done

# echo "[step] stable run (v1, ${STABLE_DURATION_SEC}s)"
# "$PYTHON_BIN" tools/run_selfplay_workload.py \
#   --mode v1 \
#   --device "$DEVICE" \
#   --seed "$SEED" \
#   --duration-sec "$STABLE_DURATION_SEC" \
#   --num-games-per-iter 64 \
#   --mcts-simulations "$MCTS_SIMS" \
#   --v1-threads 1 \
#   --v1-concurrent-games 64 \
#   --v1-child-eval-mode value_only \
#   --v1-sample-moves false \
#   --v1-finalize-graph on \
#   --v1-inference-backend py \
#   --v1-inference-batch-size 512 \
#   --v1-inference-warmup-iters 5 \
#   --collect-step-timing \
#   --plot-step-breakdown \
#   --plot-stability \
#   --output-json "$OUT_DIR/stable_v1_180s.json"

echo "[step] power-efficiency probe (v1, target >=400W on H20)"
echo "[step] profile: sims=8192, cg=512, batch=1024, sample_moves=false, finalize_graph=on"
V1_FINALIZE_GRAPH_MAX_ENTRIES=256 \
"$PYTHON_BIN" tools/run_selfplay_workload.py \
  --mode v1 \
  --device "$DEVICE" \
  --seed "$SEED" \
  --duration-sec "${POWER_PROBE_DURATION_SEC:-180}" \
  --num-games-per-iter 512 \
  --mcts-simulations "${POWER_PROBE_MCTS_SIMS:-8192}" \
  --v1-threads 8 \
  --v1-concurrent-games 512 \
  --v1-child-eval-mode value_only \
  --v1-sample-moves false \
  --v1-finalize-graph on \
  --v1-inference-backend py \
  --v1-inference-batch-size 2048 \
  --v1-inference-warmup-iters 5 \
  --collect-step-timing \
  --plot-step-breakdown \
  --plot-stability \
  --output-json "$OUT_DIR/power_probe_v1_400w_target.json"

if command -v nsys >/dev/null 2>&1; then
  for CG in 32 64; do
    for FG in off on; do
      echo "[step] nsys v1 cg=$CG finalize_graph=$FG"
      "$PYTHON_BIN" tools/nsys_v0_v1_compare.py \
        --device "$DEVICE" \
        --seed "$SEED" \
        --duration-sec "$NSYS_DURATION_SEC" \
        --num-games-per-iter 64 \
        --mcts-simulations "$MCTS_SIMS" \
        --v1-threads 1 \
        --v1-concurrent-games "$CG" \
        --v1-child-eval-mode value_only \
        --v1-sample-moves false \
        --v1-finalize-graph "$FG" \
        --v1-inference-backend py \
        --v1-inference-batch-size 512 \
        --v1-inference-warmup-iters 5 \
        --profile-modes v1 \
        --output-dir "$OUT_DIR/nsys_v1_cg${CG}_${FG}"
    done
  done
else
  echo "[warn] nsys not found, skip nsys profile stage"
fi

find "$OUT_DIR" -type f | sort > "$OUT_DIR/files.txt"
echo "[done] all outputs saved under: $OUT_DIR"
