#!/usr/bin/env bash
# High-load v1 staged training script (self-play -> train -> infer).
set -euo pipefail

export PYTHONPATH="./:./build/v0/src:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"

PROFILE="${PROFILE:-aggressive}" # stable | aggressive
TRAIN_STRATEGY="${TRAIN_STRATEGY:-ddp}" # ddp | data_parallel | none
DEVICE="${DEVICE:-cuda:0}"

SELF_PLAY_DEVICES="${SELF_PLAY_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
TRAIN_DEVICES="${TRAIN_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
INFER_DEVICES="${INFER_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_v1_big}"
RUN_ROOT="${RUN_ROOT:-./v1/data/stage_runs}"
RUN_INFER_STAGE="${RUN_INFER_STAGE:-1}"

TEMPERATURE_INIT="${TEMPERATURE_INIT:-1.0}"
TEMPERATURE_FINAL="${TEMPERATURE_FINAL:-0.1}"
TEMPERATURE_THRESHOLD="${TEMPERATURE_THRESHOLD:-10}"
EXPLORATION_WEIGHT="${EXPLORATION_WEIGHT:-1.0}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.25}"
SOFT_VALUE_K="${SOFT_VALUE_K:-2.0}"
MAX_GAME_PLIES="${MAX_GAME_PLIES:-512}"
SELF_PLAY_CONCURRENT_GAMES="${SELF_PLAY_CONCURRENT_GAMES:-8192}"
SELF_PLAY_BACKEND="${SELF_PLAY_BACKEND:-process}" # auto | thread | process
SELF_PLAY_SHARD_DIR="${SELF_PLAY_SHARD_DIR:-}"

if [[ "$PROFILE" == "stable" ]]; then
  : "${ITERATIONS:=60}"
  : "${SELF_PLAY_GAMES:=12288}"
  : "${MCTS_SIMULATIONS:=1536}"
  : "${BATCH_SIZE:=8192}"
  : "${EPOCHS:=3}"
  : "${LR:=2.5e-4}"
  : "${WEIGHT_DECAY:=1e-4}"
elif [[ "$PROFILE" == "aggressive" ]]; then
  : "${ITERATIONS:=80}"
  : "${SELF_PLAY_GAMES:=32768}"
  : "${MCTS_SIMULATIONS:=4096}"
  : "${BATCH_SIZE:=16384}"
  : "${EPOCHS:=4}"
  : "${LR:=2e-4}"
  : "${WEIGHT_DECAY:=1e-4}"
else
  echo "[big_train_v1] unsupported PROFILE=$PROFILE (use stable/aggressive)" >&2
  exit 1
fi

: "${INFER_BATCH_SIZE:=4096}"
: "${INFER_WARMUP_ITERS:=20}"
: "${INFER_ITERS:=80}"

csv_count() {
  local csv="$1"
  local IFS=','
  local parts=()
  read -r -a parts <<< "$csv"
  local n=0
  local item=""
  for item in "${parts[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "$item" ]] && continue
    n=$((n + 1))
  done
  echo "$n"
}

to_visible_indices() {
  local csv="$1"
  local IFS=','
  local parts=()
  read -r -a parts <<< "$csv"
  local out=()
  local item=""
  for item in "${parts[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "$item" ]] && continue
    item="${item#cuda:}"
    out+=("$item")
  done
  (
    IFS=','
    echo "${out[*]}"
  )
}

build_local_cuda_list() {
  local count="$1"
  local out=()
  local idx=0
  for ((idx = 0; idx < count; idx++)); do
    out+=("cuda:${idx}")
  done
  (
    IFS=','
    echo "${out[*]}"
  )
}

TRAIN_NPROC="$(csv_count "$TRAIN_DEVICES")"
if [[ "$TRAIN_STRATEGY" == "ddp" && "$TRAIN_NPROC" -le 1 ]]; then
  echo "[big_train_v1] TRAIN_STRATEGY=ddp needs >1 gpu; fallback to data_parallel"
  TRAIN_STRATEGY="data_parallel"
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "$CHECKPOINT_DIR" "$RUN_DIR" logs
LOG_FILE="logs/big_train_v1_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[big_train_v1] run_tag=$RUN_TAG"
echo "[big_train_v1] profile=$PROFILE train_strategy=$TRAIN_STRATEGY"
echo "[big_train_v1] python=$PYTHON_BIN"
echo "[big_train_v1] self_play_devices=$SELF_PLAY_DEVICES"
echo "[big_train_v1] train_devices=$TRAIN_DEVICES"
echo "[big_train_v1] infer_devices=$INFER_DEVICES"
echo "[big_train_v1] checkpoints=$CHECKPOINT_DIR"
echo "[big_train_v1] run_dir=$RUN_DIR"
echo "[big_train_v1] self_play_concurrent_games=$SELF_PLAY_CONCURRENT_GAMES"
echo "[big_train_v1] self_play_backend=$SELF_PLAY_BACKEND"
if [[ -n "$SELF_PLAY_SHARD_DIR" ]]; then
  echo "[big_train_v1] self_play_shard_dir=$SELF_PLAY_SHARD_DIR"
fi

LATEST_MODEL="${LOAD_CHECKPOINT:-}"
if [[ -n "$LATEST_MODEL" && ! -f "$LATEST_MODEL" ]]; then
  echo "[big_train_v1] load checkpoint not found: $LATEST_MODEL" >&2
  exit 1
fi

GLOBAL_VISIBLE="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "[big_train_v1] global CUDA_VISIBLE_DEVICES=$GLOBAL_VISIBLE"
if [[ "${CUDA_LAUNCH_BLOCKING:-0}" == "1" ]]; then
  echo "[big_train_v1] warning: CUDA_LAUNCH_BLOCKING=1 is debug-only and hurts throughput."
  if [[ -z "${V1_FINALIZE_GRAPH:-}" ]]; then
    export V1_FINALIZE_GRAPH=off
    echo "[big_train_v1] set V1_FINALIZE_GRAPH=off for CUDA_LAUNCH_BLOCKING compatibility."
  fi
fi

for ((it = 1; it <= ITERATIONS; it++)); do
  ITER_TAG="$(printf "%03d" "$it")"
  SELFPLAY_FILE="${RUN_DIR}/selfplay_iter_${ITER_TAG}.pt"
  SELFPLAY_STATS_JSON="${RUN_DIR}/selfplay_iter_${ITER_TAG}.json"
  TRAIN_METRICS_JSON="${RUN_DIR}/train_iter_${ITER_TAG}.json"
  INFER_JSON="${RUN_DIR}/infer_iter_${ITER_TAG}.json"
  CKPT_NAME="model_iter_${ITER_TAG}.pt"
  CKPT_PATH="${CHECKPOINT_DIR}/${CKPT_NAME}"

  echo
  echo "[big_train_v1] ===== Iteration ${it}/${ITERATIONS} ====="
  echo "[big_train_v1] stage=selfplay output=$SELFPLAY_FILE"
  SP_CMD=(
    "$PYTHON_BIN" scripts/train_entry.py
    --pipeline v1
    --stage selfplay
    --device "$DEVICE"
    --devices "$SELF_PLAY_DEVICES"
    --self_play_games "$SELF_PLAY_GAMES"
    --mcts_simulations "$MCTS_SIMULATIONS"
    --temperature_init "$TEMPERATURE_INIT"
    --temperature_final "$TEMPERATURE_FINAL"
    --temperature_threshold "$TEMPERATURE_THRESHOLD"
    --exploration_weight "$EXPLORATION_WEIGHT"
    --dirichlet_alpha "$DIRICHLET_ALPHA"
    --dirichlet_epsilon "$DIRICHLET_EPSILON"
    --soft_value_k "$SOFT_VALUE_K"
    --max_game_plies "$MAX_GAME_PLIES"
    --self_play_concurrent_games "$SELF_PLAY_CONCURRENT_GAMES"
    --self_play_backend "$SELF_PLAY_BACKEND"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --self_play_output "$SELFPLAY_FILE"
    --self_play_stats_json "$SELFPLAY_STATS_JSON"
  )
  if [[ -n "$SELF_PLAY_SHARD_DIR" ]]; then
    SP_CMD+=(--self_play_shard_dir "$SELF_PLAY_SHARD_DIR")
  fi
  if [[ -n "$LATEST_MODEL" && -f "$LATEST_MODEL" ]]; then
    SP_CMD+=(--load_checkpoint "$LATEST_MODEL")
  fi
  CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${SP_CMD[@]}"

  echo "[big_train_v1] stage=train input=$SELFPLAY_FILE strategy=$TRAIN_STRATEGY"
  if [[ "$TRAIN_STRATEGY" == "ddp" ]]; then
    TRAIN_VISIBLE="$(to_visible_indices "$TRAIN_DEVICES")"
    LOCAL_TRAIN_DEVICES="$(build_local_cuda_list "$TRAIN_NPROC")"
    DDP_CMD=(
      torchrun --standalone --nproc_per_node="$TRAIN_NPROC"
      scripts/train_entry.py
      --pipeline v1
      --stage train
      --device cuda:0
      --train_devices "$LOCAL_TRAIN_DEVICES"
      --train_strategy ddp
      --batch_size "$BATCH_SIZE"
      --epochs "$EPOCHS"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --checkpoint_dir "$CHECKPOINT_DIR"
      --self_play_input "$SELFPLAY_FILE"
      --checkpoint_name "$CKPT_NAME"
      --metrics_output "$TRAIN_METRICS_JSON"
    )
    if [[ -n "$LATEST_MODEL" && -f "$LATEST_MODEL" ]]; then
      DDP_CMD+=(--load_checkpoint "$LATEST_MODEL")
    fi
    CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE" "${DDP_CMD[@]}"
  else
    TRAIN_CMD=(
      "$PYTHON_BIN" scripts/train_entry.py
      --pipeline v1
      --stage train
      --device "$DEVICE"
      --train_devices "$TRAIN_DEVICES"
      --train_strategy "$TRAIN_STRATEGY"
      --batch_size "$BATCH_SIZE"
      --epochs "$EPOCHS"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --checkpoint_dir "$CHECKPOINT_DIR"
      --self_play_input "$SELFPLAY_FILE"
      --checkpoint_name "$CKPT_NAME"
      --metrics_output "$TRAIN_METRICS_JSON"
    )
    if [[ -n "$LATEST_MODEL" && -f "$LATEST_MODEL" ]]; then
      TRAIN_CMD+=(--load_checkpoint "$LATEST_MODEL")
    fi
    CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${TRAIN_CMD[@]}"
  fi

  LATEST_MODEL="$CKPT_PATH"
  if [[ ! -f "$LATEST_MODEL" ]]; then
    echo "[big_train_v1] expected checkpoint missing: $LATEST_MODEL" >&2
    exit 1
  fi

  if [[ "$RUN_INFER_STAGE" == "1" ]]; then
    echo "[big_train_v1] stage=infer checkpoint=$LATEST_MODEL"
    CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "$PYTHON_BIN" scripts/train_entry.py \
      --pipeline v1 \
      --stage infer \
      --device "$DEVICE" \
      --infer_devices "$INFER_DEVICES" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --load_checkpoint "$LATEST_MODEL" \
      --infer_batch_size "$INFER_BATCH_SIZE" \
      --infer_warmup_iters "$INFER_WARMUP_ITERS" \
      --infer_iters "$INFER_ITERS" \
      --infer_output "$INFER_JSON"
  fi

  echo "[big_train_v1] iteration ${it} done checkpoint=$LATEST_MODEL"
done

echo
echo "[big_train_v1] completed all iterations."
echo "[big_train_v1] final_checkpoint=$LATEST_MODEL"
echo "[big_train_v1] log_file=$LOG_FILE"
