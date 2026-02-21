#!/usr/bin/env bash
# High-load v1 staged training script (self-play -> train -> eval -> infer).
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
RUN_EVAL_STAGE="${RUN_EVAL_STAGE:-1}"

TEMPERATURE_INIT="${TEMPERATURE_INIT:-1.0}"
TEMPERATURE_FINAL="${TEMPERATURE_FINAL:-0.1}"
TEMPERATURE_THRESHOLD="${TEMPERATURE_THRESHOLD:-10}"
EXPLORATION_WEIGHT="${EXPLORATION_WEIGHT:-1.0}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.25}"
SOFT_VALUE_K="${SOFT_VALUE_K:-2.0}"
SOFT_LABEL_ALPHA="${SOFT_LABEL_ALPHA:-1.0}"
MAX_GAME_PLIES="${MAX_GAME_PLIES:-512}"
SELF_PLAY_CONCURRENT_GAMES="${SELF_PLAY_CONCURRENT_GAMES:-8192}"
SELF_PLAY_OPENING_RANDOM_MOVES="${SELF_PLAY_OPENING_RANDOM_MOVES:-6}"
SELF_PLAY_BACKEND="${SELF_PLAY_BACKEND:-process}" # auto | thread | process
SELF_PLAY_SHARD_DIR="${SELF_PLAY_SHARD_DIR:-}"

EVAL_GAMES_VS_RANDOM="${EVAL_GAMES_VS_RANDOM:-1000}"
EVAL_GAMES_VS_PREVIOUS="${EVAL_GAMES_VS_PREVIOUS:-1000}"
EVAL_MCTS_SIMULATIONS="${EVAL_MCTS_SIMULATIONS:-1024}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.05}"
EVAL_BACKEND="${EVAL_BACKEND:-v1}" # v0 | legacy | v1
EVAL_BATCH_LEAVES="${EVAL_BATCH_LEAVES:-1024}"
EVAL_INFER_BACKEND="${EVAL_INFER_BACKEND:-graph}"
EVAL_INFER_BATCH_SIZE="${EVAL_INFER_BATCH_SIZE:-1024}"
EVAL_INFER_WARMUP_ITERS="${EVAL_INFER_WARMUP_ITERS:-5}"
EVAL_SAMPLE_MOVES="${EVAL_SAMPLE_MOVES:-1}" # 0 | 1
EVAL_DEVICES="${EVAL_DEVICES:-$INFER_DEVICES}"
EVAL_V1_CONCURRENT_GAMES="${EVAL_V1_CONCURRENT_GAMES:-8192}"
EVAL_V1_OPENING_RANDOM_MOVES="${EVAL_V1_OPENING_RANDOM_MOVES:-6}"

SELF_PLAY_ALLOC_CONF="${SELF_PLAY_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:512}"
SELF_PLAY_MEMORY_ANCHOR_MB="${SELF_PLAY_MEMORY_ANCHOR_MB:-0}"

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
  : "${SELF_PLAY_GAMES:=2089952}"
  : "${MCTS_SIMULATIONS:=1024}"
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

if [[ -z "${EVAL_WORKERS:-}" ]]; then
  EVAL_WORKERS="$(csv_count "$EVAL_DEVICES")"
fi

TRAIN_NPROC="$(csv_count "$TRAIN_DEVICES")"
if [[ "$TRAIN_STRATEGY" == "ddp" && "$TRAIN_NPROC" -le 1 ]]; then
  echo "[big_train_v1] TRAIN_STRATEGY=ddp needs >1 gpu; fallback to data_parallel"
  TRAIN_STRATEGY="data_parallel"
fi
TRAIN_TOTAL_CPU_THREADS="${TRAIN_TOTAL_CPU_THREADS:-200}"
TRAIN_THREADS_PER_RANK="$TRAIN_TOTAL_CPU_THREADS"
if [[ "$TRAIN_STRATEGY" == "ddp" && "$TRAIN_NPROC" -gt 0 ]]; then
  TRAIN_THREADS_PER_RANK="$(( TRAIN_TOTAL_CPU_THREADS / TRAIN_NPROC ))"
fi
if [[ "$TRAIN_THREADS_PER_RANK" -lt 1 ]]; then
  TRAIN_THREADS_PER_RANK=1
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "$CHECKPOINT_DIR" "$RUN_DIR" logs
LOG_FILE="logs/big_train_v1_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[big_train_v1] run_tag=$RUN_TAG"
echo "[big_train_v1] profile=$PROFILE train_strategy=$TRAIN_STRATEGY"
echo "[big_train_v1] train_total_cpu_threads=$TRAIN_TOTAL_CPU_THREADS train_threads_per_rank=$TRAIN_THREADS_PER_RANK"
echo "[big_train_v1] python=$PYTHON_BIN"
echo "[big_train_v1] self_play_devices=$SELF_PLAY_DEVICES"
echo "[big_train_v1] train_devices=$TRAIN_DEVICES"
echo "[big_train_v1] infer_devices=$INFER_DEVICES"
echo "[big_train_v1] checkpoints=$CHECKPOINT_DIR"
echo "[big_train_v1] run_dir=$RUN_DIR"
echo "[big_train_v1] self_play_concurrent_games=$SELF_PLAY_CONCURRENT_GAMES"
echo "[big_train_v1] self_play_opening_random_moves=$SELF_PLAY_OPENING_RANDOM_MOVES"
echo "[big_train_v1] soft_label_alpha=$SOFT_LABEL_ALPHA"
echo "[big_train_v1] self_play_backend=$SELF_PLAY_BACKEND"
if [[ -n "$SELF_PLAY_SHARD_DIR" ]]; then
  echo "[big_train_v1] self_play_shard_dir=$SELF_PLAY_SHARD_DIR"
fi
echo "[big_train_v1] run_eval_stage=$RUN_EVAL_STAGE eval_backend=$EVAL_BACKEND"
echo "[big_train_v1] eval_games_vs_random=$EVAL_GAMES_VS_RANDOM eval_games_vs_previous=$EVAL_GAMES_VS_PREVIOUS"
echo "[big_train_v1] eval_devices=$EVAL_DEVICES eval_workers=$EVAL_WORKERS eval_mcts_sims=$EVAL_MCTS_SIMULATIONS"
echo "[big_train_v1] eval_v1_concurrent_games=$EVAL_V1_CONCURRENT_GAMES"
echo "[big_train_v1] eval_v1_opening_random_moves=$EVAL_V1_OPENING_RANDOM_MOVES eval_sample_moves=$EVAL_SAMPLE_MOVES"
echo "[big_train_v1] selfplay_alloc_conf=$SELF_PLAY_ALLOC_CONF selfplay_memory_anchor_mb=$SELF_PLAY_MEMORY_ANCHOR_MB"

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
  EVAL_JSON="${RUN_DIR}/eval_iter_${ITER_TAG}.json"
  INFER_JSON="${RUN_DIR}/infer_iter_${ITER_TAG}.json"
  CKPT_NAME="model_iter_${ITER_TAG}.pt"
  CKPT_PATH="${CHECKPOINT_DIR}/${CKPT_NAME}"
  PREV_MODEL_FOR_EVAL="$LATEST_MODEL"

  echo
  echo "[big_train_v1] ===== Iteration ${it}/${ITERATIONS} ====="
  echo "[big_train_v1] stage=selfplay output=$SELFPLAY_FILE"
  SP_CMD=(
    "$PYTHON_BIN" scripts/train_entry.py
    --pipeline v1
    --stage selfplay
    --device "$DEVICE"
    --devices "$SELF_PLAY_DEVICES"
    --train_devices "$TRAIN_DEVICES"
    --infer_devices "$INFER_DEVICES"
    --self_play_games "$SELF_PLAY_GAMES"
    --mcts_simulations "$MCTS_SIMULATIONS"
    --temperature_init "$TEMPERATURE_INIT"
    --temperature_final "$TEMPERATURE_FINAL"
    --temperature_threshold "$TEMPERATURE_THRESHOLD"
    --exploration_weight "$EXPLORATION_WEIGHT"
    --dirichlet_alpha "$DIRICHLET_ALPHA"
    --dirichlet_epsilon "$DIRICHLET_EPSILON"
    --soft_value_k "$SOFT_VALUE_K"
    --soft_label_alpha "$SOFT_LABEL_ALPHA"
    --max_game_plies "$MAX_GAME_PLIES"
    --self_play_concurrent_games "$SELF_PLAY_CONCURRENT_GAMES"
    --self_play_opening_random_moves "$SELF_PLAY_OPENING_RANDOM_MOVES"
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
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$SELF_PLAY_ALLOC_CONF}" \
  V1_SELFPLAY_MEMORY_ANCHOR_MB="$SELF_PLAY_MEMORY_ANCHOR_MB" \
  CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${SP_CMD[@]}"

  if [[ -f "$SELFPLAY_STATS_JSON" ]]; then
    "$PYTHON_BIN" - "$SELFPLAY_STATS_JSON" <<'PY'
import json
import sys
path = str(sys.argv[1])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
black = int(data.get("black_wins", 0))
white = int(data.get("white_wins", 0))
draws = int(data.get("draws", 0))
games = int(data.get("num_games", black + white + draws))
decisive = black + white
decisive_ratio = (decisive / games) if games > 0 else 0.0
draw_ratio = (draws / games) if games > 0 else 0.0
print(
    "[big_train_v1] selfplay outcomes: "
    f"W-L-D(black-win/white-win/draw)={black}-{white}-{draws}, "
    f"decisive_games={decisive}/{games} ({decisive_ratio*100.0:.2f}%), "
    f"draw_rate={draw_ratio*100.0:.2f}%"
)
vsum = data.get("value_target_summary")
if isinstance(vsum, dict):
    nonzero = int(vsum.get("nonzero_count", 0))
    total = int(vsum.get("total", 0))
    pos = int(vsum.get("positive_count", 0))
    neg = int(vsum.get("negative_count", 0))
    nonzero_ratio = float(vsum.get("nonzero_ratio", 0.0))
    print(
        "[big_train_v1] selfplay value targets: "
        f"nonzero={nonzero}/{total} ({nonzero_ratio*100.0:.2f}%), "
        f"positive={pos}, negative={neg}"
    )
mvsum = data.get("mixed_value_target_summary")
if isinstance(mvsum, dict):
    m_nonzero = int(mvsum.get("nonzero_count", 0))
    m_total = int(mvsum.get("total", 0))
    m_pos = int(mvsum.get("positive_count", 0))
    m_neg = int(mvsum.get("negative_count", 0))
    m_nonzero_ratio = float(mvsum.get("nonzero_ratio", 0.0))
    print(
        "[big_train_v1] selfplay mixed value targets: "
        f"nonzero={m_nonzero}/{m_total} ({m_nonzero_ratio*100.0:.2f}%), "
        f"positive={m_pos}, negative={m_neg}"
    )
piece_delta = data.get("piece_delta_buckets")
if isinstance(piece_delta, dict):
    bucket_total = 0
    nonzero_tokens = []
    for delta in range(-18, 19):
        key = str(delta)
        count = int(piece_delta.get(key, 0) or 0)
        bucket_total += count
        if count > 0:
            nonzero_tokens.append(f"{delta}:{count}")
    token_text = ",".join(nonzero_tokens) if nonzero_tokens else "none"
    print(
        "[big_train_v1] selfplay piece_delta buckets: "
        f"total={bucket_total}/{games}, nonzero={{{token_text}}}"
    )
    if bucket_total != games:
        print(
            "[big_train_v1] warning: piece_delta bucket total mismatch "
            f"total={bucket_total} expected={games}"
        )
PY
  fi

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
      --devices "$LOCAL_TRAIN_DEVICES"
      --infer_devices "$LOCAL_TRAIN_DEVICES"
      --train_devices "$LOCAL_TRAIN_DEVICES"
      --train_strategy ddp
      --batch_size "$BATCH_SIZE"
      --epochs "$EPOCHS"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --soft_label_alpha "$SOFT_LABEL_ALPHA"
      --checkpoint_dir "$CHECKPOINT_DIR"
      --self_play_input "$SELFPLAY_FILE"
      --checkpoint_name "$CKPT_NAME"
      --metrics_output "$TRAIN_METRICS_JSON"
    )
    if [[ -n "$LATEST_MODEL" && -f "$LATEST_MODEL" ]]; then
      DDP_CMD+=(--load_checkpoint "$LATEST_MODEL")
    fi
    OMP_NUM_THREADS="$TRAIN_THREADS_PER_RANK" \
    MKL_NUM_THREADS="$TRAIN_THREADS_PER_RANK" \
    OPENBLAS_NUM_THREADS="$TRAIN_THREADS_PER_RANK" \
    NUMEXPR_NUM_THREADS="$TRAIN_THREADS_PER_RANK" \
    CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE" "${DDP_CMD[@]}"
  else
    TRAIN_CMD=(
      "$PYTHON_BIN" scripts/train_entry.py
      --pipeline v1
      --stage train
      --device "$DEVICE"
      --devices "$TRAIN_DEVICES"
      --infer_devices "$INFER_DEVICES"
      --train_devices "$TRAIN_DEVICES"
      --train_strategy "$TRAIN_STRATEGY"
      --batch_size "$BATCH_SIZE"
      --epochs "$EPOCHS"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --soft_label_alpha "$SOFT_LABEL_ALPHA"
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

  if [[ "$RUN_EVAL_STAGE" == "1" ]]; then
    echo "[big_train_v1] stage=eval checkpoint=$LATEST_MODEL"
    EVAL_CMD=(
      "$PYTHON_BIN" scripts/eval_checkpoint.py
      --challenger_checkpoint "$LATEST_MODEL"
      --device "$DEVICE"
      --eval_devices "$EVAL_DEVICES"
      --eval_workers "$EVAL_WORKERS"
      --backend "$EVAL_BACKEND"
      --mcts_simulations "$EVAL_MCTS_SIMULATIONS"
      --temperature "$EVAL_TEMPERATURE"
      --eval_games_vs_random "$EVAL_GAMES_VS_RANDOM"
      --eval_games_vs_previous "$EVAL_GAMES_VS_PREVIOUS"
      --batch_leaves "$EVAL_BATCH_LEAVES"
      --inference_backend "$EVAL_INFER_BACKEND"
      --inference_batch_size "$EVAL_INFER_BATCH_SIZE"
      --inference_warmup_iters "$EVAL_INFER_WARMUP_ITERS"
      --v1_concurrent_games "$EVAL_V1_CONCURRENT_GAMES"
      --v1_opening_random_moves "$EVAL_V1_OPENING_RANDOM_MOVES"
      --output_json "$EVAL_JSON"
    )
    if [[ "$EVAL_SAMPLE_MOVES" == "1" ]]; then
      EVAL_CMD+=(--sample_moves)
    fi
    if [[ -n "$PREV_MODEL_FOR_EVAL" && -f "$PREV_MODEL_FOR_EVAL" ]]; then
      EVAL_CMD+=(--previous_checkpoint "$PREV_MODEL_FOR_EVAL")
    fi
    CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${EVAL_CMD[@]}"
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
