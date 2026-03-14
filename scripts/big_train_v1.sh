#!/usr/bin/env bash
# High-load v1 staged training script (self-play -> train -> eval -> infer).
set -euo pipefail

export PYTHONPATH="./:./build/v0/src:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PROFILE="${PROFILE:-aggressive}" # stable | aggressive
TRAIN_STRATEGY="${TRAIN_STRATEGY:-ddp}" # ddp | data_parallel | none
DEVICE="${DEVICE:-cuda:0}"

SELF_PLAY_DEVICES="${SELF_PLAY_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
TRAIN_DEVICES="${TRAIN_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
INFER_DEVICES="${INFER_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_v1_big_1}"
RUN_ROOT="${RUN_ROOT:-./v1/data/stage_runs}"
RUN_INFER_STAGE="${RUN_INFER_STAGE:-1}"
RUN_EVAL_STAGE="${RUN_EVAL_STAGE:-1}"
LEGACY_LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
TRAIN_BASE_MODEL="${TRAIN_BASE_MODEL:-}"
BEST_PREVIOUS_MODEL="${BEST_PREVIOUS_MODEL:-}"
MODEL_INIT_SEED="${MODEL_INIT_SEED:-20260314}"

TEMPERATURE_INIT="${TEMPERATURE_INIT:-1.0}"
TEMPERATURE_FINAL="${TEMPERATURE_FINAL:-0.1}"
TEMPERATURE_THRESHOLD="${TEMPERATURE_THRESHOLD:-10}"
EXPLORATION_WEIGHT="${EXPLORATION_WEIGHT:-1.0}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.25}"
SOFT_VALUE_K="${SOFT_VALUE_K:-2.0}"
SOFT_LABEL_ALPHA="${SOFT_LABEL_ALPHA:-1.0}"
SOFT_LABEL_ALPHA_FINAL="${SOFT_LABEL_ALPHA_FINAL:-1.0}"
ANTI_DRAW_PENALTY="${ANTI_DRAW_PENALTY:-0.0}"
POLICY_DRAW_WEIGHT="${POLICY_DRAW_WEIGHT:-1.0}"
POLICY_DRAW_WEIGHT_FINAL="${POLICY_DRAW_WEIGHT_FINAL:-1.0}"
MAX_GAME_PLIES="${MAX_GAME_PLIES:-512}"
SELF_PLAY_CONCURRENT_GAMES="${SELF_PLAY_CONCURRENT_GAMES:-8192}"
SELF_PLAY_OPENING_RANDOM_MOVES="${SELF_PLAY_OPENING_RANDOM_MOVES:-6}"
SELF_PLAY_OPENING_RANDOM_MOVES_FINAL="${SELF_PLAY_OPENING_RANDOM_MOVES_FINAL:-0}"
REPLAY_WINDOW="${REPLAY_WINDOW:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SELF_PLAY_BACKEND="${SELF_PLAY_BACKEND:-process}" # auto | thread | process
SELF_PLAY_SHARD_DIR="${SELF_PLAY_SHARD_DIR:-}"
SELF_PLAY_TARGET_SAMPLES_PER_SHARD="${SELF_PLAY_TARGET_SAMPLES_PER_SHARD:-0}"  # kept for fallback; overridden by CHUNK_TARGET_BYTES when set
CHUNK_TARGET_BYTES="${CHUNK_TARGET_BYTES:-8589934592}"  # 8 GiB per chunk; staged process self-play always emits manifest output
STREAMING_LOAD="${STREAMING_LOAD:-1}"
STREAMING_WORKERS="${STREAMING_WORKERS:-4}"
OPTIMIZER_STATE_WORK_PATH="${OPTIMIZER_STATE_WORK_PATH:-$CHECKPOINT_DIR/optimizer_state_work.pt}"

EVAL_GAMES_VS_BASELINE="${EVAL_GAMES_VS_BASELINE:-0}"
EVAL_GAMES_VS_SELF="${EVAL_GAMES_VS_SELF:-0}"
EVAL_BASELINE_CHECKPOINT="${EVAL_BASELINE_CHECKPOINT:-}"

EVAL_GAMES_VS_RANDOM="${EVAL_GAMES_VS_RANDOM:-2000}"
EVAL_GAMES_VS_PREVIOUS="${EVAL_GAMES_VS_PREVIOUS:-2000}"
EVAL_MCTS_SIMULATIONS="${EVAL_MCTS_SIMULATIONS:-1024}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.05}"
EVAL_BACKEND="${EVAL_BACKEND:-v1}" # v0 | legacy | v1
EVAL_BATCH_LEAVES="${EVAL_BATCH_LEAVES:-1024}"
EVAL_INFER_BACKEND="${EVAL_INFER_BACKEND:-graph}"
EVAL_INFER_BATCH_SIZE="${EVAL_INFER_BATCH_SIZE:-1024}"
EVAL_INFER_WARMUP_ITERS="${EVAL_INFER_WARMUP_ITERS:-5}"
EVAL_SAMPLE_MOVES="${EVAL_SAMPLE_MOVES:-0}" # 0 | 1
EVAL_DEVICES="${EVAL_DEVICES:-$INFER_DEVICES}"
EVAL_V1_CONCURRENT_GAMES="${EVAL_V1_CONCURRENT_GAMES:-8192}"
EVAL_V1_OPENING_RANDOM_MOVES="${EVAL_V1_OPENING_RANDOM_MOVES:-0}" # strict eval default: no random opening moves
EVAL_VS_RANDOM_TEMPERATURE="${EVAL_VS_RANDOM_TEMPERATURE:-0.0}"
EVAL_VS_RANDOM_V1_OPENING_RANDOM_MOVES="${EVAL_VS_RANDOM_V1_OPENING_RANDOM_MOVES:-0}"
EVAL_VS_PREVIOUS_TEMPERATURE="${EVAL_VS_PREVIOUS_TEMPERATURE:-1.0}"
EVAL_VS_PREVIOUS_SAMPLE_MOVES="${EVAL_VS_PREVIOUS_SAMPLE_MOVES:-1}" # 0 | 1
EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES="${EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES:-0}"

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
  : "${SELF_PLAY_GAMES:=522488}"
  : "${MCTS_SIMULATIONS:=65536}"
  : "${BATCH_SIZE:=16384}"
  : "${EPOCHS:=4}"
  : "${LR:=1e-4}"
  : "${WEIGHT_DECAY:=1e-4}"
else
  printf '[big_train_v1] unsupported PROFILE=%s (use stable/aggressive)\n' "$PROFILE" >&2
  exit 1
fi

: "${LR_COSINE_FINAL_SCALE:=0.5}"
: "${INFER_BATCH_SIZE:=4096}"
: "${INFER_WARMUP_ITERS:=20}"
: "${INFER_ITERS:=80}"

compute_curriculum_values() {
  local iter_idx="$1"
  "$PYTHON_BIN" - "$iter_idx" "$ITERATIONS" "$SELF_PLAY_OPENING_RANDOM_MOVES" "$SELF_PLAY_OPENING_RANDOM_MOVES_FINAL" "$SOFT_LABEL_ALPHA" "$SOFT_LABEL_ALPHA_FINAL" "$LR" "$POLICY_DRAW_WEIGHT" <<'PY'
import sys

it = int(sys.argv[1])
total = max(1, int(sys.argv[2]))
opening_start = float(sys.argv[3])
opening_final = float(sys.argv[4])
alpha_start = float(sys.argv[5])
alpha_final = float(sys.argv[6])
lr_now = float(sys.argv[7])
pdw_now = float(sys.argv[8])

if total <= 1:
    progress = 1.0
else:
    progress = (it - 1) / float(total - 1)
progress = max(0.0, min(1.0, progress))

opening_now = int(round(opening_start + (opening_final - opening_start) * progress))
opening_now = max(0, opening_now)

alpha_now = alpha_start + (alpha_final - alpha_start) * progress
alpha_now = max(0.0, min(1.0, alpha_now))

print(f"{opening_now} {alpha_now:.6f} {lr_now:.12g} {pdw_now:.6f}")
PY
}

check_train_metrics_finite() {
  local metrics_path="$1"
  local output=""
  local rc=0
  output="$("$PYTHON_BIN" - "$metrics_path" <<'PY'
import json
import math
import re
import sys

path = str(sys.argv[1])
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as exc:
    print(f"[big_train_v1] warning: failed to load train metrics: {exc}")
    raise SystemExit(1)

row = None
if isinstance(data, list) and data:
    row = data[-1]
elif isinstance(data, dict):
    row = data

if not isinstance(row, dict):
    print("[big_train_v1] warning: train metrics payload missing train row.")
    raise SystemExit(1)

required_keys = ("train_avg_loss", "train_avg_policy_loss", "train_avg_value_loss")
required_invalid = []
for key in required_keys:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        required_invalid.append((key, value))

non_finite_numeric = []
for key, value in row.items():
    if isinstance(value, bool):
        continue
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        non_finite_numeric.append((str(key), value))

if required_invalid or non_finite_numeric:
    iter_tag = "unknown"
    match = re.search(r"train_iter_(\d+)\.json$", path)
    if match:
        iter_tag = match.group(1)
    checkpoint = row.get("checkpoint", "<missing>")
    print("[big_train_v1] warning: non-finite training metrics detected; mark candidate invalid.")
    print(
        "[big_train_v1] invalid_metrics_context: "
        f"metrics_path={path!r} iter_tag={iter_tag} checkpoint={checkpoint!r}"
    )
    for key, value in required_invalid:
        print(f"[big_train_v1] invalid_required_metric {key}={value!r}")
    for key, value in sorted(non_finite_numeric):
        print(f"[big_train_v1] non_finite_metric {key}={value!r}")
    context_keys = (
        "self_play_value_nonfinite_local",
        "self_play_soft_value_nonfinite_local",
        "self_play_positions_local",
        "self_play_payload_load_sec",
        "checkpoint_load_sec",
        "train_first_batch_sec",
        "train_time_sec",
        "train_strategy",
        "streaming",
    )
    context_tokens = []
    for key in context_keys:
        if key in row:
            context_tokens.append(f"{key}={row.get(key)!r}")
    if context_tokens:
        print("[big_train_v1] invalid_metrics_snapshot: " + ", ".join(context_tokens))
    raise SystemExit(1)

raise SystemExit(0)
PY
  )" || rc=$?
  if [[ -n "$output" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      log "$line"
    done <<< "$output"
  fi
  return "$rc"
}

gating_accept_candidate() {
  local eval_path="$1"
  local output=""
  local rc=0
  output="$("$PYTHON_BIN" - "$eval_path" <<'PY'
import json
import sys

path = str(sys.argv[1])
try:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
except Exception as exc:
    print(f"[big_train_v1] warning: failed to load eval report for gating: {exc}")
    raise SystemExit(1)

rows = report.get("results")
if not isinstance(rows, list):
    print("[big_train_v1] warning: eval report has no results list for gating.")
    raise SystemExit(1)

target = None
for row in rows:
    if isinstance(row, dict) and str(row.get("name", "")).strip() == "vs_previous":
        target = row
        break

if not isinstance(target, dict):
    print("[big_train_v1] warning: gating requires vs_previous stats, but none were found.")
    raise SystemExit(1)

wins = int(target.get("wins", 0) or 0)
losses = int(target.get("losses", 0) or 0)
draws = int(target.get("draws", 0) or 0)
accept = wins > losses
decision = "accept" if accept else "reject"
print(
    "[big_train_v1] gating(vs_best_previous): "
    f"wins={wins} losses={losses} draws={draws} -> {decision}"
)
raise SystemExit(0 if accept else 1)
PY
  )" || rc=$?
  if [[ -n "$output" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      log "$line"
    done <<< "$output"
  fi
  return "$rc"
}

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

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "$CHECKPOINT_DIR" "$RUN_DIR" logs
LOG_FILE="logs/big_train_v1_${RUN_TAG}.log"
exec 9>>"$LOG_FILE"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  GIT_DIRTY=1
else
  GIT_DIRTY=0
fi

timestamp_utc() {
  date -u '+[%Y-%m-%d %H:%M:%S UTC]'
}

_emit_log_line() {
  local stream="$1"
  shift || true
  local ts
  ts="$(timestamp_utc)"
  if [[ "$#" -eq 0 ]]; then
    if [[ "$stream" == "stderr" ]]; then
      printf '%s\n' "$ts" >&2
    else
      printf '%s\n' "$ts"
    fi
    printf '%s\n' "$ts" >&9
    return 0
  fi
  if [[ "$stream" == "stderr" ]]; then
    printf '%s %s\n' "$ts" "$*" >&2
  else
    printf '%s %s\n' "$ts" "$*"
  fi
  printf '%s %s\n' "$ts" "$*" >&9
}

log() {
  _emit_log_line stdout "$@"
}

log_err() {
  _emit_log_line stderr "$@"
}

run_logged() {
  "$@" \
    > >(
      while IFS= read -r line || [[ -n "$line" ]]; do
        _emit_log_line stdout "$line"
      done
    ) \
    2> >(
      while IFS= read -r line || [[ -n "$line" ]]; do
        _emit_log_line stderr "$line"
      done
    )
}

STAGE_MAX_RETRIES="${STAGE_MAX_RETRIES:-2}"

run_with_retry() {
  local stage_label="$1"
  shift
  local max_retries="$STAGE_MAX_RETRIES"
  local attempt=0
  local rc=0
  while [[ "$attempt" -le "$max_retries" ]]; do
    attempt=$((attempt + 1))
    rc=0
    run_logged "$@" && return 0 || rc=$?
    log_err "[big_train_v1] $stage_label failed (exit_code=$rc, attempt=$attempt/$((max_retries + 1)))"
    log_err "[big_train_v1] diagnostics for $stage_label failure:"
    log_err "[big_train_v1]   date=$(date '+%Y-%m-%d %H:%M:%S')"
    run_logged bash -lc 'free -h 2>/dev/null | head -3' || true
    run_logged bash -lc 'dmesg -T 2>/dev/null | tail -20' || true
    if [[ "$attempt" -le "$max_retries" ]]; then
      local sleep_sec=$((attempt * 5))
      log_err "[big_train_v1] retrying $stage_label in ${sleep_sec}s ..."
      sleep "$sleep_sec"
      sync 2>/dev/null || true
      echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
  done
  log_err "[big_train_v1] $stage_label exhausted retries ($((max_retries + 1)) attempts), aborting"
  return "$rc"
}

if [[ -z "${EVAL_WORKERS:-}" ]]; then
  EVAL_WORKERS="$(csv_count "$EVAL_DEVICES")"
fi

TRAIN_NPROC="$(csv_count "$TRAIN_DEVICES")"
if [[ "$TRAIN_STRATEGY" == "ddp" && "$TRAIN_NPROC" -le 1 ]]; then
  log "[big_train_v1] TRAIN_STRATEGY=ddp needs >1 gpu; fallback to data_parallel"
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

if [[ -n "$LEGACY_LOAD_CHECKPOINT" && -z "$TRAIN_BASE_MODEL" ]]; then
  TRAIN_BASE_MODEL="$LEGACY_LOAD_CHECKPOINT"
fi

log "[big_train_v1] run_tag=$RUN_TAG"
log "[big_train_v1] git_branch=$GIT_BRANCH git_commit=$GIT_COMMIT git_dirty=$GIT_DIRTY"
log "[big_train_v1] profile=$PROFILE train_strategy=$TRAIN_STRATEGY"
log "[big_train_v1] train_total_cpu_threads=$TRAIN_TOTAL_CPU_THREADS train_threads_per_rank=$TRAIN_THREADS_PER_RANK"
log "[big_train_v1] python=$PYTHON_BIN"
log "[big_train_v1] self_play_devices=$SELF_PLAY_DEVICES"
log "[big_train_v1] train_devices=$TRAIN_DEVICES"
log "[big_train_v1] infer_devices=$INFER_DEVICES"
log "[big_train_v1] checkpoints=$CHECKPOINT_DIR"
log "[big_train_v1] run_dir=$RUN_DIR"
log "[big_train_v1] train_base_model=${TRAIN_BASE_MODEL:-none}"
log "[big_train_v1] best_previous_model_init=${BEST_PREVIOUS_MODEL:-none}"
log "[big_train_v1] model_init_seed=$MODEL_INIT_SEED (used when train_base_model=none)"
log "[big_train_v1] self_play_concurrent_games=$SELF_PLAY_CONCURRENT_GAMES"
log "[big_train_v1] self_play_opening_random_moves=$SELF_PLAY_OPENING_RANDOM_MOVES"
log "[big_train_v1] chunk_target_bytes=$CHUNK_TARGET_BYTES (self_play_target_samples_per_shard=$SELF_PLAY_TARGET_SAMPLES_PER_SHARD fallback)"
log "[big_train_v1] soft_label_alpha=$SOFT_LABEL_ALPHA anti_draw_penalty=$ANTI_DRAW_PENALTY"
log "[big_train_v1] soft_value_k=$SOFT_VALUE_K"
log "[big_train_v1] policy_draw_weight=$POLICY_DRAW_WEIGHT->$POLICY_DRAW_WEIGHT_FINAL"
log "[big_train_v1] lr=$LR lr_cosine_final_scale=$LR_COSINE_FINAL_SCALE warmup_steps=$WARMUP_STEPS"
log "[big_train_v1] streaming_load=$STREAMING_LOAD streaming_workers=$STREAMING_WORKERS"
log "[big_train_v1] replay_window=$REPLAY_WINDOW"
log "[big_train_v1] optimizer_state_work_path=$OPTIMIZER_STATE_WORK_PATH"
log "[big_train_v1] opening_random_schedule=$SELF_PLAY_OPENING_RANDOM_MOVES->$SELF_PLAY_OPENING_RANDOM_MOVES_FINAL"
log "[big_train_v1] soft_label_alpha_schedule=$SOFT_LABEL_ALPHA->$SOFT_LABEL_ALPHA_FINAL"
log "[big_train_v1] self_play_backend=$SELF_PLAY_BACKEND"
if [[ -n "$SELF_PLAY_SHARD_DIR" ]]; then
  log "[big_train_v1] self_play_shard_dir=$SELF_PLAY_SHARD_DIR"
fi
log "[big_train_v1] run_eval_stage=$RUN_EVAL_STAGE eval_backend=$EVAL_BACKEND"
log "[big_train_v1] eval_games_vs_random=$EVAL_GAMES_VS_RANDOM eval_games_vs_previous=$EVAL_GAMES_VS_PREVIOUS"
log "[big_train_v1] eval_devices=$EVAL_DEVICES eval_workers=$EVAL_WORKERS eval_mcts_sims=$EVAL_MCTS_SIMULATIONS"
log "[big_train_v1] eval_v1_concurrent_games=$EVAL_V1_CONCURRENT_GAMES"
log "[big_train_v1] eval_v1_opening_random_moves=$EVAL_V1_OPENING_RANDOM_MOVES eval_sample_moves=$EVAL_SAMPLE_MOVES"
log "[big_train_v1] eval_vs_random temperature=$EVAL_VS_RANDOM_TEMPERATURE opening_random_moves=$EVAL_VS_RANDOM_V1_OPENING_RANDOM_MOVES"
log "[big_train_v1] eval_vs_previous temperature=$EVAL_VS_PREVIOUS_TEMPERATURE sample_moves=$EVAL_VS_PREVIOUS_SAMPLE_MOVES opening_random_moves=$EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES"
log "[big_train_v1] selfplay_alloc_conf=$SELF_PLAY_ALLOC_CONF selfplay_memory_anchor_mb=$SELF_PLAY_MEMORY_ANCHOR_MB"

LATEST_MODEL="$TRAIN_BASE_MODEL"
if [[ -n "$LATEST_MODEL" && ! -f "$LATEST_MODEL" ]]; then
  log_err "[big_train_v1] train base checkpoint not found: $LATEST_MODEL"
  exit 1
fi
BEST_MODEL="$BEST_PREVIOUS_MODEL"
if [[ -z "$BEST_MODEL" ]]; then
  BEST_MODEL="$LATEST_MODEL"
fi
if [[ -n "$BEST_MODEL" && ! -f "$BEST_MODEL" ]]; then
  log_err "[big_train_v1] best previous checkpoint not found: $BEST_MODEL"
  exit 1
fi

GLOBAL_VISIBLE="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
log "[big_train_v1] global CUDA_VISIBLE_DEVICES=$GLOBAL_VISIBLE"
if [[ "${CUDA_LAUNCH_BLOCKING:-0}" == "1" ]]; then
  log "[big_train_v1] warning: CUDA_LAUNCH_BLOCKING=1 is debug-only and hurts throughput."
  if [[ -z "${V1_FINALIZE_GRAPH:-}" ]]; then
    export V1_FINALIZE_GRAPH=off
    log "[big_train_v1] set V1_FINALIZE_GRAPH=off for CUDA_LAUNCH_BLOCKING compatibility."
  fi
fi

for ((it = 1; it <= ITERATIONS; it++)); do
  read -r CUR_OPENING_RANDOM_MOVES CUR_SOFT_LABEL_ALPHA CUR_LR CUR_POLICY_DRAW_WEIGHT <<< "$(compute_curriculum_values "$it")"
  ITER_TAG="$(printf "%03d" "$it")"
  SELFPLAY_FILE="${RUN_DIR}/selfplay_iter_${ITER_TAG}.pt"
  SELFPLAY_STATS_JSON="${RUN_DIR}/selfplay_iter_${ITER_TAG}.json"
  TRAIN_METRICS_JSON="${RUN_DIR}/train_iter_${ITER_TAG}.json"
  EVAL_JSON="${RUN_DIR}/eval_iter_${ITER_TAG}.json"
  EVAL_BASELINE_JSON="${RUN_DIR}/eval_baseline_iter_${ITER_TAG}.json"
  EVAL_SELF_JSON="${RUN_DIR}/eval_self_iter_${ITER_TAG}.json"
  INFER_JSON="${RUN_DIR}/infer_iter_${ITER_TAG}.json"
  CKPT_NAME="model_iter_${ITER_TAG}.pt"
  CKPT_PATH="${CHECKPOINT_DIR}/${CKPT_NAME}"
  BASE_MODEL_FOR_ITER="$LATEST_MODEL"
  GATING_BASE_MODEL="$BEST_MODEL"

  log
  log "[big_train_v1] ===== Iteration ${it}/${ITERATIONS} ====="
  log "[big_train_v1] curriculum opening_random_moves=$CUR_OPENING_RANDOM_MOVES soft_label_alpha=$CUR_SOFT_LABEL_ALPHA lr=$CUR_LR policy_draw_weight=$CUR_POLICY_DRAW_WEIGHT"
  log "[big_train_v1] base_model_for_iter=${BASE_MODEL_FOR_ITER:-none} gating_base_model=${GATING_BASE_MODEL:-none}"
  log "[big_train_v1] stage=selfplay output=$SELFPLAY_FILE"
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
    --soft_label_alpha "$CUR_SOFT_LABEL_ALPHA"
    --max_game_plies "$MAX_GAME_PLIES"
    --self_play_concurrent_games "$SELF_PLAY_CONCURRENT_GAMES"
    --self_play_opening_random_moves "$CUR_OPENING_RANDOM_MOVES"
    --self_play_backend "$SELF_PLAY_BACKEND"
    --self_play_target_samples_per_shard "$SELF_PLAY_TARGET_SAMPLES_PER_SHARD"
    --self_play_chunk_target_bytes "$CHUNK_TARGET_BYTES"
    --model_init_seed "$MODEL_INIT_SEED"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --self_play_output "$SELFPLAY_FILE"
    --self_play_iteration_seed "$it"
    --self_play_stats_json "$SELFPLAY_STATS_JSON"
  )
  if [[ -n "$SELF_PLAY_SHARD_DIR" ]]; then
    SP_CMD+=(--self_play_shard_dir "$SELF_PLAY_SHARD_DIR")
  fi
  if [[ -n "$BASE_MODEL_FOR_ITER" && -f "$BASE_MODEL_FOR_ITER" ]]; then
    SP_CMD+=(--load_checkpoint "$BASE_MODEL_FOR_ITER")
  fi
  run_with_retry "selfplay(iter=$it)" \
    env PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$SELF_PLAY_ALLOC_CONF}" \
    V1_SELFPLAY_MEMORY_ANCHOR_MB="$SELF_PLAY_MEMORY_ANCHOR_MB" \
    CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${SP_CMD[@]}"

  if [[ -f "$SELFPLAY_STATS_JSON" ]]; then
    run_logged "$PYTHON_BIN" - "$SELFPLAY_STATS_JSON" <<'PY'
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

  log "[big_train_v1] stage=train input=$SELFPLAY_FILE strategy=$TRAIN_STRATEGY"
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
      --lr "$CUR_LR"
      --weight_decay "$WEIGHT_DECAY"
      --soft_label_alpha "$CUR_SOFT_LABEL_ALPHA"
      --warmup_steps "$WARMUP_STEPS"
      --checkpoint_dir "$CHECKPOINT_DIR"
      --self_play_input "$SELFPLAY_FILE"
      --streaming_load "$STREAMING_LOAD"
      --streaming_workers "$STREAMING_WORKERS"
      --optimizer_state_path "$OPTIMIZER_STATE_WORK_PATH"
      --model_init_seed "$MODEL_INIT_SEED"
      --checkpoint_name "$CKPT_NAME"
      --metrics_output "$TRAIN_METRICS_JSON"
    )
    if [[ -n "$BASE_MODEL_FOR_ITER" && -f "$BASE_MODEL_FOR_ITER" ]]; then
      DDP_CMD+=(--load_checkpoint "$BASE_MODEL_FOR_ITER")
    fi
    run_with_retry "train-ddp(iter=$it)" \
      env OMP_NUM_THREADS="$TRAIN_THREADS_PER_RANK" \
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
      --lr "$CUR_LR"
      --weight_decay "$WEIGHT_DECAY"
      --soft_label_alpha "$CUR_SOFT_LABEL_ALPHA"
      --warmup_steps "$WARMUP_STEPS"
      --checkpoint_dir "$CHECKPOINT_DIR"
      --self_play_input "$SELFPLAY_FILE"
      --streaming_load "$STREAMING_LOAD"
      --streaming_workers "$STREAMING_WORKERS"
      --optimizer_state_path "$OPTIMIZER_STATE_WORK_PATH"
      --model_init_seed "$MODEL_INIT_SEED"
      --checkpoint_name "$CKPT_NAME"
      --metrics_output "$TRAIN_METRICS_JSON"
    )
    if [[ -n "$BASE_MODEL_FOR_ITER" && -f "$BASE_MODEL_FOR_ITER" ]]; then
      TRAIN_CMD+=(--load_checkpoint "$BASE_MODEL_FOR_ITER")
    fi
    run_with_retry "train(iter=$it)" \
      env CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${TRAIN_CMD[@]}"
  fi

  CANDIDATE_MODEL="$CKPT_PATH"
  if [[ ! -f "$CANDIDATE_MODEL" ]]; then
    log_err "[big_train_v1] expected checkpoint missing: $CANDIDATE_MODEL"
    exit 1
  fi
  CANDIDATE_VALID=1
  if [[ -f "$TRAIN_METRICS_JSON" ]]; then
    if ! check_train_metrics_finite "$TRAIN_METRICS_JSON"; then
      CANDIDATE_VALID=0
    fi
  else
    log "[big_train_v1] warning: training metrics not found, mark candidate invalid."
    CANDIDATE_VALID=0
  fi
  if [[ "$CANDIDATE_VALID" != "1" && -f "$OPTIMIZER_STATE_WORK_PATH" ]]; then
    log "[big_train_v1] drop optimizer continuity state after invalid candidate: $OPTIMIZER_STATE_WORK_PATH"
    rm -f "$OPTIMIZER_STATE_WORK_PATH" || true
  fi

  CANDIDATE_ACCEPTED=0

  if [[ "$RUN_EVAL_STAGE" == "1" && "$CANDIDATE_VALID" == "1" ]]; then
    if [[ "$EVAL_GAMES_VS_RANDOM" -gt 0 ]]; then
      log "[big_train_v1] stage=eval_vs_random checkpoint=$CANDIDATE_MODEL"
      EVAL_RANDOM_CMD=(
        "$PYTHON_BIN" scripts/eval_checkpoint.py
        --challenger_checkpoint "$CANDIDATE_MODEL"
        --match_name vs_random
        --device "$DEVICE"
        --eval_devices "$EVAL_DEVICES"
        --eval_workers "$EVAL_WORKERS"
        --backend "$EVAL_BACKEND"
        --mcts_simulations "$EVAL_MCTS_SIMULATIONS"
        --temperature "$EVAL_VS_RANDOM_TEMPERATURE"
        --eval_games_vs_random "$EVAL_GAMES_VS_RANDOM"
        --eval_games_vs_previous 0
        --batch_leaves "$EVAL_BATCH_LEAVES"
        --inference_backend "$EVAL_INFER_BACKEND"
        --inference_batch_size "$EVAL_INFER_BATCH_SIZE"
        --inference_warmup_iters "$EVAL_INFER_WARMUP_ITERS"
        --v1_concurrent_games "$EVAL_V1_CONCURRENT_GAMES"
        --v1_opening_random_moves "$EVAL_VS_RANDOM_V1_OPENING_RANDOM_MOVES"
      )
      run_logged env CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${EVAL_RANDOM_CMD[@]}"
    fi

    if [[ -n "$GATING_BASE_MODEL" && -f "$GATING_BASE_MODEL" ]]; then
      EVAL_GAMES_VS_PREVIOUS_FOR_ITER="$EVAL_GAMES_VS_PREVIOUS"
      if [[ "$EVAL_GAMES_VS_PREVIOUS_FOR_ITER" -lt 2 ]]; then
        EVAL_GAMES_VS_PREVIOUS_FOR_ITER=2
        log "[big_train_v1] gating requires vs_best_previous games; bump eval_games_vs_previous to 2 for this iteration."
      fi
      log "[big_train_v1] stage=eval_vs_previous checkpoint=$CANDIDATE_MODEL best_previous=$GATING_BASE_MODEL"
      EVAL_PREV_CMD=(
        "$PYTHON_BIN" scripts/eval_checkpoint.py
        --challenger_checkpoint "$CANDIDATE_MODEL"
        --previous_checkpoint "$GATING_BASE_MODEL"
        --match_name vs_previous
        --device "$DEVICE"
        --eval_devices "$EVAL_DEVICES"
        --eval_workers "$EVAL_WORKERS"
        --backend "$EVAL_BACKEND"
        --mcts_simulations "$EVAL_MCTS_SIMULATIONS"
        --temperature "$EVAL_VS_PREVIOUS_TEMPERATURE"
        --eval_games_vs_random 0
        --eval_games_vs_previous "$EVAL_GAMES_VS_PREVIOUS_FOR_ITER"
        --batch_leaves "$EVAL_BATCH_LEAVES"
        --inference_backend "$EVAL_INFER_BACKEND"
        --inference_batch_size "$EVAL_INFER_BATCH_SIZE"
        --inference_warmup_iters "$EVAL_INFER_WARMUP_ITERS"
        --v1_concurrent_games "$EVAL_V1_CONCURRENT_GAMES"
        --v1_opening_random_moves "$EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES"
        --output_json "$EVAL_JSON"
      )
      if [[ "$EVAL_VS_PREVIOUS_SAMPLE_MOVES" == "1" ]]; then
        EVAL_PREV_CMD+=(--sample_moves)
      fi
      run_logged env CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${EVAL_PREV_CMD[@]}"
    fi

    # Extra eval: vs fixed baseline (iter_001 or user-specified anchor).
    BASELINE_CKPT="${EVAL_BASELINE_CHECKPOINT}"
    if [[ -z "$BASELINE_CKPT" ]]; then
      BASELINE_CKPT="${CHECKPOINT_DIR}/model_iter_001.pt"
    fi
    if [[ "$EVAL_GAMES_VS_BASELINE" -gt 0 && -f "$BASELINE_CKPT" && "$BASELINE_CKPT" != "$CANDIDATE_MODEL" ]]; then
      log "[big_train_v1] stage=eval_vs_baseline checkpoint=$CANDIDATE_MODEL baseline=$BASELINE_CKPT"
      EVAL_BASE_CMD=(
        "$PYTHON_BIN" scripts/eval_checkpoint.py
        --challenger_checkpoint "$CANDIDATE_MODEL"
        --previous_checkpoint "$BASELINE_CKPT"
        --match_name vs_baseline
        --device "$DEVICE"
        --eval_devices "$EVAL_DEVICES"
        --eval_workers "$EVAL_WORKERS"
        --backend "$EVAL_BACKEND"
        --mcts_simulations "$EVAL_MCTS_SIMULATIONS"
        --temperature "$EVAL_VS_PREVIOUS_TEMPERATURE"
        --eval_games_vs_random 0
        --eval_games_vs_previous "$EVAL_GAMES_VS_BASELINE"
        --batch_leaves "$EVAL_BATCH_LEAVES"
        --inference_backend "$EVAL_INFER_BACKEND"
        --inference_batch_size "$EVAL_INFER_BATCH_SIZE"
        --inference_warmup_iters "$EVAL_INFER_WARMUP_ITERS"
        --v1_concurrent_games "$EVAL_V1_CONCURRENT_GAMES"
        --v1_opening_random_moves "$EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES"
        --output_json "$EVAL_BASELINE_JSON"
      )
      if [[ "$EVAL_VS_PREVIOUS_SAMPLE_MOVES" == "1" ]]; then
        EVAL_BASE_CMD+=(--sample_moves)
      fi
      run_logged env CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${EVAL_BASE_CMD[@]}"
    fi

    # Extra eval: vs self (draw rate monitor).
    if [[ "$EVAL_GAMES_VS_SELF" -gt 0 ]]; then
      log "[big_train_v1] stage=eval_vs_self checkpoint=$CANDIDATE_MODEL"
      EVAL_SELF_CMD=(
        "$PYTHON_BIN" scripts/eval_checkpoint.py
        --challenger_checkpoint "$CANDIDATE_MODEL"
        --previous_checkpoint "$CANDIDATE_MODEL"
        --match_name vs_self
        --device "$DEVICE"
        --eval_devices "$EVAL_DEVICES"
        --eval_workers "$EVAL_WORKERS"
        --backend "$EVAL_BACKEND"
        --mcts_simulations "$EVAL_MCTS_SIMULATIONS"
        --temperature "$EVAL_VS_PREVIOUS_TEMPERATURE"
        --eval_games_vs_random 0
        --eval_games_vs_previous "$EVAL_GAMES_VS_SELF"
        --batch_leaves "$EVAL_BATCH_LEAVES"
        --inference_backend "$EVAL_INFER_BACKEND"
        --inference_batch_size "$EVAL_INFER_BATCH_SIZE"
        --inference_warmup_iters "$EVAL_INFER_WARMUP_ITERS"
        --v1_concurrent_games "$EVAL_V1_CONCURRENT_GAMES"
        --v1_opening_random_moves "$EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES"
        --output_json "$EVAL_SELF_JSON"
      )
      if [[ "$EVAL_VS_PREVIOUS_SAMPLE_MOVES" == "1" ]]; then
        EVAL_SELF_CMD+=(--sample_moves)
      fi
      run_logged env CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "${EVAL_SELF_CMD[@]}"
    fi
  fi

  if [[ "$CANDIDATE_VALID" != "1" ]]; then
    CANDIDATE_ACCEPTED=0
    log "[big_train_v1] candidate invalid (non-finite metrics); skip eval and keep best_previous frozen."
  elif [[ -z "$GATING_BASE_MODEL" || ! -f "$GATING_BASE_MODEL" ]]; then
    CANDIDATE_ACCEPTED=1
    log "[big_train_v1] bootstrap best_previous checkpoint selected: $CANDIDATE_MODEL"
  elif [[ "$RUN_EVAL_STAGE" != "1" ]]; then
    CANDIDATE_ACCEPTED=1
    log "[big_train_v1] warning: RUN_EVAL_STAGE=0, gating skipped; promote candidate by default."
  elif gating_accept_candidate "$EVAL_JSON"; then
    CANDIDATE_ACCEPTED=1
  else
    CANDIDATE_ACCEPTED=0
  fi

  if [[ "$CANDIDATE_ACCEPTED" == "1" ]]; then
    BEST_MODEL="$CANDIDATE_MODEL"
  elif [[ "$CANDIDATE_VALID" == "1" ]]; then
    log "[big_train_v1] candidate did not beat best_previous; keep training from latest candidate and retain best_previous for gating."
  else
    log "[big_train_v1] invalid candidate, best frozen, latest advanced."
  fi
  LATEST_MODEL="$CANDIDATE_MODEL"
  if [[ -z "$LATEST_MODEL" || ! -f "$LATEST_MODEL" ]]; then
    log_err "[big_train_v1] latest checkpoint missing after iteration: $LATEST_MODEL"
    exit 1
  fi
  if [[ -n "$BEST_MODEL" && ! -f "$BEST_MODEL" ]]; then
    log_err "[big_train_v1] best checkpoint missing after iteration: $BEST_MODEL"
    exit 1
  fi

  if [[ "$RUN_INFER_STAGE" == "1" ]]; then
    log "[big_train_v1] stage=infer checkpoint=$LATEST_MODEL"
    run_logged env CUDA_VISIBLE_DEVICES="$GLOBAL_VISIBLE" "$PYTHON_BIN" scripts/train_entry.py \
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

  log "[big_train_v1] iteration ${it} done candidate=$CANDIDATE_MODEL latest=$LATEST_MODEL best_previous=$BEST_MODEL"
done

log
log "[big_train_v1] completed all iterations."
log "[big_train_v1] final_checkpoint=$LATEST_MODEL"
log "[big_train_v1] final_best_previous_checkpoint=$BEST_MODEL"
log "[big_train_v1] log_file=$LOG_FILE"
