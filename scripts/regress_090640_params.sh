#!/usr/bin/env bash
set -euo pipefail

REPO=/2023533024/users/zhangmq/liuzhou
NV_BIN=/2023533024/users/zhangmq/condaenvs/naivetorch/bin
export PATH="$NV_BIN:$PATH"
export PYTHON_BIN="$NV_BIN/python"
cd "$REPO"

echo "[env] python=$(which python) torchrun=$(which torchrun)"
python -V

STAMP="$(date -u +%Y%m%d_%H%M%S)"
EXP_ROOT="${EXP_ROOT:-$REPO/rollback_090640_${STAMP}}"
mkdir -p "$EXP_ROOT"/{logs,checkpoints,runs,summary}

ITERATIONS="${ITERATIONS:-15}"
PROFILE="${PROFILE:-aggressive}"
SELF_PLAY_DEVICES="${SELF_PLAY_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
TRAIN_DEVICES="${TRAIN_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
INFER_DEVICES="${INFER_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

ALL_CASES=(
  s0_current_baseline
  s1_lr_2e4
  s2_lr_2e4_warmup0
  s3_add_eval_090640
  s4_add_replay1
  s5_full_090640_param_surface
)

if [[ -n "${CASES:-}" ]]; then
  read -r -a SELECTED_CASES <<< "$CASES"
else
  SELECTED_CASES=("${ALL_CASES[@]}")
fi

declare -A VALID_CASES=()
for case_name in "${ALL_CASES[@]}"; do
  VALID_CASES["$case_name"]=1
done
for case_name in "${SELECTED_CASES[@]}"; do
  if [[ -z "${VALID_CASES[$case_name]:-}" ]]; then
    echo "[error] unknown case: $case_name" >&2
    echo "[error] valid cases: ${ALL_CASES[*]}" >&2
    exit 2
  fi
done

SUMMARY_TSV="$EXP_ROOT/summary/cases.tsv"
{
  printf "case\tparam_overrides\tlog_file\texit_status\tnote\n"
} > "$SUMMARY_TSV"

summary_patterns='run_tag=|lr=|warmup_steps=|streaming_load=|replay_window=|eval_games_vs_random=|eval_games_vs_previous=|eval_vs_random temperature=|eval_vs_previous temperature=|eval_vs_previous:|eval_vs_random:|selfplay outcomes:|selfplay hard value targets nonzero=|iter_signature:|gating_base_model=|completed all iterations|final_checkpoint|final_best_previous_checkpoint'

FAILURES=0

run_case() {
  local case_name="$1"
  local ckpt_dir="$EXP_ROOT/checkpoints/$case_name"
  local run_root="$EXP_ROOT/runs/$case_name"
  local log_file="$EXP_ROOT/logs/${case_name}.log"
  local summary_file="$EXP_ROOT/summary/${case_name}.summary.txt"
  mkdir -p "$ckpt_dir" "$run_root"

  local lr="1e-4"
  local warmup_steps="100"
  local eval_games_vs_random="2000"
  local eval_games_vs_previous="2000"
  local eval_vs_random_temperature="0.0"
  local eval_vs_previous_temperature="1.0"
  local eval_vs_previous_sample_moves="1"
  local replay_window="4"
  local streaming_load="1"
  local lr_cosine_final_scale="0.5"
  local param_overrides="baseline"
  local note=""

  case "$case_name" in
    s0_current_baseline)
      note="Current main baseline with stable init and optimizer continuity preserved."
      ;;
    s1_lr_2e4)
      lr="2e-4"
      param_overrides="LR=2e-4"
      note="First rollback step: higher training step size."
      ;;
    s2_lr_2e4_warmup0)
      lr="2e-4"
      warmup_steps="0"
      param_overrides="LR=2e-4;WARMUP_STEPS=0"
      note="Aligns early training dynamics closer to 090640."
      ;;
    s3_add_eval_090640)
      lr="2e-4"
      warmup_steps="0"
      eval_games_vs_random="1000"
      eval_games_vs_previous="1000"
      eval_vs_random_temperature="0.05"
      eval_vs_previous_temperature="0.05"
      eval_vs_previous_sample_moves="0"
      param_overrides="LR=2e-4;WARMUP_STEPS=0;EVAL_GAMES_VS_RANDOM=1000;EVAL_GAMES_VS_PREVIOUS=1000;EVAL_VS_RANDOM_TEMPERATURE=0.05;EVAL_VS_PREVIOUS_TEMPERATURE=0.05;EVAL_VS_PREVIOUS_SAMPLE_MOVES=0"
      note="Adds 090640 eval and gating parameter surface."
      ;;
    s4_add_replay1)
      lr="2e-4"
      warmup_steps="0"
      eval_games_vs_random="1000"
      eval_games_vs_previous="1000"
      eval_vs_random_temperature="0.05"
      eval_vs_previous_temperature="0.05"
      eval_vs_previous_sample_moves="0"
      replay_window="1"
      param_overrides="LR=2e-4;WARMUP_STEPS=0;EVAL_GAMES_VS_RANDOM=1000;EVAL_GAMES_VS_PREVIOUS=1000;EVAL_VS_RANDOM_TEMPERATURE=0.05;EVAL_VS_PREVIOUS_TEMPERATURE=0.05;EVAL_VS_PREVIOUS_SAMPLE_MOVES=0;REPLAY_WINDOW=1"
      note="Adds replay_window rollback after the main training and eval shifts."
      ;;
    s5_full_090640_param_surface)
      lr="2e-4"
      warmup_steps="0"
      eval_games_vs_random="1000"
      eval_games_vs_previous="1000"
      eval_vs_random_temperature="0.05"
      eval_vs_previous_temperature="0.05"
      eval_vs_previous_sample_moves="0"
      replay_window="1"
      streaming_load="0"
      lr_cosine_final_scale="1.0"
      param_overrides="LR=2e-4;WARMUP_STEPS=0;EVAL_GAMES_VS_RANDOM=1000;EVAL_GAMES_VS_PREVIOUS=1000;EVAL_VS_RANDOM_TEMPERATURE=0.05;EVAL_VS_PREVIOUS_TEMPERATURE=0.05;EVAL_VS_PREVIOUS_SAMPLE_MOVES=0;REPLAY_WINDOW=1;STREAMING_LOAD=0;LR_COSINE_FINAL_SCALE=1.0"
      note="Final 090640 parameter surface alignment. STREAMING_LOAD and LR_COSINE_FINAL_SCALE may be parameter-surface-only in current code."
      ;;
    *)
      echo "[error] unsupported case=$case_name" >&2
      return 2
      ;;
  esac

  echo "============================================================"
  echo "[run] case=$case_name"
  echo "[run] checkpoint_dir=$ckpt_dir"
  echo "[run] run_root=$run_root"
  echo "[run] param_overrides=$param_overrides"
  echo "============================================================"

  set +e
  env \
    PROFILE="$PROFILE" \
    ITERATIONS="$ITERATIONS" \
    TRAIN_STRATEGY=ddp \
    SELF_PLAY_BACKEND=process \
    SELF_PLAY_DEVICES="$SELF_PLAY_DEVICES" \
    TRAIN_DEVICES="$TRAIN_DEVICES" \
    INFER_DEVICES="$INFER_DEVICES" \
    RUN_EVAL_STAGE=1 \
    RUN_INFER_STAGE=0 \
    CHECKPOINT_DIR="$ckpt_dir" \
    RUN_ROOT="$run_root" \
    MODEL_INIT_SEED=20260314 \
    OPTIMIZER_STATE_WORK_PATH="$ckpt_dir/optimizer_state_work.pt" \
    LR="$lr" \
    WARMUP_STEPS="$warmup_steps" \
    LR_COSINE_FINAL_SCALE="$lr_cosine_final_scale" \
    REPLAY_WINDOW="$replay_window" \
    STREAMING_LOAD="$streaming_load" \
    EVAL_GAMES_VS_RANDOM="$eval_games_vs_random" \
    EVAL_GAMES_VS_PREVIOUS="$eval_games_vs_previous" \
    EVAL_VS_RANDOM_TEMPERATURE="$eval_vs_random_temperature" \
    EVAL_VS_PREVIOUS_TEMPERATURE="$eval_vs_previous_temperature" \
    EVAL_VS_PREVIOUS_SAMPLE_MOVES="$eval_vs_previous_sample_moves" \
    bash scripts/big_train_v1.sh 2>&1 | tee "$log_file"
  local run_status=${PIPESTATUS[0]}
  set -e

  {
    echo "# case=$case_name"
    echo "# param_overrides=$param_overrides"
    echo "# exit_status=$run_status"
    echo "# note=$note"
    echo "# log_file=$log_file"
    if [[ -f "$log_file" ]]; then
      rg -n "$summary_patterns" "$log_file" || true
    else
      echo "# log_missing=1"
    fi
  } > "$summary_file"

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$case_name" \
    "$param_overrides" \
    "$log_file" \
    "$run_status" \
    "$note" >> "$SUMMARY_TSV"

  if [[ "$run_status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    echo "[warn] case failed: $case_name exit_status=$run_status"
  else
    echo "[done] $case_name"
  fi
}

for case_name in "${SELECTED_CASES[@]}"; do
  run_case "$case_name"
done

echo
echo "[all done] EXP_ROOT=$EXP_ROOT"
echo "[summary files]"
ls -1 "$EXP_ROOT/summary"
echo "[cases]"
cat "$SUMMARY_TSV"

if [[ "$FAILURES" -ne 0 ]]; then
  echo "[result] failures=$FAILURES"
  exit 1
fi
