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
EXP_ROOT="$REPO/ablation_4x5_${STAMP}"
mkdir -p "$EXP_ROOT"/{logs,checkpoints,runs,summary}

# 默认就是你要的 5 iter；可在执行前 export ITERATIONS=... 覆盖
ITERATIONS="${ITERATIONS:-5}"
PROFILE="${PROFILE:-aggressive}"

# 4卡默认；可覆盖
SELF_PLAY_DEVICES="${SELF_PLAY_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
TRAIN_DEVICES="${TRAIN_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
INFER_DEVICES="${INFER_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

run_case() {
  local case_name="$1"       # baseline | init_only | no_optstate | both_off
  local model_init_seed="$2" # 20260314 or 0
  local opt_mode="$3"        # enabled | disabled

  local ckpt_dir="$EXP_ROOT/checkpoints/$case_name"
  local run_root="$EXP_ROOT/runs/$case_name"
  local log_file="$EXP_ROOT/logs/${case_name}.log"
  mkdir -p "$ckpt_dir" "$run_root"

  local opt_path=""
  if [[ "$opt_mode" == "enabled" ]]; then
    opt_path="$ckpt_dir/optimizer_state_work.pt"
  else
    opt_path=""  # 显式空值：禁用 optimizer continuity
  fi

  echo "============================================================"
  echo "[run] case=$case_name model_init_seed=$model_init_seed opt_mode=$opt_mode"
  echo "[run] checkpoint_dir=$ckpt_dir"
  echo "[run] run_root=$run_root"
  echo "============================================================"

  env \
    PROFILE="$PROFILE" \
    TRAIN_STRATEGY=ddp \
    SELF_PLAY_BACKEND=process \
    SELF_PLAY_DEVICES="$SELF_PLAY_DEVICES" \
    TRAIN_DEVICES="$TRAIN_DEVICES" \
    INFER_DEVICES="$INFER_DEVICES" \
    ITERATIONS="$ITERATIONS" \
    RUN_EVAL_STAGE=1 \
    RUN_INFER_STAGE=0 \
    CHECKPOINT_DIR="$ckpt_dir" \
    RUN_ROOT="$run_root" \
    MODEL_INIT_SEED="$model_init_seed" \
    OPTIMIZER_STATE_WORK_PATH="$opt_path" \
    EVAL_GAMES_VS_RANDOM=2000 \
    EVAL_GAMES_VS_PREVIOUS=2000 \
    EVAL_VS_RANDOM_TEMPERATURE=0.0 \
    EVAL_VS_PREVIOUS_TEMPERATURE=1.0 \
    EVAL_VS_PREVIOUS_SAMPLE_MOVES=1 \
    bash scripts/big_train_v1.sh | tee "$log_file"

  rg -n "iter_signature:|\\[eval\\] vs_random|\\[eval\\] vs_previous|gating\\(vs_best_previous\\)|completed all iterations|final_checkpoint|final_best_previous_checkpoint" \
    "$log_file" > "$EXP_ROOT/summary/${case_name}.summary.txt" || true

  echo "[done] $case_name"
}

run_case baseline   20260314 enabled
run_case init_only  0        enabled
run_case no_optstate 20260314 disabled
run_case both_off   0        disabled

echo
echo "[all done] EXP_ROOT=$EXP_ROOT"
echo "[summary files]"
ls -1 "$EXP_ROOT/summary"
