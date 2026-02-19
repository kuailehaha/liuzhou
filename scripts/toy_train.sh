#!/usr/bin/env bash
# Shared toy training script for v0/v1.
set -euo pipefail

export PYTHONPATH="./:./build/v0/src:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"

PIPELINE="${PIPELINE:-v0}" # v0 or v1
ITERATIONS="${ITERATIONS:-40}"
SELF_PLAY_GAMES="${SELF_PLAY_GAMES:-6400}"
MCTS_SIMULATIONS="${MCTS_SIMULATIONS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-cuda:0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_${PIPELINE}}"
V0_BATCH_LEAVES="${V0_BATCH_LEAVES:-512}"
V0_DATA_DIR="${V0_DATA_DIR:-./v0/data/self_play}"
V0_EVAL_GAMES="${V0_EVAL_GAMES:-20}"

mkdir -p logs
LOG_FILE="logs/train_${PIPELINE}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting shared toy training..."
echo "Pipeline: $PIPELINE"
echo "Python: $PYTHON_BIN"
echo "Log file: $LOG_FILE"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON_BIN" scripts/train_entry.py \
  --pipeline "$PIPELINE" \
  --iterations "$ITERATIONS" \
  --self_play_games "$SELF_PLAY_GAMES" \
  --mcts_simulations "$MCTS_SIMULATIONS" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --temperature_init 1.0 \
  --temperature_final 0.1 \
  --temperature_threshold 10 \
  --exploration_weight 1.0 \
  --dirichlet_alpha 0.3 \
  --dirichlet_epsilon 0.25 \
  --soft_value_k 2.0 \
  --max_game_plies 512 \
  --device "$DEVICE" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --v0_batch_leaves "$V0_BATCH_LEAVES" \
  --v0_eval_games "$V0_EVAL_GAMES" \
  --v0_data_dir "$V0_DATA_DIR" 2>&1 | tee "$LOG_FILE"
