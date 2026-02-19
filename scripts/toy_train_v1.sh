#!/usr/bin/env bash
# v1 toy training script (GPU-first self-play + tensor-native training)
set -euo pipefail

export PYTHONPATH="./:./build/v0/src:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"

mkdir -p logs
LOG_FILE="logs/train_v1_$(date +%Y%m%d_%H%M%S).log"

echo "Starting v1 training..."
echo "Python: $PYTHON_BIN"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop (or detach tmux session)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON_BIN" -m v1.train \
  --iterations 40 \
  --self_play_games 512 \
  --mcts_simulations 512 \
  --batch_size 4096 \
  --epochs 3 \
  --lr 0.0003 \
  --weight_decay 0.0001 \
  --temperature_init 1.0 \
  --temperature_final 0.1 \
  --temperature_threshold 10 \
  --exploration_weight 1.0 \
  --dirichlet_alpha 0.3 \
  --dirichlet_epsilon 0.25 \
  --soft_value_k 2.0 \
  --max_game_plies 512 \
  --device cuda:0 \
  --checkpoint_dir ./checkpoints_v1 2>&1 | tee "$LOG_FILE"
