#!/usr/bin/env bash
# v1 toy training script (GPU-first self-play + tensor-native training)
set -euo pipefail

export PYTHONPATH="./:./build/v0/src:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"

# Keep v1 train defaults aligned with scripts/toy_train.sh as much as current
# v1 interface allows.
ITERATIONS="${ITERATIONS:-40}"
SELF_PLAY_GAMES="${SELF_PLAY_GAMES:-6400}"   # v0: 200 * 32
MCTS_SIMULATIONS="${MCTS_SIMULATIONS:-1024}" # match v0 toy search budget
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p logs
LOG_FILE="logs/train_v1_$(date +%Y%m%d_%H%M%S).log"

echo "Starting v1 training..."
echo "Python: $PYTHON_BIN"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop (or detach tmux session)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON_BIN" -m v1.train \
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
  --checkpoint_dir ./checkpoints_v1 2>&1 | tee "$LOG_FILE"
