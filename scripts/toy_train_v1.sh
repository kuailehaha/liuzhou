#!/usr/bin/env bash
# v1 toy training wrapper (through shared train entry)
set -euo pipefail

export PYTHONPATH="./:./build/v0/src:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python}"

ITERATIONS="${ITERATIONS:-40}"
SELF_PLAY_GAMES="${SELF_PLAY_GAMES:-6400}"
MCTS_SIMULATIONS="${MCTS_SIMULATIONS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_v1}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
TRAIN_DEVICES="${TRAIN_DEVICES:-$DEVICE}"

mkdir -p logs
LOG_FILE="logs/train_v1_$(date +%Y%m%d_%H%M%S).log"

echo "Starting v1 training via shared entry..."
echo "Python: $PYTHON_BIN"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop (or detach tmux session)"
echo "Self-play devices: $DEVICES"
echo "Train devices: $TRAIN_DEVICES"

EXTRA_ARGS=()
if [[ -n "$DEVICES" ]]; then
  EXTRA_ARGS+=(--devices "$DEVICES")
fi
if [[ -n "$TRAIN_DEVICES" ]]; then
  EXTRA_ARGS+=(--train_devices "$TRAIN_DEVICES")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/train_entry.py \
  --pipeline v1 \
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
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
