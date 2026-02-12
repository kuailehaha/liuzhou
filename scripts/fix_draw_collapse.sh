#!/usr/bin/env bash
# 修复和棋崩溃问题的训练脚本
# 策略：激进resign + decisive_only + 增加探索
set -euo pipefail
export PYTHONPATH="./:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs
LOG_FILE="logs/train_fix_$(date +%Y%m%d_%H%M%S).log"

echo "=== Fix Draw Collapse Training ==="
echo "Key changes:"
echo "  1. Decisive only: filter out all draws"
echo "  2. Aggressive resign: -0.95 threshold, min 20 moves"
echo "  3. More exploration: 6 random opening moves"
echo "  4. Lower draw weight: 0.1 (in case any draws slip through)"
echo ""
echo "Log file: $LOG_FILE"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m v0.train \
  --iterations 40 \
  --self_play_workers 200 \
  --self_play_games_per_worker 32 \
  --mcts_simulations 1024 \
  --self_play_batch_leaves 512 \
  --self_play_inference_batch_size 512 \
  --self_play_opening_random_moves 6 \
  --self_play_resign_threshold -0.95 \
  --self_play_resign_min_moves 20 \
  --self_play_resign_consecutive 2 \
  --decisive_only \
  --epochs 3 \
  --lr 0.0003 \
  --replay_window 1 \
  --policy_draw_weight 0.1 \
  --batch_size 4096 \
  --device cuda \
  --self_play_devices auto \
  --train_devices cuda:0 \
  --eval_devices auto \
  --eval_games_vs_random 200 \
  --eval_games_vs_best 200 \
  --eval_games_vs_previous 200 \
  --mcts_sims_eval 256 \
  --eval_backend v0 \
  --win_rate_threshold 0.55 \
  --checkpoint_dir ./checkpoints_v0_fix \
  "$@" 2>&1 | tee "$LOG_FILE"
