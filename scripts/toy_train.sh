#!/usr/bin/env bash
# v0 训练脚本（稳定性优先：对上一迭代评估 + 略降 LR + 略早 resign 以减轻和棋主导）
# 详见 TRAINING_STABILITY.md
set -euo pipefail
export PYTHONPATH="./:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"

# 创建 logs 目录（如果不存在）
mkdir -p logs

# 生成带时间戳的日志文件名
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop (or detach tmux session)"

# 使用 tee 同时输出到终端和日志文件
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m v0.train \
  --iterations 40 \
  --self_play_workers 200 \
  --self_play_games_per_worker 32 \
  --mcts_simulations 1024 \
  --self_play_batch_leaves 512 \
  --self_play_inference_batch_size 512 \
  --self_play_opening_random_moves 2 \
  --self_play_resign_threshold -0.8 \
  --self_play_resign_min_moves 36 \
  --self_play_resign_consecutive 3 \
  --epochs 3 \
  --lr 0.0003 \
  --replay_window 4 \
  --policy_draw_weight 0.3 \
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
  --checkpoint_dir ./checkpoints_v0 2>&1 | tee "$LOG_FILE"
