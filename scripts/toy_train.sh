#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH

# # legacy 版训练脚本
# python -m src.train \
#   --iterations 3 \
#   --self_play_workers 8 \
#   --self_play_games_per_worker 8 \
#   --mcts_simulations 800 \
#   --epochs 5 \
#   --batch_size 256 \
#   --device cuda \
#   --eval_games_vs_random 200 \
#   --eval_games_vs_best 200 \
#   --mcts_sims_eval 20 \
#   --checkpoint_dir ./checkpoints_legacy

# v0 版训练脚本
python -m v0.train \
  --iterations 40 \
  --self_play_workers 8 \
  --self_play_games_per_worker 80 \
  --mcts_simulations 800 \
  --self_play_batch_leaves 256 \
  --epochs 5 \
  --decisive_only \
  --batch_size 256 \
  --device cuda \
  --eval_games_vs_random 200 \
  --eval_games_vs_best 200 \
  --mcts_sims_eval 20 \
  --checkpoint_dir ./checkpoints_v0
