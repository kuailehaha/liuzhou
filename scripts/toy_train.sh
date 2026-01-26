#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/.cache/liuzhou
# export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH
export PYTHONPATH="/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src${PYTHONPATH:+:$PYTHONPATH}"

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
  --self_play_workers 12 \
  --self_play_games_per_worker 80 \
  --mcts_simulations 1024 \
  --self_play_batch_leaves 512 \
  --self_play_inference_batch_size 512 \
  --self_play_opening_random_moves 8 \
  --self_play_resign_threshold -0.8 \
  --self_play_resign_min_moves 10 \
  --self_play_resign_consecutive 3 \
  --epochs 5 \
  --value_draw_weight 0.1 \
  --policy_draw_weight 0.3 \
  --batch_size 512 \
  --device cuda \
  --eval_games_vs_random 200 \
  --eval_games_vs_best 200 \
  --mcts_sims_eval 256 \
  --eval_backend v0 \
  --checkpoint_dir ./checkpoints_v0 \
  --load_checkpoint ./checkpoints_v0/model_iter_6.pt
