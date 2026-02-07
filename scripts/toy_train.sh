#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="./:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"

# v0 版训练脚本
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m v0.train \
  --iterations 40 \
  --self_play_workers 200 \
  --self_play_games_per_worker 8 \
  --mcts_simulations 1024 \
  --self_play_batch_leaves 512 \
  --self_play_inference_batch_size 512 \
  --self_play_opening_random_moves 1 \
  --self_play_resign_threshold -0.9 \
  --self_play_resign_min_moves 100 \
  --self_play_resign_consecutive 3 \
  --epochs 8 \
  --lr 0.003 \
  --value_draw_weight 0.1 \
  --policy_draw_weight 0.3 \
  --batch_size 4096 \
  --device cuda \
  --self_play_devices auto \
  --train_devices auto \
  --eval_devices auto \
  --eval_games_vs_random 200 \
  --eval_games_vs_best 200 \
  --mcts_sims_eval 256 \
  --eval_backend v0 \
  --checkpoint_dir ./checkpoints_v0
