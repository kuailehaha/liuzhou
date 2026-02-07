#!/usr/bin/env bash
# v0 训练脚本（稳定性优先：对上一迭代评估 + 略降 LR + 略早 resign 以减轻和棋主导）
# 详见 TRAINING_STABILITY.md
set -euo pipefail
export PYTHONPATH="./:./v0/build/src${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m v0.train \
  --iterations 40 \
  --self_play_workers 200 \
  --self_play_games_per_worker 64 \
  --mcts_simulations 1024 \
  --self_play_batch_leaves 512 \
  --self_play_inference_batch_size 512 \
  --self_play_opening_random_moves 2 \
  --self_play_resign_threshold -0.9 \
  --self_play_resign_min_moves 36 \
  --self_play_resign_consecutive 3 \
  --epochs 8 \
  --lr 0.002 \
  --value_draw_weight 0.1 \
  --policy_draw_weight 0.3 \
  --batch_size 4096 \
  --device cuda \
  --self_play_devices auto \
  --train_devices auto \
  --eval_devices auto \
  --eval_games_vs_random 200 \
  --eval_games_vs_best 200 \
  --eval_games_vs_previous 100 \
  --mcts_sims_eval 256 \
  --eval_backend v0 \
  --win_rate_threshold 0.55 \
  --checkpoint_dir ./checkpoints_v0
