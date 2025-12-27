cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH

# 先测试自博弈
python -m v0.generate_data \
  --num_games 4 \
  --mcts_simulations 50 \
  --batch_leaves 256 \
  --device cuda

# 再测试训练
python -m v0.train \
  --data_files ./v0/data/self_play/*.jsonl \
  --iterations 1 \
  --epochs 2 \
  --batch_size 16 \
  --device cuda \
  --eval_games_vs_random 4 \
  --eval_games_vs_best 0