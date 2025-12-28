cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:$PYTHONPATH

# 先测试自博弈
python -m v0.generate_data \
  --num_games 64 \
  --mcts_simulations 800 \
  --batch_leaves 256 \
  --device cuda

# 再测试训练
python -m v0.train \

  --data_files ./v0/data/self_play/*.jsonl \
  --iterations 3 \
  --epochs 5 \
  --batch_size 128 \
  --device cuda \
  --eval_games_vs_random 100 \
  --eval_games_vs_best 0