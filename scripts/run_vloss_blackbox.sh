#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/.cache/liuzhou
export PYTHONPATH=/home/ubuntu/.cache/liuzhou:/home/ubuntu/.cache/liuzhou/v0/build/src:${PYTHONPATH:-}

PYTHON=${PYTHON:-python}
MCTS_SIMS=512
NUM_STATES=20

echo "== virtual loss blackbox matrix =="
echo "mcts_sims=$MCTS_SIMS num_states=$NUM_STATES"

for batch in 1 256; do
  for vloss in 1 0; do
    echo ""
    echo "---- batch_leaves=$batch virtual_loss=$vloss ----"
    "$PYTHON" tests/v0/check_v0_policy_index_alignment.py \
      --mcts_sims "$MCTS_SIMS" \
      --num_states "$NUM_STATES" \
      --batch_leaves "$batch" \
      --virtual_loss "$vloss"
  done
done
