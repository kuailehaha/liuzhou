# v1 Tensorized Pipeline (Scaffold)

This directory hosts the in-progress rewrite of the self-play/training loop
using tensor-native, batched components.  The goal is to eliminate Python-side
per-game loops and unlock GPU execution for the heavy stages (MCTS, self-play,
data packing).

## Layout

- `common/` – shared helpers (device handling, tensor utilities).
- `game/` – batched state representation, vectorized rule application, move encoding.
- `mcts/` – vectorized Monte Carlo Tree Search (shared NN evaluations).
- `net/` – input/output glue between the model and batched action encoding.
- `self_play/` – orchestration for batched rollouts and data collection.
- `train/` – training pipeline consuming tensorized samples.

Each module currently exposes the target API via `NotImplementedError` stubs.
They will be filled in incrementally while keeping the legacy `src/` pipeline
stable for reference and regression comparison.

## Migration Roadmap

1. Implement tensorized game-state conversions (`game/state_batch.py`) and
   vectorized rule helpers (`game/rules_tensor.py`).  Ensure round-trip parity
   with legacy `GameState` via dedicated tests.
2. Finalize move encoding (`game/move_encoder.py`) and neural IO helpers
   (`net/encoding.py`).  Provide legal-mask aware sampling utilities.
3. Build `VectorizedMCTS` on top of the new encoding, including batched leaf
   expansion and virtual loss handling.
4. Replace the legacy self-play loop with `self_play/runner.py`, collecting
   tensors directly for training.
5. Wire the new datasets/pipeline under `train/` and add CLI entry points.
6. Mirror critical tests under `tests/v1/` to guarantee parity and prevent
   regressions.

Progress through these steps will be tracked in `TODO.md`.  The legacy pipeline
remains untouched so we can A/B compare behaviours during the rollout.

## Utilities

- `tools/cross_check_mcts.py` – 对拍脚本，随机采样若干局面并比较传统 `src.mcts.MCTS` 与
  `v1.mcts.vectorized_mcts.VectorizedMCTS` 的策略分布差异，便于在重构过程中快速验证行为一致性。

(torchenv) PS D:\CODES\liuzhou> tree /f /a ./v1 | findstr /v "\.pyc __pycache__ \__init__.py" 

|   README.md
|   
+---common
|   |   tensor_utils.py
|   |
|
+---game
|   |   move_encoder.py
|   |   rules_tensor.py
|   |   state_batch.py
|   |
|
+---mcts
|   |   node_storage.py
|   |   vectorized_mcts.py
|   |
|
+---net
|   |   encoding.py
|   |   policy_decoder.py
|   |
|
+---self_play
|   |   runner.py
|   |   samples.py
|   |
|
+---train
|   |   dataset.py
|   |   pipeline.py
|   |
|