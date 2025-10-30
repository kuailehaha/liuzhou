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

