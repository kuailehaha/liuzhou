# v0 Pipeline Notes (Data -> Train -> Eval)

This document summarizes the v0 pipeline and highlights the most likely
failure points when eval win-rate vs RandomAgent stays near zero.

## 1) Data generation (self-play)

- Entry point: `v0/generate_data.py`
- Core self-play loop: `v0/python/self_play_runner.py`
- MCTS backend: `v0/python/mcts.py` -> `v0_core.MCTSCore` (C++)
- Moves applied for the "real" game state in Python: `src/move_generator.py`

Flow:
1. `self_play_single_game_v0` calls `mcts.search(state)` for policy.
2. It samples a move from that policy and updates the Python `GameState`
   via `apply_move` (legacy rules).
3. MCTS root is advanced with the same move (v0 action encoding).
4. Game ends when `state.get_winner()` returns non-None or move limit reached.
5. Output: `game_states`, `game_policies`, `result`, `soft_value`.

Notes:
- `result` is from BLACK perspective: +1 win / -1 loss / 0 draw.
- `soft_value` is a material heuristic: tanh(k * material_delta).

## 2) Sample serialization

- `v0/python/state_io.py`
  - `flatten_training_games` expands each game into per-position samples.
  - Each sample uses the current player perspective:
    `value = sign * result`, `soft_value = sign * soft_value`.
    `sign = +1` if current_player is BLACK else `-1`.

Record format (JSONL):
```
{ "state": {...}, "policy": [...], "value": 1/-1/0, "soft_value": ... }
```

## 3) Training

- Entry point: `v0/train.py`
- Actual trainer: `src/train.py` (`train_network`)
- Inputs:
  - `state_to_tensor(state, state.current_player)` (legacy encoding)
  - `generate_all_legal_moves(state)` (legacy move order)
  - target `policy` (from v0 self-play)
  - target `value`, `soft_value`

Losses:
- Policy: KLDiv over legal moves, normalized per-sample
- Value: MSE between model value and mixed target
  `y_mix = (1 - alpha) * value + alpha * soft_value`

### Value sign chain (no reversal)
1. Self-play result is BLACK perspective.
2. `flatten_training_games` flips by current player.
3. Model is trained to predict value from current-player perspective.
4. MCTS backprop in both legacy and v0 flips sign on each ply.

Conclusion: value sign is consistent; no obvious "objective reversed".

## 4) Evaluation

- `v0/train.py` uses legacy eval:
  - `src.evaluate.MCTSAgent` (legacy MCTS)
  - `RandomAgent`
  - `play_single_game` with `max_moves=200` (draw if limit hit)

This means:
- v0 self-play uses v0 MCTS + v0 core rules
- evaluation uses legacy MCTS + legacy rules
Any divergence between v0 rules and legacy rules can destroy the signal.

## 5) Likely failure points (based on current logs)

1. **Draw-heavy data**
   - If most games end by move limit, value labels are ~0.
   - Value loss is tiny (~0.003), indicating low target magnitude.
   - Policy targets can be close to uniform; model learns to be "safe".

2. **v0 vs legacy rules mismatch**
   - Self-play uses v0 kernels for legal moves and apply-moves.
   - Training/eval uses legacy rule engine.
   - If any phase rules differ, policy/value targets become inconsistent.

3. **Static offline data**
   - Offline mode reuses the same data every iteration.
   - If that data is from a weak model, learning plateaus fast.

4. **Policy-target alignment**
   - Requires v0 policy order to match `generate_all_legal_moves` order.
   - If mismatched, training becomes noise even though "valid_policy_samples"
     stays high.

## 6) Quick diagnostics to run

1. Check draw ratio in data:
   - Count how many samples have `value == 0`.
2. Check policy length alignment:
   - For random samples, assert `len(policy) == len(generate_all_legal_moves(state))`.
3. Compare v0 vs legacy legal moves:
   - Use parity tests under `tests/v0/` or add a small script to compare
     legal move sets for random states.
4. Evaluate with v0 MCTS (not legacy):
   - If v0-vs-random is higher, the eval backend mismatch is the culprit.

## 7) Suggested next steps

- Run a short self-play with the latest model (not random init) and
  retrain; check if win rate moves.
- Temporarily set `soft_label_alpha = 0` to focus value on win/loss.
- Reduce move limit or add a small win bonus to reduce draws.

