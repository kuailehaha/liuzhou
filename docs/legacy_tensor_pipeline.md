# Legacy Tensor Pipeline Reference

This note captures the legacy (non-vectorised) inference/training pipeline so the v1 tensorised flow can stay drop-in compatible.

## Core domain objects
- `GameState` (`src/game_state.py`): 6x6 board (`board: List[List[int]]`, with `Player.BLACK.value == 1`, `Player.WHITE.value == -1`), per-player marked sets (`marked_black`, `marked_white`), phase enum, counters, and `current_player`.
- `MoveType` (`src/move_generator.py`): alias for `Dict[str, Any]`. Each move dict always contains `phase` and `action_type`, plus phase-specific coordinates such as `position`, `from_position`, `to_position`.

## Legacy pipeline functions
| Stage | Function (module) | Inputs | Outputs | Notes |
| ----- | ----------------- | ------ | ------- | ----- |
| State encoding | `state_to_tensor(state, player_to_act)` (`src/neural_network.py`) | `GameState`, `Player` | `torch.FloatTensor` shaped `(1, 11, 6, 6)` on **CPU** | Channel layout: `0=self pieces`, `1=opponent pieces`, `2=self marks`, `3=opponent marks`, `4-10` = one-hot phases `PLACEMENT->COUNTER_REMOVAL`. Dtype defaults to `torch.float32`. No automatic device move. |
| Policy/value forward | `ChessNet.forward(x)` (`src/neural_network.py`) | `torch.FloatTensor` `(B, 11, 6, 6)` on model device | Tuple: `log_policy_pos1`, `log_policy_pos2`, `log_policy_mark_capture` each `(B, 36)` float32 log-softmax; `value` `(B, 1)` float32 in `[-1, 1]` | `NUM_INPUT_CHANNELS` = 11, `board_size` = 6 => flatten factor 36. Outputs already log-softmaxed per head. |
| Policy projection | `get_move_probabilities(log_policy_pos1, log_policy_pos2, log_policy_mark_capture, legal_moves, board_size, device)` (`src/neural_network.py`) | Three log-policy tensors (either `(36,)` or `(1, 36)`), `legal_moves: list[MoveType]`, `board_size: int`, `device: Union[str, torch.device]` | `probabilities: list[float]` (length = `len(legal_moves)`, detached to CPU), `combined_log_probs: torch.Tensor` `(N,)` on `device` | Flattens inputs, indexes via `r * board_size + c`. Movement score = `pos2[from] + pos1[to]`. Special cases: `process_removal` uses zero log-score; if every entry is `-inf`, replaces with zeros before softmax; single legal move => `[1.0]` without softmax. |
| Move generation | `generate_all_legal_moves(state)` (`src/move_generator.py`) | `GameState` | `list[MoveType]` | Dispatches by `state.phase`. Returns empty list if `state.is_game_over()`. Order of the returned list is the canonical ordering expected by MCTS/training. |
| Move application | `apply_move(state, move, quiet=False)` (`src/move_generator.py`) | `GameState`, `MoveType`, optional `quiet: bool` | New `GameState` instance | Validates `move['phase']`/`action_type`. Delegates to phase-specific helpers, increments `move_count`, returns the updated state (callers treat it as immutable). |

## Interaction summary
1. `generate_all_legal_moves` builds the ordered legal move list consumed by both MCTS and training.
2. `state_to_tensor` encodes a single `GameState` for the current player into the `(1, 11, 6, 6)` CPU tensor used as network input.
3. `ChessNet.forward` produces the three log-policy heads plus value; downstream code typically squeezes batch dim before projection.
4. `get_move_probabilities` maps the per-head logits back onto the `legal_moves` ordering, yielding sampling probabilities and raw combined log-scores.
5. `apply_move` executes the chosen move to obtain the successor `GameState` for the next loop iteration.

Any v1 tensorised modules should preserve these contracts (shapes, devices, ordering, fallback behaviour) to remain drop-in replacements for the legacy flow.
