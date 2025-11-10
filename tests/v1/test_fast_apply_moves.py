import random

import torch

from src.game_state import GameState, Phase
from src.move_generator import apply_move, generate_all_legal_moves

from v1.game.move_encoder import ActionEncodingSpec, DEFAULT_ACTION_SPEC, encode_actions
from v1.game.state_batch import TensorStateBatch, from_game_states, to_game_states
from v1.game.fast_apply_moves import batch_apply_moves_fast


ACTION_KIND_PLACEMENT = 1
ACTION_KIND_MOVEMENT = 2
ACTION_KIND_MARK_SELECTION = 3
ACTION_KIND_CAPTURE_SELECTION = 4
ACTION_KIND_FORCED_REMOVAL_SELECTION = 5
ACTION_KIND_COUNTER_REMOVAL_SELECTION = 6
ACTION_KIND_NO_MOVES_REMOVAL_SELECTION = 7
ACTION_KIND_PROCESS_REMOVAL = 8


def _metadata_to_move(code, state: GameState, spec: ActionEncodingSpec):
    kind, primary, secondary, extra = [int(x) for x in code]
    board_size = state.BOARD_SIZE
    if kind == ACTION_KIND_PLACEMENT:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.PLACEMENT, "action_type": "place", "position": (r, c)}
    if kind == ACTION_KIND_MOVEMENT:
        r_from, c_from = divmod(primary, board_size)
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        dr, dc = dirs[secondary]
        return {
            "phase": Phase.MOVEMENT,
            "action_type": "move",
            "from_position": (r_from, c_from),
            "to_position": (r_from + dr, c_from + dc),
        }
    if kind == ACTION_KIND_MARK_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.MARK_SELECTION, "action_type": "mark", "position": (r, c)}
    if kind == ACTION_KIND_CAPTURE_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.CAPTURE_SELECTION, "action_type": "capture", "position": (r, c)}
    if kind == ACTION_KIND_FORCED_REMOVAL_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.FORCED_REMOVAL, "action_type": "remove", "position": (r, c)}
    if kind == ACTION_KIND_COUNTER_REMOVAL_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.COUNTER_REMOVAL, "action_type": "counter_remove", "position": (r, c)}
    if kind == ACTION_KIND_NO_MOVES_REMOVAL_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.MOVEMENT, "action_type": "no_moves_remove", "position": (r, c)}
    if kind == ACTION_KIND_PROCESS_REMOVAL:
        return {"phase": Phase.REMOVAL, "action_type": "process_removal"}
    raise ValueError(f"Unsupported action kind: {kind}")


def _sample_states(num_states: int, max_moves: int) -> list[GameState]:
    rng = random.Random(0xC0FFEE)
    states = []
    for _ in range(num_states):
        state = GameState()
        steps = rng.randint(0, max_moves)
        for _ in range(steps):
            legal = generate_all_legal_moves(state)
            if not legal:
                break
            move = rng.choice(legal)
            state = apply_move(state, move, quiet=True)
            if state.is_game_over():
                break
        states.append(state)
    return states


def test_fast_apply_moves_matches_python():
    states = _sample_states(num_states=10000, max_moves=80)
    batch = from_game_states(states, device=torch.device("cpu"))
    spec = DEFAULT_ACTION_SPEC

    result = encode_actions(batch, spec, return_metadata=True)
    assert isinstance(result, tuple)
    legal_mask, metadata = result
    assert metadata is not None

    parent_indices = []
    action_codes = []
    expected_states = []

    for idx, state in enumerate(states):
        legal_indices = legal_mask[idx].nonzero(as_tuple=False).view(-1)
        for action_idx in legal_indices.tolist():
            code = metadata[idx, action_idx]
            move = _metadata_to_move(code, state, spec)
            new_state = apply_move(state, move, quiet=True)
            parent_indices.append(idx)
            action_codes.append(code.tolist())
            expected_states.append(new_state)

    assert action_codes, "No legal actions found for sampled states."

    parent_tensor = torch.tensor(parent_indices, dtype=torch.long)
    codes_tensor = torch.tensor(action_codes, dtype=torch.int32)

    fast_batch = batch_apply_moves_fast(batch, parent_tensor, codes_tensor)
    assert fast_batch is not None, "Fast apply extension unavailable."

    expected_batch = from_game_states(expected_states, device=torch.device("cpu"))

    torch.testing.assert_close(fast_batch.board, expected_batch.board)
    torch.testing.assert_close(fast_batch.marks_black, expected_batch.marks_black)
    torch.testing.assert_close(fast_batch.marks_white, expected_batch.marks_white)
    torch.testing.assert_close(fast_batch.phase, expected_batch.phase)
    torch.testing.assert_close(fast_batch.current_player, expected_batch.current_player)
    torch.testing.assert_close(
        fast_batch.pending_marks_required, expected_batch.pending_marks_required
    )
    torch.testing.assert_close(
        fast_batch.pending_marks_remaining, expected_batch.pending_marks_remaining
    )
    torch.testing.assert_close(
        fast_batch.pending_captures_required, expected_batch.pending_captures_required
    )
    torch.testing.assert_close(
        fast_batch.pending_captures_remaining, expected_batch.pending_captures_remaining
    )
    torch.testing.assert_close(
        fast_batch.forced_removals_done, expected_batch.forced_removals_done
    )
    torch.testing.assert_close(fast_batch.move_count, expected_batch.move_count)

