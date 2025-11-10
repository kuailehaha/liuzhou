"""
State batch tensor conversion regression tests.

Usage:
  pytest tests/v1/test_state_batch.py -q
"""

import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase, Player

from v1.game.move_encoder import DEFAULT_ACTION_SPEC, action_to_index, decode_action_indices, encode_actions
from v1.game.rules_tensor import apply_placement, generate_placement_moves
from v1.game.state_batch import from_game_states, to_game_states


def test_state_batch_roundtrip_with_marks():
    state = GameState()
    state.board[0][0] = Player.BLACK.value
    state.board[1][1] = Player.WHITE.value
    state.marked_black.add((0, 0))
    state.marked_white.add((1, 1))
    state.phase = Phase.MARK_SELECTION
    state.pending_marks_required = 1
    state.pending_marks_remaining = 1
    state.move_count = 3

    batch = from_game_states([state])
    restored = to_game_states(batch)[0]

    assert restored.board == state.board
    assert restored.marked_black == state.marked_black
    assert restored.marked_white == state.marked_white
    assert restored.phase == state.phase
    assert restored.current_player == state.current_player
    assert restored.pending_marks_remaining == state.pending_marks_remaining
    assert restored.move_count == state.move_count


def test_generate_placement_moves_initial_state():
    batch = from_game_states([GameState()])
    mask = generate_placement_moves(batch)

    assert mask.shape == (1, GameState.BOARD_SIZE, GameState.BOARD_SIZE)
    assert mask.all()


def test_apply_placement_updates_state():
    batch = from_game_states([GameState()])
    indices = torch.tensor([0], dtype=torch.long)

    updated = apply_placement(batch, indices)
    new_state = to_game_states(updated)[0]

    assert new_state.board[0][0] == Player.BLACK.value
    assert new_state.current_player == Player.WHITE
    assert new_state.phase == Phase.PLACEMENT


def test_encode_actions_has_expected_shape():
    batch = from_game_states([GameState()])
    mask = encode_actions(batch, DEFAULT_ACTION_SPEC)

    assert mask.shape == (1, DEFAULT_ACTION_SPEC.total_dim)
    # Initial state only has placement moves legal
    placement_sum = mask[:, : DEFAULT_ACTION_SPEC.placement_dim].sum()
    movement_sum = mask[:, DEFAULT_ACTION_SPEC.placement_dim : DEFAULT_ACTION_SPEC.placement_dim + DEFAULT_ACTION_SPEC.movement_dim].sum()
    selection_sum = mask[:, DEFAULT_ACTION_SPEC.placement_dim + DEFAULT_ACTION_SPEC.movement_dim : DEFAULT_ACTION_SPEC.placement_dim + DEFAULT_ACTION_SPEC.movement_dim + DEFAULT_ACTION_SPEC.selection_dim].sum()
    aux_sum = mask[:, -DEFAULT_ACTION_SPEC.auxiliary_dim :].sum()

    assert placement_sum.item() == GameState.BOARD_SIZE * GameState.BOARD_SIZE
    assert movement_sum.item() == 0
    assert selection_sum.item() == 0
    assert aux_sum.item() == 0


def test_decode_actions_maps_to_move_dicts():
    batch = from_game_states([GameState()])
    # placement action index 5
    placement_idx = torch.tensor([5], dtype=torch.long)
    decoded = decode_action_indices(placement_idx, batch, DEFAULT_ACTION_SPEC)[0]
    assert decoded["action_type"] == "place"
    assert decoded["position"] == (0, 5)


def test_action_to_index_roundtrip():
    batch = from_game_states([GameState()])
    spec = DEFAULT_ACTION_SPEC
    decoded = decode_action_indices(torch.tensor([7]), batch, spec)[0]
    idx = action_to_index(decoded, GameState.BOARD_SIZE, spec)
    assert idx == 7
