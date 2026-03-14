from src.game_state import GameState, Phase, Player
from src.move_generator import apply_move


def test_mark_selection_does_not_adjudicate_before_movement_starts() -> None:
    moves = [
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (2, 4)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (5, 2)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (1, 4)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 4)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (2, 3)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (1, 0)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (1, 3)},
    ]

    state = GameState()
    for move in moves:
        state = apply_move(state, move, quiet=True)

    assert state.phase == Phase.MARK_SELECTION
    assert state.get_winner() is None
    assert not state.is_game_over()


def test_winner_check_remains_active_in_movement_stage() -> None:
    state = GameState()
    state.phase = Phase.MOVEMENT
    # WHITE has <4 pieces, so BLACK should win once movement stage has started.
    state.board[0][0] = 1
    state.board[0][1] = 1
    state.board[0][2] = 1
    state.board[0][3] = 1
    state.board[5][0] = -1
    state.board[5][1] = -1
    state.board[5][2] = -1

    assert state.get_winner() is not None
    assert state.is_game_over()
