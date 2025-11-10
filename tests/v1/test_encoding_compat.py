"""
Tensor encoding parity regression tests for v1.

Usage:
  pytest tests/v1/test_encoding_compat.py -q
"""

import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase, Player
from src.neural_network import state_to_tensor
from v1.game.state_batch import from_game_states
from v1.net.encoding import states_to_model_input


def _scenario_initial(state: GameState) -> None:
    # Default starting position: placement phase, black to move.
    pass


def _scenario_mark_selection(state: GameState) -> None:
    state.current_player = Player.WHITE
    state.phase = Phase.MARK_SELECTION
    state.board[0][0] = Player.WHITE.value
    state.board[5][5] = Player.BLACK.value
    state.marked_white = {(0, 0)}
    state.marked_black = {(5, 5)}


def _scenario_capture_selection(state: GameState) -> None:
    state.current_player = Player.BLACK
    state.phase = Phase.CAPTURE_SELECTION
    state.pending_captures_required = 2
    state.pending_captures_remaining = 1
    state.board[1][1] = Player.BLACK.value
    state.board[2][2] = Player.WHITE.value
    state.marked_white = {(2, 2)}
    state.marked_black = set()


def _scenario_movement_with_marks(state: GameState) -> None:
    state.current_player = Player.BLACK
    state.phase = Phase.MOVEMENT
    state.board[3][3] = Player.BLACK.value
    state.board[4][4] = Player.WHITE.value
    state.marked_black = {(3, 3)}
    state.marked_white = {(4, 4)}


def _scenario_forced_removal(state: GameState) -> None:
    state.current_player = Player.WHITE
    state.phase = Phase.FORCED_REMOVAL
    state.forced_removals_done = 0
    state.board[0][5] = Player.BLACK.value
    state.board[5][0] = Player.WHITE.value


def _scenario_counter_removal(state: GameState) -> None:
    state.current_player = Player.BLACK
    state.phase = Phase.COUNTER_REMOVAL
    state.board[2][3] = Player.BLACK.value
    state.board[3][2] = Player.WHITE.value


SCENARIOS = [
    ("initial_state", _scenario_initial),
    ("mark_selection", _scenario_mark_selection),
    ("capture_selection", _scenario_capture_selection),
    ("movement_with_marks", _scenario_movement_with_marks),
    ("forced_removal", _scenario_forced_removal),
    ("counter_removal", _scenario_counter_removal),
]


@pytest.mark.parametrize("name, configurator", SCENARIOS)
def test_states_to_model_input_matches_legacy_encoding(name: str, configurator) -> None:
    state = GameState()
    state.marked_black.clear()
    state.marked_white.clear()
    configurator(state)

    legacy = state_to_tensor(state, state.current_player)
    batch = from_game_states([state])
    tensorized = states_to_model_input(batch)

    assert tensorized.shape == legacy.shape
    assert tensorized.dtype == legacy.dtype
    assert tensorized.device == legacy.device
    torch.testing.assert_close(tensorized, legacy, atol=0.0, rtol=0.0)
