import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase, Player
from src.move_generator import generate_all_legal_moves

from v1.game.move_encoder import (
    DEFAULT_ACTION_SPEC,
    action_to_index,
    decode_action_indices,
    encode_actions,
    encode_actions_python,
)
from v1.game.state_batch import from_game_states
from v1.game.fast_legal_mask import encode_actions_fast


def _state_placement() -> GameState:
    state = GameState()
    return state


def _state_movement() -> GameState:
    state = GameState()
    state.phase = Phase.MOVEMENT
    state.current_player = Player.BLACK
    state.board[2][2] = Player.BLACK.value
    state.board[2][3] = 0
    state.board[3][2] = 0
    state.board[4][4] = Player.WHITE.value
    return state


def _state_movement_no_moves() -> GameState:
    state = GameState()
    state.phase = Phase.MOVEMENT
    state.current_player = Player.BLACK
    state.board = [[0] * state.BOARD_SIZE for _ in range(state.BOARD_SIZE)]
    state.board[0][0] = Player.BLACK.value
    state.board[0][1] = Player.WHITE.value
    state.board[1][0] = Player.WHITE.value
    state.board[5][5] = Player.WHITE.value
    return state


def _state_mark_selection() -> GameState:
    state = GameState()
    state.phase = Phase.MARK_SELECTION
    state.current_player = Player.BLACK
    state.pending_marks_required = 1
    state.pending_marks_remaining = 1
    state.board[0][0] = Player.WHITE.value
    state.board[1][1] = Player.WHITE.value
    state.board[5][5] = Player.BLACK.value
    return state


def _state_capture_selection() -> GameState:
    state = GameState()
    state.phase = Phase.CAPTURE_SELECTION
    state.current_player = Player.BLACK
    state.pending_captures_required = 2
    state.pending_captures_remaining = 1
    state.board[2][2] = Player.WHITE.value
    state.board[4][4] = Player.BLACK.value
    return state


def _state_forced_removal_white_turn() -> GameState:
    state = GameState()
    state.phase = Phase.FORCED_REMOVAL
    state.current_player = Player.WHITE
    state.forced_removals_done = 0
    state.board[0][5] = Player.BLACK.value
    state.board[5][5] = Player.WHITE.value
    return state


def _state_forced_removal_black_turn() -> GameState:
    state = GameState()
    state.phase = Phase.FORCED_REMOVAL
    state.current_player = Player.BLACK
    state.forced_removals_done = 1
    state.board[5][0] = Player.WHITE.value
    state.board[0][5] = Player.BLACK.value
    return state


def _state_counter_removal() -> GameState:
    state = GameState()
    state.phase = Phase.COUNTER_REMOVAL
    state.current_player = Player.BLACK
    state.board[3][3] = Player.WHITE.value
    state.board[2][2] = Player.BLACK.value
    return state


def _state_removal_phase() -> GameState:
    state = GameState()
    state.phase = Phase.REMOVAL
    state.marked_black.add((0, 0))
    state.board[0][0] = Player.BLACK.value
    state.board[5][5] = Player.WHITE.value
    return state


SCENARIOS = [
    ("placement", _state_placement),
    ("movement", _state_movement),
    ("movement_no_moves", _state_movement_no_moves),
    ("mark_selection", _state_mark_selection),
    ("capture_selection", _state_capture_selection),
    ("forced_removal_white", _state_forced_removal_white_turn),
    ("forced_removal_black", _state_forced_removal_black_turn),
    ("counter_removal", _state_counter_removal),
    ("removal_phase", _state_removal_phase),
]


@pytest.mark.parametrize("name,builder", SCENARIOS)
def test_encode_actions_matches_legacy_moves(name, builder):
    state = builder()
    batch = from_game_states([state])

    mask = encode_actions(batch, DEFAULT_ACTION_SPEC).squeeze(0)
    legal_moves = generate_all_legal_moves(state)

    active_indices = mask.nonzero(as_tuple=False).view(-1).tolist()
    assert len(active_indices) == len(legal_moves), f"{name}: mask/legal count mismatch"

    for move in legal_moves:
        idx = action_to_index(move, state.BOARD_SIZE, DEFAULT_ACTION_SPEC)
        assert idx is not None, f"{name}: move not encodable {move}"
        assert mask[idx].item(), f"{name}: mask does not enable index {idx} for move {move}"

        decoded = decode_action_indices(torch.tensor([idx]), batch, DEFAULT_ACTION_SPEC)[0]
        assert decoded == move, f"{name}: decoded move mismatch {decoded} vs {move}"


def test_action_to_index_movement_direction_consistency():
    state = _state_movement()
    batch = from_game_states([state])
    mask = encode_actions(batch, DEFAULT_ACTION_SPEC).squeeze(0)
    legal_moves = [m for m in generate_all_legal_moves(state) if m["action_type"] == "move"]
    indices = [action_to_index(m, state.BOARD_SIZE, DEFAULT_ACTION_SPEC) for m in legal_moves]

    to_positions = {
        decode_action_indices(torch.tensor([idx]), batch, DEFAULT_ACTION_SPEC)[0]["to_position"]
        for idx in indices
    }
    expected = {m["to_position"] for m in legal_moves}
    assert to_positions == expected


def test_fast_encoder_matches_python_when_available():
    states = [builder() for _, builder in SCENARIOS]
    batch = from_game_states(states)

    fast_mask = encode_actions_fast(batch, DEFAULT_ACTION_SPEC)
    if fast_mask is None:
        pytest.skip("C++ fast encoder unavailable on this platform.")

    python_mask = encode_actions_python(batch, DEFAULT_ACTION_SPEC)
    assert torch.equal(fast_mask, python_mask.cpu())
