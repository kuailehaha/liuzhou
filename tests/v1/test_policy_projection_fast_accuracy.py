import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase, Player
from src.move_generator import generate_all_legal_moves
from src.neural_network import get_move_probabilities

from v1.game.move_encoder import DEFAULT_ACTION_SPEC, action_to_index, encode_actions
from v1.game.state_batch import from_game_states
from v1.net.encoding import _project_policy_logits_python
from v1.net.project_policy_logits_fast import project_policy_logits_fast


def _movement_state_multiple() -> GameState:
    state = GameState()
    state.phase = Phase.MOVEMENT
    state.current_player = Player.BLACK
    state.board[2][2] = Player.BLACK.value
    state.board[2][3] = 0
    state.board[2][1] = 0
    state.board[3][2] = 0
    state.board[1][2] = 0
    return state


def _forced_removal_single() -> GameState:
    state = GameState()
    state.phase = Phase.FORCED_REMOVAL
    state.current_player = Player.WHITE
    state.forced_removals_done = 0
    state.board = [[0] * state.BOARD_SIZE for _ in range(state.BOARD_SIZE)]
    state.board[0][0] = Player.BLACK.value
    state.board[5][5] = Player.WHITE.value
    return state


def _placement_state() -> GameState:
    return GameState()


def test_fast_extension_matches_python_and_legacy():
    spec = DEFAULT_ACTION_SPEC
    states = [
        _movement_state_multiple(),
        _forced_removal_single(),
        _placement_state(),
    ]
    batch = from_game_states(states)
    legal_mask = encode_actions(batch, spec)

    torch.manual_seed(2048)
    log_shape = (len(states), states[0].BOARD_SIZE * states[0].BOARD_SIZE)
    log_p1 = torch.randn(log_shape)
    log_p2 = torch.randn_like(log_p1)
    log_pmc = torch.randn_like(log_p1)

    fast_result = project_policy_logits_fast(log_p1, log_p2, log_pmc, legal_mask, spec)
    if fast_result is None:
        pytest.skip("project_policy_logits_fast extension unavailable")
    probs_fast, logits_fast = fast_result

    probs_python, logits_python = _project_policy_logits_python(
        (log_p1, log_p2, log_pmc), legal_mask, spec
    )

    torch.testing.assert_close(probs_fast, probs_python, atol=1e-6, rtol=0)
    torch.testing.assert_close(logits_fast, logits_python, atol=1e-6, rtol=0)

    for idx, state in enumerate(states):
        legal_moves = generate_all_legal_moves(state)
        legacy_probs, legacy_raw = get_move_probabilities(
            log_p1[idx],
            log_p2[idx],
            log_pmc[idx],
            legal_moves,
            board_size=state.BOARD_SIZE,
            device=str(log_p1.device),
        )
        legacy_probs_tensor = torch.tensor(legacy_probs, dtype=probs_fast.dtype)
        legacy_raw_tensor = legacy_raw.to(logits_fast.device, dtype=logits_fast.dtype)

        for mv_idx, move in enumerate(legal_moves):
            flat_index = action_to_index(move, state.BOARD_SIZE, spec)
            assert flat_index is not None
            torch.testing.assert_close(
                probs_fast[idx, flat_index],
                legacy_probs_tensor[mv_idx],
                atol=1e-6,
                rtol=0,
            )
            torch.testing.assert_close(
                logits_fast[idx, flat_index],
                legacy_raw_tensor[mv_idx],
                atol=1e-6,
                rtol=0,
            )
