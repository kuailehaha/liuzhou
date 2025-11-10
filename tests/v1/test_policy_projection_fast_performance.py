"""
Performance regression test for fast policy projection.

Usage:
  pytest tests/v1/test_policy_projection_fast_performance.py -s > tools/result/policy_proj_perf.txt
Seeds: torch.manual_seed(0xF00DCAFE); repeats=5 for timing.
"""

import time

import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase, Player
from src.move_generator import generate_all_legal_moves
from src.neural_network import get_move_probabilities

from v1.game.move_encoder import DEFAULT_ACTION_SPEC, encode_actions
from v1.game.state_batch import from_game_states
from v1.net.encoding import _project_policy_logits_python
from v1.net.project_policy_logits_fast import project_policy_logits_fast


SEED = 0xF00DCAFE


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


def _run_legacy(states, log_p1, log_p2, log_pmc):
    board_size = states[0].BOARD_SIZE
    for idx, state in enumerate(states):
        legal_moves = generate_all_legal_moves(state)
        get_move_probabilities(
            log_p1[idx],
            log_p2[idx],
            log_pmc[idx],
            legal_moves,
            board_size=board_size,
            device=str(log_p1.device),
        )


def _time_call(fn, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    end = time.perf_counter()
    return (end - start) / repeats


@pytest.mark.slow
def test_policy_projection_performance_cpu():
    spec = DEFAULT_ACTION_SPEC
    base_states = [
        _movement_state_multiple(),
        _forced_removal_single(),
        _placement_state(),
    ]
    batch_size = 96
    states = (base_states * (batch_size // len(base_states) + 1))[:batch_size]

    batch = from_game_states(states)
    legal_mask = encode_actions(batch, spec)

    torch.manual_seed(SEED)
    log_shape = (batch_size, states[0].BOARD_SIZE * states[0].BOARD_SIZE)
    log_p1 = torch.randn(log_shape)
    log_p2 = torch.randn_like(log_p1)
    log_pmc = torch.randn_like(log_p1)

    fast_result = project_policy_logits_fast(log_p1, log_p2, log_pmc, legal_mask, spec)
    if fast_result is None:
        pytest.skip("project_policy_logits_fast extension unavailable")

    # Warm-up runs to avoid cold-start artefacts.
    _project_policy_logits_python((log_p1, log_p2, log_pmc), legal_mask, spec)
    project_policy_logits_fast(log_p1, log_p2, log_pmc, legal_mask, spec)
    _run_legacy(states, log_p1, log_p2, log_pmc)

    with torch.no_grad():
        python_time = _time_call(
            lambda: _project_policy_logits_python((log_p1, log_p2, log_pmc), legal_mask, spec)
        )
        fast_time = _time_call(
            lambda: project_policy_logits_fast(log_p1, log_p2, log_pmc, legal_mask, spec)
        )
        legacy_time = _time_call(lambda: _run_legacy(states, log_p1, log_p2, log_pmc))

    print(
        f"\n[policy_projection] legacy={legacy_time*1e3:.2f} ms | "
        f"tensor-python={python_time*1e3:.2f} ms | tensor-fast={fast_time*1e3:.2f} ms"
    )

    # Note: Performance varies by environment; do not assert ordering here.
