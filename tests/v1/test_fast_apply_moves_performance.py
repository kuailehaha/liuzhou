"""
Performance regression test for batch apply moves fast path.

Usage:
  pytest tests/v1/test_fast_apply_moves_performance.py -s > tools/result/apply_moves_perf.txt
Seeds: random.Random(0xF00DCAFE); max_random_moves=80; repeats=5.
"""

import random
import time

import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase
from src.move_generator import apply_move, generate_all_legal_moves

from v1.game.fast_apply_moves import batch_apply_moves_fast
from v1.game.move_encoder import DEFAULT_ACTION_SPEC, encode_actions
from v1.game.state_batch import from_game_states


SEED = 0xF00DCAFE


ACTION_KIND_PLACEMENT = 1
ACTION_KIND_MOVEMENT = 2
ACTION_KIND_MARK_SELECTION = 3
ACTION_KIND_CAPTURE_SELECTION = 4
ACTION_KIND_FORCED_REMOVAL_SELECTION = 5
ACTION_KIND_COUNTER_REMOVAL_SELECTION = 6
ACTION_KIND_NO_MOVES_REMOVAL_SELECTION = 7
ACTION_KIND_PROCESS_REMOVAL = 8


def _metadata_to_move(code, state: GameState):
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


def _sample_states(num_states: int, max_random_moves: int) -> list[GameState]:
    rng = random.Random(SEED)
    states: list[GameState] = []
    for _ in range(num_states):
        state = GameState()
        steps = rng.randint(0, max_random_moves)
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


def _prepare_actions(states, mask, metadata):
    parent_indices = []
    action_codes = []
    python_moves = []
    for idx, state in enumerate(states):
        legal = mask[idx].nonzero(as_tuple=False).view(-1)
        for action_idx in legal.tolist():
            code = metadata[idx, action_idx]
            parent_indices.append(idx)
            action_codes.append(code.tolist())
            python_moves.append(_metadata_to_move(code, state))
    return parent_indices, action_codes, python_moves


def _python_apply(states, parent_indices, moves):
    return [apply_move(states[parent], move, quiet=True) for parent, move in zip(parent_indices, moves)]


def _legacy_apply(states):
    for state in states:
        moves = generate_all_legal_moves(state)
        for move in moves:
            apply_move(state, move, quiet=True)


def _time_call(fn, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    end = time.perf_counter()
    return (end - start) / repeats


@pytest.mark.slow
def test_fast_apply_moves_performance_cpu():
    states = _sample_states(num_states=10000, max_random_moves=50)
    batch = from_game_states(states, device=torch.device("cpu"))
    spec = DEFAULT_ACTION_SPEC

    result = encode_actions(batch, spec, return_metadata=True)
    if not isinstance(result, tuple):
        pytest.skip("encode_actions metadata unavailable")
    legal_mask, metadata = result
    if metadata is None:
        pytest.skip("encode_actions metadata unavailable")

    parent_indices, action_codes, python_moves = _prepare_actions(states, legal_mask, metadata)
    if not action_codes:
        pytest.skip("No legal actions found to benchmark.")

    parent_tensor = torch.tensor(parent_indices, dtype=torch.long)
    codes_tensor = torch.tensor(action_codes, dtype=torch.int32)

    fast_batch = batch_apply_moves_fast(batch, parent_tensor, codes_tensor)
    if fast_batch is None:
        pytest.skip("fast apply extension unavailable")

    # Warm ups
    _legacy_apply(states)
    _python_apply(states, parent_indices, python_moves)
    batch_apply_moves_fast(batch, parent_tensor, codes_tensor)

    legacy_time = _time_call(lambda: _legacy_apply(states))
    python_time = _time_call(lambda: _python_apply(states, parent_indices, python_moves))
    fast_time = _time_call(lambda: batch_apply_moves_fast(batch, parent_tensor, codes_tensor))

    print(
        f"\n[apply_moves] legacy-python={legacy_time*1e3:.2f} ms | tensor-python={python_time*1e3:.2f} ms | fast-cpp={fast_time*1e3:.2f} ms "
        f"(actions={len(action_codes)})"
    )

    # Note: Performance can vary across hardware, BLAS backends, and interpreter builds.
    # We do not assert ordering here; the benchmark is informational and tracked via logs.
