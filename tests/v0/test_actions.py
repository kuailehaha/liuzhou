"""
Accuracy tests comparing legacy Python move generation with the v0 action encoder.

This test samples random game states by playing random legal moves from the root
position, then checks whether `generate_all_legal_moves` matches the moves
reconstructed from `v0_core.encode_actions_fast`.

Seed / sampling conventions:
    - rng seed: 0x7777
    - num_states: 400
    - max_random_moves: 160

Usage:
    pytest tests/v0/test_actions.py -v
    pytest tests/v0/test_actions.py -v --tb=short  # show tracebacks
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import pytest
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import move_to_key
from v0.python.move_encoder import decode_action_indices, DEFAULT_ACTION_SPEC
from v0.python.state_batch import from_game_states

try:
    import v0_core
except ImportError:
    v0_core = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 0x7777
NUM_STATES = 400
MAX_RANDOM_MOVES = 160

MoveKey = Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_state(rng: random.Random, max_moves: int) -> GameState:
    """Generate a random game state by playing random moves."""
    state = GameState()
    for _ in range(max_moves):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def _canonical_moves(moves: Sequence[dict]) -> Dict[MoveKey, dict]:
    """Convert moves to canonical form for comparison."""
    return {move_to_key(move): move for move in moves}


def _decode_from_mask(state: GameState) -> List[dict]:
    """Decode legal moves from v0_core mask output."""
    batch = from_game_states([state], torch.device("cpu"))
    mask, _ = v0_core.encode_actions_fast(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
        batch.pending_marks_required,
        batch.pending_marks_remaining,
        batch.pending_captures_required,
        batch.pending_captures_remaining,
        batch.forced_removals_done,
        DEFAULT_ACTION_SPEC.placement_dim,
        DEFAULT_ACTION_SPEC.movement_dim,
        DEFAULT_ACTION_SPEC.selection_dim,
        DEFAULT_ACTION_SPEC.auxiliary_dim,
    )
    mask_row = mask[0]
    indices = torch.nonzero(mask_row, as_tuple=False).view(-1)
    if indices.numel() == 0:
        return []

    replicated_states = [state.copy() for _ in range(indices.numel())]
    decode_batch = from_game_states(replicated_states, torch.device("cpu"))
    decoded = decode_action_indices(indices, decode_batch, DEFAULT_ACTION_SPEC)
    return [move for move in decoded if move is not None]


def _compare_once(state: GameState) -> Tuple[List[MoveKey], List[MoveKey]]:
    """Compare legacy and v0 move generation for a single state."""
    legacy_moves = _canonical_moves(generate_all_legal_moves(state))
    v0_moves = _canonical_moves(_decode_from_mask(state))

    missing = sorted(legacy_moves.keys() - v0_moves.keys())
    extra = sorted(v0_moves.keys() - legacy_moves.keys())
    return missing, extra


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_v0_actions_match_legacy() -> None:
    """Verify v0 action encoder produces same legal moves as legacy generator."""
    if v0_core is None:
        pytest.skip("v0_core extension not importable")

    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    phase_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    issues: List[Tuple[int, GameState, List[MoveKey], List[MoveKey]]] = []

    for idx in range(NUM_STATES):
        state = _random_state(rng, MAX_RANDOM_MOVES)
        phase_name = state.phase.name
        phase_counter[phase_name] += 1

        missing, extra = _compare_once(state)
        if missing or extra:
            issue_counter[phase_name] += 1
            issues.append((idx, state, missing, extra))
            if len(issues) >= 10:  # limit detailed output
                break

    total_states = sum(phase_counter.values())
    print(f"\nChecked {total_states} random states (seed=0x{SEED:X})")
    for phase, count in sorted(phase_counter.items()):
        diff = issue_counter.get(phase, 0)
        print(f"  {phase:18s}: {count:5d} states | mismatches: {diff}")

    if issues:
        detail_lines = []
        for idx, state, missing, extra in issues[:10]:
            detail_lines.append(
                f"Case #{idx}: phase={state.phase.name}, move_count={state.move_count}"
            )
            if missing:
                detail_lines.append(f"  Missing (legacy only): {missing[:5]}...")
            if extra:
                detail_lines.append(f"  Extra (v0 only): {extra[:5]}...")
        
        pytest.fail(
            f"Found {len(issues)} mismatches between legacy and v0 move generation:\n"
            + "\n".join(detail_lines)
        )


@pytest.mark.slow
def test_v0_actions_extended() -> None:
    """Extended accuracy test with more states (marked slow)."""
    if v0_core is None:
        pytest.skip("v0_core extension not importable")

    rng = random.Random(SEED + 1)
    torch.manual_seed(SEED + 1)

    num_extended = 1000
    mismatches = 0

    for idx in range(num_extended):
        state = _random_state(rng, MAX_RANDOM_MOVES)
        missing, extra = _compare_once(state)
        if missing or extra:
            mismatches += 1

    print(f"\nExtended check: {num_extended} states, {mismatches} mismatches")
    assert mismatches == 0, f"Found {mismatches} mismatches in extended test"

