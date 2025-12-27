"""
Accuracy tests for v0 TensorStateBatch conversions.

Validates that v0 state batch round-trips (GameState -> TensorStateBatch -> GameState)
preserve all state fields correctly.

Seed / sampling conventions:
    - rng seed: 0
    - num_samples: 128
    - max_moves: 60

Usage:
    pytest tests/v0/test_state_batch.py -v
"""
from __future__ import annotations

import random
from typing import List

import pytest
import torch

import v0.python.state_batch as state_batch_v0
from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 0
NUM_SAMPLES = 128
MAX_MOVES = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_state(rng: random.Random, max_moves: int) -> GameState:
    """Generate a random game state by playing random moves."""
    state = GameState()
    for _ in range(rng.randint(0, max_moves)):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def _states_equal(a: GameState, b: GameState) -> bool:
    """Check if two game states are equal in all relevant fields."""
    return (
        a.phase == b.phase
        and a.current_player == b.current_player
        and a.board == b.board
        and a.marked_black == b.marked_black
        and a.marked_white == b.marked_white
        and a.pending_marks_required == b.pending_marks_required
        and a.pending_marks_remaining == b.pending_marks_remaining
        and a.pending_captures_required == b.pending_captures_required
        and a.pending_captures_remaining == b.pending_captures_remaining
        and a.forced_removals_done == b.forced_removals_done
        and a.move_count == b.move_count
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_v0_state_batch_roundtrip_cpu() -> None:
    """Test that v0 state batch round-trips correctly on CPU."""
    random.seed(SEED)
    torch.manual_seed(SEED)

    states: List[GameState] = [_random_state(random.Random(SEED + i), MAX_MOVES) 
                                for i in range(NUM_SAMPLES)]

    # Convert to batch and back
    batch = state_batch_v0.from_game_states(states, device="cpu")
    restored = state_batch_v0.to_game_states(batch)

    # Check round-trip correctness
    mismatches = []
    for i, (original, restored_state) in enumerate(zip(states, restored)):
        if not _states_equal(original, restored_state):
            mismatches.append(i)

    if mismatches:
        first_idx = mismatches[0]
        pytest.fail(
            f"State mismatch at index {first_idx}\n"
            f"Original: phase={states[first_idx].phase}, player={states[first_idx].current_player}\n"
            f"Restored: phase={restored[first_idx].phase}, player={restored[first_idx].current_player}"
        )

    # Verify device
    assert batch.board.device.type == "cpu"

    print(f"\n[test_v0_state_batch_roundtrip_cpu] success: {NUM_SAMPLES} samples | "
          f"board shape={tuple(batch.board.shape)}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_v0_state_batch_roundtrip_cuda() -> None:
    """Test that v0 state batch round-trips correctly on CUDA."""
    random.seed(SEED)
    torch.manual_seed(SEED)

    states: List[GameState] = [_random_state(random.Random(SEED + i), MAX_MOVES) 
                                for i in range(NUM_SAMPLES)]

    # Convert to batch on CUDA
    batch = state_batch_v0.from_game_states(states, device="cuda")

    # Verify device
    assert batch.board.device.type == "cuda"

    # Convert back (to_game_states should handle CUDA tensors)
    restored = state_batch_v0.to_game_states(batch)

    # Check round-trip correctness
    mismatches = []
    for i, (original, restored_state) in enumerate(zip(states, restored)):
        if not _states_equal(original, restored_state):
            mismatches.append(i)

    assert not mismatches, f"Found {len(mismatches)} mismatches on CUDA round-trip"

    print(f"\n[test_v0_state_batch_roundtrip_cuda] success: {NUM_SAMPLES} samples | "
          f"board shape={tuple(batch.board.shape)}")


def test_v0_state_batch_empty() -> None:
    """Test that empty state list raises ValueError (as expected by v0 implementation)."""
    with pytest.raises(ValueError, match="at least one"):
        state_batch_v0.from_game_states([], device="cpu")


def test_v0_state_batch_single() -> None:
    """Test handling of single state."""
    state = GameState()
    batch = state_batch_v0.from_game_states([state], device="cpu")
    restored = state_batch_v0.to_game_states(batch)
    
    assert len(restored) == 1
    assert _states_equal(state, restored[0])

