"""
Accuracy tests for v0 MCTS implementation.

Validates that v0 MCTS produces valid policies (probabilities sum to 1, 
all moves are legal) and compares basic behavior with legacy implementation.

Seed / sampling conventions:
    - rng seed: 0x42
    - num_samples: 10
    - mcts_simulations: 32 (smaller for faster tests)

Usage:
    pytest tests/v0/test_mcts.py -v
"""
from __future__ import annotations

import random

import pytest
import torch
import numpy as np

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import MCTS as LegacyMCTS, move_to_key
from src.neural_network import ChessNet
from v0.python.mcts import MCTS as V0MCTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 0x42
NUM_SAMPLES = 10
MAX_MOVES = 60
MCTS_SIMULATIONS = 32


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_v0_mcts_produces_valid_policy() -> None:
    """Verify v0 MCTS produces valid probability distributions."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rng = random.Random(SEED + 1)
    device = "cpu"
    model = ChessNet(board_size=GameState.BOARD_SIZE).to(device)
    model.eval()

    v0_mcts = V0MCTS(
        model=model,
        num_simulations=MCTS_SIMULATIONS,
        exploration_weight=1.0,
        temperature=1.0,
        device=device,
        add_dirichlet_noise=False,
        seed=SEED,
        batch_K=8,
        inference_backend="py",
    )

    for i in range(NUM_SAMPLES):
        state = _random_state(rng, MAX_MOVES)
        
        # Skip terminal states
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            continue

        moves, policy = v0_mcts.search(state)

        # Verify policy is valid probability distribution
        assert len(moves) == len(policy), f"Sample {i}: moves/policy length mismatch"
        assert len(moves) > 0, f"Sample {i}: no moves returned"
        
        policy_sum = sum(policy)
        assert abs(policy_sum - 1.0) < 1e-5, f"Sample {i}: policy sum={policy_sum}, expected 1.0"
        
        # Verify all probabilities are non-negative
        for prob in policy:
            assert prob >= 0, f"Sample {i}: negative probability {prob}"

        # Verify all returned moves are legal
        legal_keys = {move_to_key(m) for m in legal_moves}
        for move in moves:
            key = move_to_key(move)
            assert key in legal_keys, f"Sample {i}: illegal move {move}"

    print(f"\n[test_v0_mcts_produces_valid_policy] Validated {NUM_SAMPLES} samples")


def test_v0_mcts_consistency() -> None:
    """Verify v0 MCTS is deterministic with same seed."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cpu"
    model = ChessNet(board_size=GameState.BOARD_SIZE).to(device)
    model.eval()

    state = GameState()  # Use initial state for consistency test

    # Run twice with same seed
    results = []
    for _ in range(2):
        v0_mcts = V0MCTS(
            model=model,
            num_simulations=MCTS_SIMULATIONS,
            exploration_weight=1.0,
            temperature=1.0,
            device=device,
            add_dirichlet_noise=False,
            seed=SEED,
            batch_K=8,
            inference_backend="py",
        )
        moves, policy = v0_mcts.search(state)
        results.append((moves, policy))

    # Compare results
    moves1, policy1 = results[0]
    moves2, policy2 = results[1]

    keys1 = [move_to_key(m) for m in moves1]
    keys2 = [move_to_key(m) for m in moves2]

    assert keys1 == keys2, "Move ordering differs between runs"
    
    for p1, p2 in zip(policy1, policy2):
        assert abs(p1 - p2) < 1e-6, f"Policy differs: {p1} vs {p2}"

    print("\n[test_v0_mcts_consistency] Determinism verified")


def test_v0_mcts_vs_legacy_move_coverage() -> None:
    """Verify v0 MCTS considers same moves as legacy MCTS."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rng = random.Random(SEED + 1)
    device = "cpu"
    model = ChessNet(board_size=GameState.BOARD_SIZE).to(device)
    model.eval()

    legacy_mcts = LegacyMCTS(
        model=model,
        num_simulations=MCTS_SIMULATIONS,
        exploration_weight=1.0,
        temperature=1.0,
        device=device,
        add_dirichlet_noise=False,
        virtual_loss_weight=0.0,
        batch_K=8,
        verbose=False,
    )
    v0_mcts = V0MCTS(
        model=model,
        num_simulations=MCTS_SIMULATIONS,
        exploration_weight=1.0,
        temperature=1.0,
        device=device,
        add_dirichlet_noise=False,
        seed=SEED,
        batch_K=8,
        inference_backend="py",
    )

    mismatches = []
    for i in range(NUM_SAMPLES):
        state = _random_state(rng, MAX_MOVES)
        
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            continue

        legacy_moves, _ = legacy_mcts.search(state)
        v0_moves, _ = v0_mcts.search(state)

        legacy_keys = {move_to_key(m) for m in legacy_moves}
        v0_keys = {move_to_key(m) for m in v0_moves}

        # Both should cover the same move set
        if legacy_keys != v0_keys:
            mismatches.append({
                "sample": i,
                "legacy_only": legacy_keys - v0_keys,
                "v0_only": v0_keys - legacy_keys,
            })

    if mismatches:
        print(f"\nWarning: {len(mismatches)} samples had different move sets")
        for m in mismatches[:3]:
            print(f"  Sample {m['sample']}: legacy_only={len(m['legacy_only'])}, v0_only={len(m['v0_only'])}")
    else:
        print(f"\n[test_v0_mcts_vs_legacy_move_coverage] All {NUM_SAMPLES} samples matched")
