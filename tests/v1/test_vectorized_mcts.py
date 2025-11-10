"""
Vectorized MCTS smoke tests ensuring deterministic policy outputs.

Usage:
  pytest tests/v1/test_vectorized_mcts.py -q
Seeds: torch.manual_seed(0xF00DCAFE).
"""

import pytest

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from src.move_generator import apply_move

from v1.game.state_batch import from_game_states
from v1.game.move_encoder import decode_action_indices
from v1.mcts.vectorized_mcts import VectorizedMCTS, VectorizedMCTSConfig


torch = pytest.importorskip("torch")

SEED = 0xF00DCAFE


def test_vectorized_mcts_returns_probabilities():
    torch.manual_seed(SEED)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    config = VectorizedMCTSConfig(num_simulations=4, temperature=1.0, exploration_weight=1.0)
    vmcts = VectorizedMCTS(model=model, config=config, device="cpu")

    batch = from_game_states([GameState()])
    policies, mask = vmcts.search(batch)

    assert policies.shape == mask.shape
    assert policies.shape[0] == 1
    assert torch.isclose(policies.sum(), torch.tensor(1.0), atol=1e-4)
    masked_probs = policies[mask]
    assert torch.all(masked_probs >= 0)
    assert torch.isclose(masked_probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_advance_roots_reuses_tree():
    torch.manual_seed(SEED)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    config = VectorizedMCTSConfig(num_simulations=4, temperature=1.0, exploration_weight=1.0)
    vmcts = VectorizedMCTS(model=model, config=config, device="cpu")

    state0 = GameState()
    batch0 = from_game_states([state0])
    policies, mask = vmcts.search(batch0)

    root_before = vmcts._roots[0]

    legal_indices = mask[0].nonzero(as_tuple=False).flatten()
    chosen_idx = int(legal_indices[0].item())

    vmcts.advance_roots(batch0, torch.tensor([chosen_idx], dtype=torch.long))
    root_after = vmcts._roots[0]
    assert root_after is not root_before
    assert root_after.parent is None

    move = decode_action_indices(torch.tensor([chosen_idx]), batch0)[0]
    next_state = apply_move(state0, move, quiet=True)
    batch1 = from_game_states([next_state])

    vmcts.search(batch1)
    assert vmcts._roots[0] is root_after
