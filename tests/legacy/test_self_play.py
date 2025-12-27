"""
Integration test for legacy self-play pipeline.

Usage:
    pytest tests/legacy/test_self_play.py -v
"""
import torch

from src.game_state import GameState
from src.mcts import self_play
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


def test_self_play_generation_smoke():
    """Generate a small batch of self-play data to ensure pipeline wiring works."""
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    board_size = GameState.BOARD_SIZE
    model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()

    games_requested = 1
    mcts_simulations = 10

    training_data = self_play(
        model=model,
        num_games=games_requested,
        mcts_simulations=mcts_simulations,
        device=device,
        verbose=False,
    )

    assert len(training_data) == games_requested
    for game_states, game_policies, result, soft_value in training_data:
        assert game_states, "Expected at least one state per game"
        assert len(game_states) == len(game_policies)
        assert result in (-1, 0, 1)
        assert -1.0 <= soft_value <= 1.0

