import pytest
import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from v1.mcts.vectorized_mcts import VectorizedMCTSConfig
from v1.self_play.runner import SelfPlayConfig, run_self_play


torch = pytest.importorskip("torch")


def test_run_self_play_shapes():
    torch.manual_seed(0)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)

    cfg = SelfPlayConfig(
        mcts=VectorizedMCTSConfig(num_simulations=2, batch_leaves=1, temperature=1.0),
        temperature_init=0.0,
        temperature_final=0.0,
        temperature_threshold=0,
        max_moves=6,
    )

    result = run_self_play(model, batch_size=2, device="cpu", config=cfg)

    assert result.states.shape[0] == 2
    assert result.policies.shape[0] == 2
    assert result.player_signs.shape[:2] == result.mask.shape[:2]
    assert result.legal_masks.shape[:2] == result.mask.shape[:2]
    assert result.legal_masks.shape[2] == cfg.mcts.action_spec.total_dim
    assert result.mask.shape[0] == 2
    assert result.states.shape[1] == result.policies.shape[1] == result.mask.shape[1]
    assert result.policies.shape[2] == cfg.mcts.action_spec.total_dim
    assert result.results.shape == (2,)
    assert result.soft_values.shape == (2,)
    assert torch.all(result.lengths == result.mask.sum(dim=1))
    assert torch.all(result.player_signs[result.mask] != 0)
