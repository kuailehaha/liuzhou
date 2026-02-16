"""Smoke-level regression for v1 tensor-native pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.self_play_gpu_runner import self_play_v1_gpu
from v1.python.train_bridge import train_network_from_tensors


@pytest.mark.smoke
def test_v1_tensor_pipeline_smoke() -> None:
    try:
        import v0_core  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"v0_core import failed: {exc}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()

    samples, stats = self_play_v1_gpu(
        model=model,
        num_games=1,
        mcts_simulations=4,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=4,
        exploration_weight=1.0,
        device=device,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        max_game_plies=96,
        sample_moves=True,
        verbose=False,
    )

    assert stats.num_games == 1
    assert samples.num_samples > 0
    assert samples.state_tensors.shape[1:] == (NUM_INPUT_CHANNELS, GameState.BOARD_SIZE, GameState.BOARD_SIZE)
    assert samples.legal_masks.shape[0] == samples.num_samples
    assert samples.policy_targets.shape[0] == samples.num_samples
    assert samples.value_targets.shape[0] == samples.num_samples

    model, train_metrics = train_network_from_tensors(
        model=model,
        samples=samples,
        batch_size=min(64, samples.num_samples),
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        soft_label_alpha=0.0,
        policy_draw_weight=1.0,
        device=device,
        use_amp=torch.cuda.is_available(),
    )

    epoch_stats = train_metrics.get("epoch_stats", [])
    assert len(epoch_stats) == 1
    avg_loss = epoch_stats[0].get("avg_loss")
    assert avg_loss is not None
    assert math.isfinite(float(avg_loss))
