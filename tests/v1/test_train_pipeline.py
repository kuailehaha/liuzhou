"""
Training pipeline smoke tests for the tensorized implementation.

Usage:
  pytest tests/v1/test_train_pipeline.py -q
Seeds: torch.manual_seed(0xF00DCAFE).
"""

import pytest
import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from v1.train.pipeline import (
    TrainingLoopConfig,
    generate_training_data,
    train_one_iteration,
    training_loop,
)


torch = pytest.importorskip("torch")

SEED = 0xF00DCAFE


def _make_model():
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    return model


def test_generate_training_data_and_train_iteration():
    torch.manual_seed(SEED)
    model = _make_model()
    cfg = TrainingLoopConfig(batch_size=2, epochs=1, learning_rate=1e-3)

    rollout = generate_training_data(model, cfg, device="cpu")
    assert rollout.states.ndim == 4
    assert rollout.policies.ndim == 2
    assert rollout.legal_masks.shape == rollout.policies.shape

    if rollout.states.shape[0] == 0:
        pytest.skip("Self-play returned no positions; rerun test.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    metrics = train_one_iteration(model, optimizer, rollout, cfg, device="cpu")

    assert metrics["samples"] >= rollout.states.shape[0]
    assert metrics["loss"] >= 0.0


def test_training_loop_runs_iterations():
    torch.manual_seed(SEED)
    model = _make_model()
    cfg = TrainingLoopConfig(batch_size=2, epochs=1, learning_rate=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = list(training_loop(model, optimizer, iterations=1, config=cfg, device="cpu"))
    assert len(history) == 1
    assert "loss" in history[0]
