"""Smoke-level regression for v1 tensor-native pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.mcts_gpu import GpuStateBatch, V1RootMCTS
from v1.python.self_play_gpu_runner import self_play_v1_gpu
from v1.python.train_bridge import train_network_from_tensors


def test_v1_soft_tan_range_and_sign() -> None:
    board = torch.zeros((3, 6, 6), dtype=torch.int8)
    board[0, :2, :] = 1
    board[1, :2, :] = -1
    soft = V1RootMCTS._soft_tan_from_board_black(board, soft_value_k=2.0)
    assert tuple(soft.shape) == (3,)
    assert torch.all(soft <= 1.0 + 1e-6)
    assert torch.all(soft >= -1.0 - 1e-6)
    assert float(soft[0].item()) > float(soft[2].item())
    assert float(soft[2].item()) > float(soft[1].item())


def test_v1_terminal_mask_next_state() -> None:
    board = torch.zeros((3, 6, 6), dtype=torch.int8)
    board[0, 0, 0] = 1  # white pieces are zero but mark-selection is still pre-movement
    board[1, 0, 0] = 1
    board[1, 0, 1] = -1
    board[2, 0, 0] = 1
    board[2, 0, 1] = -1
    zeros = torch.zeros((3,), dtype=torch.int64)
    batch = GpuStateBatch(
        board=board,
        marks_black=torch.zeros((3, 6, 6), dtype=torch.bool),
        marks_white=torch.zeros((3, 6, 6), dtype=torch.bool),
        phase=torch.tensor([2, 2, 1], dtype=torch.int64),
        current_player=torch.ones((3,), dtype=torch.int64),
        pending_marks_required=zeros.clone(),
        pending_marks_remaining=zeros.clone(),
        pending_captures_required=zeros.clone(),
        pending_captures_remaining=zeros.clone(),
        forced_removals_done=zeros.clone(),
        move_count=torch.tensor([0, GameState.MAX_MOVE_COUNT, 0], dtype=torch.int64),
        moves_since_capture=torch.tensor([0, 0, GameState.NO_CAPTURE_DRAW_LIMIT], dtype=torch.int64),
    )
    terminal = V1RootMCTS._terminal_mask_from_next_state(batch)
    assert terminal.tolist() == [False, True, True]


def test_v1_child_value_perspective_alignment() -> None:
    child_values = torch.tensor([0.2, -0.5, 0.8, -0.1], dtype=torch.float32)
    parent_players = torch.tensor([1, 1, -1, -1], dtype=torch.int64)
    child_players = torch.tensor([1, -1, -1, 1], dtype=torch.int64)
    # Keep sign when side-to-move is unchanged; flip sign only on turn switch.
    aligned = V1RootMCTS._child_values_to_parent_perspective(
        child_values=child_values,
        parent_players=parent_players,
        child_players=child_players,
    )
    expected = torch.tensor([0.2, 0.5, 0.8, 0.1], dtype=torch.float32)
    assert torch.allclose(aligned, expected, atol=1e-6, rtol=0.0)


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
        opening_random_moves=2,
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
    assert int(stats.mcts_counters.get("forced_uniform_pick_count", 0)) > 0

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
