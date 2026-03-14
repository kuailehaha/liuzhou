"""Smoke-level regression for v1 tensor-native pipeline."""

from __future__ import annotations

import json
import math

import pytest
import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.mcts_gpu import GpuStateBatch, V1RootMCTS
from v1.python.self_play_worker import run_self_play_worker
from v1.python.self_play_gpu_runner import self_play_v1_gpu
from v1.python.train_bridge import train_network_from_tensors
from v1.train import _save_self_play_payload_sharded, train_pipeline_v1


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
    board[0, 0, 0] = 1  # white pieces are zero, but mark-selection is not terminal
    board[1, 0, 0] = 1
    board[1, 0, 1] = -1
    board[2, 0, 0] = 1
    board[2, 0, 1] = -1
    zeros = torch.zeros((3,), dtype=torch.int64)
    batch = GpuStateBatch(
        board=board,
        marks_black=torch.zeros((3, 6, 6), dtype=torch.bool),
        marks_white=torch.zeros((3, 6, 6), dtype=torch.bool),
        phase=torch.tensor([2, 4, 1], dtype=torch.int64),
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


@pytest.mark.parametrize("chunk_target_bytes", [0, 1024])
def test_process_worker_always_emits_chunk_manifest(tmp_path, chunk_target_bytes: int) -> None:
    try:
        import v0_core  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"v0_core import failed: {exc}")

    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model_state_path = tmp_path / "model_state.pt"
    torch.save(model.state_dict(), model_state_path)

    manifest_path = tmp_path / f"worker_manifest_{chunk_target_bytes}.pt"
    row = run_self_play_worker(
        worker_idx=0,
        shard_device="cpu",
        shard_games=1,
        seed=7,
        model_state_path=str(model_state_path),
        output_path=str(manifest_path),
        mcts_simulations=4,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=4,
        exploration_weight=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        opening_random_moves=2,
        max_game_plies=64,
        concurrent_games_per_device=1,
        soft_label_alpha=0.0,
        target_samples_per_shard=0,
        chunk_target_bytes=int(chunk_target_bytes),
        chunk_output_dir=str(tmp_path),
        chunk_file_prefix=f"worker{chunk_target_bytes}",
    )

    payload = torch.load(manifest_path, map_location="cpu")
    assert row["output_path"] == str(manifest_path)
    assert payload["payload_format"] == "v1_worker_chunk_manifest"
    assert int(payload["num_shards"]) >= 1
    assert len(payload["shard_files"]) == int(payload["num_shards"])
    for shard_name in payload["shard_files"]:
        assert (tmp_path / str(shard_name)).is_file()


def test_stage_train_manifest_input_auto_streams(tmp_path) -> None:
    try:
        import v0_core  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"v0_core import failed: {exc}")

    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.eval()
    samples, stats = self_play_v1_gpu(
        model=model,
        num_games=1,
        mcts_simulations=4,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=4,
        exploration_weight=1.0,
        device="cpu",
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        opening_random_moves=2,
        max_game_plies=64,
        sample_moves=True,
        concurrent_games=1,
        verbose=False,
    )

    manifest_path = tmp_path / "selfplay_manifest.pt"
    saved = _save_self_play_payload_sharded(
        path=str(manifest_path),
        samples=samples.to("cpu"),
        stats=stats,
        metadata={"stage": "test"},
        num_shards=2,
        chunk_target_bytes=0,
    )
    assert saved >= 1

    metrics_path = tmp_path / "train_metrics.json"
    checkpoint_dir = tmp_path / "ckpt"
    train_pipeline_v1(
        stage="train",
        self_play_input=str(manifest_path),
        checkpoint_dir=str(checkpoint_dir),
        metrics_output=str(metrics_path),
        checkpoint_name="model_iter_001.pt",
        device="cpu",
        devices="cpu",
        train_devices="cpu",
        infer_devices="cpu",
        train_strategy="none",
        batch_size=min(8, max(1, int(samples.num_samples))),
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        soft_label_alpha=0.0,
        streaming_load=False,
        streaming_workers=0,
    )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics[0]["streaming"] is True
