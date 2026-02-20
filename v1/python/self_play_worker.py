"""Multiprocess self-play worker entry for v1."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v0.python.move_encoder import DEFAULT_ACTION_SPEC
from .self_play_gpu_runner import SelfPlayV1Stats, self_play_v1_gpu
from .trajectory_buffer import TensorSelfPlayBatch


def _is_stream_capture_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return (
        "stream is capturing" in text
        or "cudaerrorstreamcaptureunsupported" in text
        or "operation not permitted when stream is capturing" in text
    )


def _reserve_memory_anchor(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    raw = str(os.environ.get("V1_SELFPLAY_MEMORY_ANCHOR_MB", "")).strip()
    if not raw:
        return 0
    try:
        anchor_mb = max(0, int(raw))
    except ValueError:
        return 0
    if anchor_mb <= 0:
        return 0
    anchor_bytes = anchor_mb * 1024 * 1024
    try:
        # Keep a small fixed chunk alive to reduce allocator return-to-system churn.
        _anchor = torch.empty((anchor_bytes,), dtype=torch.uint8, device=device)
    except Exception:
        return 0
    # Tie the anchor to a module-level name so it lives for the worker lifetime.
    globals()["_V1_SELFPLAY_MEMORY_ANCHOR"] = _anchor
    return anchor_mb


def _empty_self_play_batch_cpu() -> TensorSelfPlayBatch:
    board = int(GameState.BOARD_SIZE)
    action_dim = int(DEFAULT_ACTION_SPEC.total_dim)
    empty_state = torch.empty(
        (0, int(NUM_INPUT_CHANNELS), board, board),
        dtype=torch.float32,
        device="cpu",
    )
    empty_action = torch.empty((0, action_dim), dtype=torch.float32, device="cpu")
    empty_mask = torch.empty((0, action_dim), dtype=torch.bool, device="cpu")
    empty_value = torch.empty((0,), dtype=torch.float32, device="cpu")
    return TensorSelfPlayBatch(
        state_tensors=empty_state,
        legal_masks=empty_mask,
        policy_targets=empty_action,
        value_targets=empty_value,
        soft_value_targets=empty_value.clone(),
    )


def _concat_self_play_batches_cpu(batches: List[TensorSelfPlayBatch]) -> TensorSelfPlayBatch:
    if not batches:
        return _empty_self_play_batch_cpu()
    if len(batches) == 1:
        return batches[0].to("cpu")
    return TensorSelfPlayBatch(
        state_tensors=torch.cat([b.state_tensors.to("cpu") for b in batches], dim=0),
        legal_masks=torch.cat([b.legal_masks.to("cpu") for b in batches], dim=0),
        policy_targets=torch.cat([b.policy_targets.to("cpu") for b in batches], dim=0),
        value_targets=torch.cat([b.value_targets.to("cpu") for b in batches], dim=0),
        soft_value_targets=torch.cat([b.soft_value_targets.to("cpu") for b in batches], dim=0),
    )


def _merge_self_play_stats(stats_list: List[SelfPlayV1Stats], elapsed_sec: float) -> SelfPlayV1Stats:
    if not stats_list:
        elapsed = max(1e-9, float(elapsed_sec))
        return SelfPlayV1Stats(
            num_games=0,
            num_positions=0,
            black_wins=0,
            white_wins=0,
            draws=0,
            avg_game_length=0.0,
            elapsed_sec=elapsed,
            positions_per_sec=0.0,
            games_per_sec=0.0,
            step_timing_ms={},
            step_timing_ratio={},
            step_timing_calls={},
            mcts_counters={},
        )

    total_games = int(sum(int(s.num_games) for s in stats_list))
    total_positions = int(sum(int(s.num_positions) for s in stats_list))
    black_wins = int(sum(int(s.black_wins) for s in stats_list))
    white_wins = int(sum(int(s.white_wins) for s in stats_list))
    draws = int(sum(int(s.draws) for s in stats_list))
    avg_game_length = float(
        sum(float(s.avg_game_length) * float(s.num_games) for s in stats_list)
        / max(1.0, float(total_games))
    )

    step_timing_ms: Dict[str, float] = {}
    step_timing_calls: Dict[str, int] = {}
    mcts_counters: Dict[str, int] = {}
    for stats in stats_list:
        for k, v in stats.step_timing_ms.items():
            step_timing_ms[str(k)] = float(step_timing_ms.get(str(k), 0.0) + float(v))
        for k, v in stats.step_timing_calls.items():
            step_timing_calls[str(k)] = int(step_timing_calls.get(str(k), 0) + int(v))
        for k, v in stats.mcts_counters.items():
            mcts_counters[str(k)] = int(mcts_counters.get(str(k), 0) + int(v))

    tracked = ("root_puct_ms", "pack_writeback_ms", "self_play_step_ms", "finalize_ms")
    total_tracked = float(sum(float(step_timing_ms.get(k, 0.0)) for k in tracked))
    step_timing_ratio = {
        k: (float(step_timing_ms.get(k, 0.0)) / total_tracked if total_tracked > 0.0 else 0.0)
        for k in tracked
    }
    for k in tracked:
        step_timing_ms.setdefault(k, 0.0)
        step_timing_calls.setdefault(k, 0)

    elapsed = max(1e-9, float(elapsed_sec))
    return SelfPlayV1Stats(
        num_games=total_games,
        num_positions=total_positions,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        avg_game_length=avg_game_length,
        elapsed_sec=elapsed,
        positions_per_sec=float(total_positions / elapsed),
        games_per_sec=float(total_games / elapsed),
        step_timing_ms=step_timing_ms,
        step_timing_ratio=step_timing_ratio,
        step_timing_calls=step_timing_calls,
        mcts_counters=mcts_counters,
    )


def run_self_play_worker(
    *,
    worker_idx: int,
    shard_device: str,
    shard_games: int,
    seed: int,
    model_state_path: str,
    output_path: str,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    soft_value_k: float,
    opening_random_moves: int,
    max_game_plies: int,
    concurrent_games_per_device: int,
) -> Dict[str, Any]:
    """Run one self-play shard inside a dedicated process and persist shard payload."""

    try:
        local_seed = int(seed)
        torch.manual_seed(local_seed)
        dev = torch.device(str(shard_device))
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
            torch.cuda.manual_seed(local_seed)
        anchor_mb_effective = _reserve_memory_anchor(dev)

        state_payload = torch.load(str(model_state_path), map_location="cpu")
        if not isinstance(state_payload, dict):
            raise RuntimeError(
                f"Invalid model_state payload type: {type(state_payload)!r} ({model_state_path})"
            )

        shard_games_i = int(max(0, int(shard_games)))
        if shard_games_i <= 0:
            raise ValueError(f"shard_games must be positive in worker, got {shard_games_i}")
        shard_concurrent = max(1, min(shard_games_i, int(concurrent_games_per_device)))
        # Run shard in fixed-size game chunks so GPU memory scales with concurrency,
        # not with total shard_games.
        shard_chunk_games = shard_concurrent

        model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
        model.load_state_dict(state_payload, strict=True)
        model.to(dev)
        model.eval()

        def _run_once(chunk_games: int) -> tuple[Any, Any]:
            return self_play_v1_gpu(
                model=model,
                num_games=int(chunk_games),
                mcts_simulations=int(mcts_simulations),
                temperature_init=float(temperature_init),
                temperature_final=float(temperature_final),
                temperature_threshold=int(temperature_threshold),
                exploration_weight=float(exploration_weight),
                device=str(dev),
                add_dirichlet_noise=True,
                dirichlet_alpha=float(dirichlet_alpha),
                dirichlet_epsilon=float(dirichlet_epsilon),
                soft_value_k=float(soft_value_k),
                opening_random_moves=int(opening_random_moves),
                max_game_plies=int(max_game_plies),
                sample_moves=True,
                concurrent_games=max(1, min(int(chunk_games), shard_concurrent)),
                verbose=False,
            )

        graph_env_explicit = "V1_FINALIZE_GRAPH" in os.environ
        graph_retry_off = False
        remaining_games = int(shard_games_i)
        batch_chunks: List[TensorSelfPlayBatch] = []
        stats_chunks: List[SelfPlayV1Stats] = []
        started = time.perf_counter()
        while remaining_games > 0:
            chunk_games = min(int(shard_chunk_games), int(remaining_games))
            try:
                chunk_batch, chunk_stats = _run_once(chunk_games)
            except Exception as first_exc:
                if (not graph_env_explicit) and (not graph_retry_off) and _is_stream_capture_error(first_exc):
                    os.environ["V1_FINALIZE_GRAPH"] = "off"
                    graph_retry_off = True
                    chunk_batch, chunk_stats = _run_once(chunk_games)
                else:
                    raise
            batch_chunks.append(chunk_batch.to("cpu"))
            stats_chunks.append(chunk_stats)
            remaining_games -= int(chunk_games)

        samples = _concat_self_play_batches_cpu(batch_chunks)
        stats = _merge_self_play_stats(
            stats_chunks,
            elapsed_sec=max(1e-9, float(time.perf_counter() - started)),
        )

        payload = {
            "state_tensors": samples.state_tensors.detach().cpu(),
            "legal_masks": samples.legal_masks.detach().cpu(),
            "policy_targets": samples.policy_targets.detach().cpu(),
            "value_targets": samples.value_targets.detach().cpu(),
            "soft_value_targets": samples.soft_value_targets.detach().cpu(),
            "stats": stats.to_dict(),
            "metadata": {
                "worker_idx": int(worker_idx),
                "device": str(dev),
                "games": int(shard_games_i),
                "games_per_chunk": int(shard_chunk_games),
                "num_chunks": int(len(batch_chunks)),
                "graph_retry_off": bool(graph_retry_off),
                "memory_anchor_mb": int(anchor_mb_effective),
                "opening_random_moves": int(opening_random_moves),
            },
        }
        os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
        torch.save(payload, str(output_path))

        return {
            "worker_idx": int(worker_idx),
            "device": str(dev),
            "games": int(shard_games_i),
            "output_path": str(output_path),
            "num_samples": int(samples.num_samples),
        }
    except Exception as exc:
        detail = traceback.format_exc()
        raise RuntimeError(
            "v1 self-play process worker failed: "
            f"worker={int(worker_idx)}, device={str(shard_device)}, games={int(shard_games)}\n{detail}"
        ) from exc
