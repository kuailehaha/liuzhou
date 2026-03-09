"""Multiprocess self-play worker entry for v1."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from .self_play_gpu_runner import SelfPlayV1Stats, self_play_v1_gpu
from .self_play_storage import (
    estimate_bytes_per_sample,
    plan_sample_ranges,
    save_self_play_payload,
    slice_batch_cpu,
)


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


def _summarize_scalar_targets(values: torch.Tensor) -> Dict[str, Any]:
    total = int(values.numel())
    if total <= 0:
        return {
            "total": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "nonzero_count": 0,
            "zero_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "nonzero_ratio": 0.0,
            "sum_abs": 0.0,
            "abs_mean": 0.0,
            "near_zero_count": 0,
            "near_zero_ratio": 0.0,
            "ge_abs_0p05_count": 0,
            "ge_abs_0p05_ratio": 0.0,
            "ge_abs_0p10_count": 0,
            "ge_abs_0p10_ratio": 0.0,
            "ge_abs_0p20_count": 0,
            "ge_abs_0p20_ratio": 0.0,
        }
    finite_mask = torch.isfinite(values)
    finite_count = int(torch.count_nonzero(finite_mask).item())
    nonfinite_count = int(total - finite_count)
    finite_values = values[finite_mask] if finite_count > 0 else values.new_empty((0,))
    positive = int(torch.count_nonzero(finite_values > 0).item())
    negative = int(torch.count_nonzero(finite_values < 0).item())
    nonzero = int(positive + negative)
    zero = int(finite_count - nonzero)
    abs_values = finite_values.abs()
    sum_abs = float(abs_values.sum().item()) if finite_count > 0 else 0.0
    near_zero_count = int(torch.count_nonzero(abs_values <= 1e-6).item()) if finite_count > 0 else 0
    ge_abs_0p05_count = int(torch.count_nonzero(abs_values >= 0.05).item()) if finite_count > 0 else 0
    ge_abs_0p10_count = int(torch.count_nonzero(abs_values >= 0.10).item()) if finite_count > 0 else 0
    ge_abs_0p20_count = int(torch.count_nonzero(abs_values >= 0.20).item()) if finite_count > 0 else 0
    return {
        "total": total,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "nonzero_count": nonzero,
        "zero_count": zero,
        "positive_count": positive,
        "negative_count": negative,
        "nonzero_ratio": float(nonzero / max(1, finite_count)),
        "sum_abs": sum_abs,
        "abs_mean": float(sum_abs / max(1, finite_count)),
        "near_zero_count": near_zero_count,
        "near_zero_ratio": float(near_zero_count / max(1, finite_count)),
        "ge_abs_0p05_count": ge_abs_0p05_count,
        "ge_abs_0p05_ratio": float(ge_abs_0p05_count / max(1, finite_count)),
        "ge_abs_0p10_count": ge_abs_0p10_count,
        "ge_abs_0p10_ratio": float(ge_abs_0p10_count / max(1, finite_count)),
        "ge_abs_0p20_count": ge_abs_0p20_count,
        "ge_abs_0p20_ratio": float(ge_abs_0p20_count / max(1, finite_count)),
    }


def _merge_target_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "total": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "nonzero_count": 0,
        "zero_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "nonzero_ratio": 0.0,
        "sum_abs": 0.0,
        "abs_mean": 0.0,
        "near_zero_count": 0,
        "near_zero_ratio": 0.0,
        "ge_abs_0p05_count": 0,
        "ge_abs_0p05_ratio": 0.0,
        "ge_abs_0p10_count": 0,
        "ge_abs_0p10_ratio": 0.0,
        "ge_abs_0p20_count": 0,
        "ge_abs_0p20_ratio": 0.0,
    }
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        for key in (
            "total",
            "finite_count",
            "nonfinite_count",
            "nonzero_count",
            "zero_count",
            "positive_count",
            "negative_count",
            "near_zero_count",
            "ge_abs_0p05_count",
            "ge_abs_0p10_count",
            "ge_abs_0p20_count",
        ):
            merged[key] = int(merged.get(key, 0) + int(summary.get(key, 0) or 0))
        merged["sum_abs"] = float(merged.get("sum_abs", 0.0) + float(summary.get("sum_abs", 0.0) or 0.0))
    finite = int(merged.get("finite_count", 0))
    nonzero = int(merged.get("nonzero_count", 0))
    merged["nonzero_ratio"] = float(nonzero / max(1, finite))
    merged["abs_mean"] = float(float(merged.get("sum_abs", 0.0)) / max(1, finite))
    merged["near_zero_ratio"] = float(int(merged.get("near_zero_count", 0)) / max(1, finite))
    merged["ge_abs_0p05_ratio"] = float(int(merged.get("ge_abs_0p05_count", 0)) / max(1, finite))
    merged["ge_abs_0p10_ratio"] = float(int(merged.get("ge_abs_0p10_count", 0)) / max(1, finite))
    merged["ge_abs_0p20_ratio"] = float(int(merged.get("ge_abs_0p20_count", 0)) / max(1, finite))
    return merged


def _merge_self_play_stats(stats_list: List[SelfPlayV1Stats], elapsed_sec: float) -> SelfPlayV1Stats:
    def _normalize_piece_delta_buckets(raw: Any) -> Dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, int] = {}
        for delta in range(-18, 19):
            key = str(delta)
            try:
                out[key] = int(raw.get(key, 0) or 0)
            except Exception:
                out[key] = 0
        return out

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
            piece_delta_buckets={str(delta): 0 for delta in range(-18, 19)},
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
    piece_delta_buckets: Dict[str, int] = {str(delta): 0 for delta in range(-18, 19)}
    for stats in stats_list:
        for k, v in stats.step_timing_ms.items():
            step_timing_ms[str(k)] = float(step_timing_ms.get(str(k), 0.0) + float(v))
        for k, v in stats.step_timing_calls.items():
            step_timing_calls[str(k)] = int(step_timing_calls.get(str(k), 0) + int(v))
        for k, v in stats.mcts_counters.items():
            mcts_counters[str(k)] = int(mcts_counters.get(str(k), 0) + int(v))
        buckets = _normalize_piece_delta_buckets(stats.piece_delta_buckets)
        for key, value in buckets.items():
            piece_delta_buckets[key] = int(piece_delta_buckets.get(key, 0) + int(value))

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
        piece_delta_buckets=piece_delta_buckets,
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
    soft_label_alpha: float,
    target_samples_per_shard: int,
    chunk_target_bytes: int,
    chunk_output_dir: str,
    chunk_file_prefix: str,
    chunk_file_ext: str,
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
        stats_chunks: List[SelfPlayV1Stats] = []
        value_summaries: List[Dict[str, Any]] = []
        soft_summaries: List[Dict[str, Any]] = []
        mixed_summaries: List[Dict[str, Any]] = []
        saved_chunk_files: List[str] = []
        saved_chunk_sizes: List[int] = []
        weighted_bps_num = 0
        weighted_bps_den = 0
        saved_chunk_idx = 0
        alpha = float(max(0.0, min(1.0, soft_label_alpha)))
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
            chunk_batch_cpu = chunk_batch.to("cpu")
            stats_chunks.append(chunk_stats)
            value_summaries.append(_summarize_scalar_targets(chunk_batch_cpu.value_targets))
            soft_summaries.append(_summarize_scalar_targets(chunk_batch_cpu.soft_value_targets))
            mixed_values = torch.clamp(
                (1.0 - alpha) * chunk_batch_cpu.value_targets + alpha * chunk_batch_cpu.soft_value_targets,
                min=-1.0,
                max=1.0,
            )
            mixed_summaries.append(_summarize_scalar_targets(mixed_values))

            bytes_per_sample = estimate_bytes_per_sample(chunk_batch_cpu)
            weighted_bps_num += int(bytes_per_sample * max(1, int(chunk_batch_cpu.num_samples)))
            weighted_bps_den += int(max(1, int(chunk_batch_cpu.num_samples)))
            chunk_ranges = plan_sample_ranges(
                total_samples=int(chunk_batch_cpu.num_samples),
                num_shards=1,
                target_samples_per_shard=int(target_samples_per_shard),
                chunk_target_bytes=int(chunk_target_bytes),
                bytes_per_sample=int(bytes_per_sample),
            )
            for start_idx, end_idx in chunk_ranges:
                chunk_name = (
                    f"{chunk_file_prefix}.chunk{int(saved_chunk_idx):05d}{chunk_file_ext}"
                )
                chunk_path = os.path.join(str(chunk_output_dir), chunk_name)
                chunk_meta = {
                    "payload_format": "v1_sharded_shard",
                    "worker_idx": int(worker_idx),
                    "device": str(dev),
                    "games": int(shard_games_i),
                    "games_per_chunk": int(shard_chunk_games),
                    "num_selfplay_batches": int(len(stats_chunks)),
                    "saved_chunk_index": int(saved_chunk_idx),
                    "graph_retry_off": bool(graph_retry_off),
                    "memory_anchor_mb": int(anchor_mb_effective),
                    "opening_random_moves": int(opening_random_moves),
                    "source_worker_manifest": os.path.basename(str(output_path)),
                }
                save_self_play_payload(
                    path=chunk_path,
                    samples=slice_batch_cpu(chunk_batch_cpu, start=int(start_idx), end=int(end_idx)),
                    stats_payload={},
                    metadata=chunk_meta,
                )
                saved_chunk_files.append(chunk_name)
                saved_chunk_sizes.append(int(end_idx - start_idx))
                saved_chunk_idx += 1
            remaining_games -= int(chunk_games)

        stats = _merge_self_play_stats(
            stats_chunks,
            elapsed_sec=max(1e-9, float(time.perf_counter() - started)),
        )
        avg_bps = int(weighted_bps_num // max(1, weighted_bps_den))
        payload = {
            "payload_format": "v1_worker_chunk_manifest",
            "version": 1,
            "num_samples": int(sum(saved_chunk_sizes)),
            "num_shards": int(len(saved_chunk_files)),
            "shard_files": list(saved_chunk_files),
            "shard_sizes": list(saved_chunk_sizes),
            "chunk_target_bytes": int(chunk_target_bytes),
            "avg_bytes_per_sample": int(avg_bps),
            "stats": stats.to_dict(),
            "value_target_summary": _merge_target_summaries(value_summaries),
            "soft_value_target_summary": _merge_target_summaries(soft_summaries),
            "mixed_value_target_summary": _merge_target_summaries(mixed_summaries),
            "metadata": {
                "worker_idx": int(worker_idx),
                "device": str(dev),
                "games": int(shard_games_i),
                "games_per_chunk": int(shard_chunk_games),
                "num_selfplay_batches": int(len(stats_chunks)),
                "saved_chunks": int(len(saved_chunk_files)),
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
            "num_samples": int(sum(saved_chunk_sizes)),
            "saved_chunks": int(len(saved_chunk_files)),
        }
    except Exception as exc:
        detail = traceback.format_exc()
        raise RuntimeError(
            "v1 self-play process worker failed: "
            f"worker={int(worker_idx)}, device={str(shard_device)}, games={int(shard_games)}\n{detail}"
        ) from exc
