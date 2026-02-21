"""v1 training entrypoint: GPU-first self-play + tensor-native training."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.mcts_gpu import TOTAL_ACTION_DIM
from v1.python.self_play_gpu_runner import SelfPlayV1Stats, self_play_v1_gpu
from v1.python.self_play_worker import run_self_play_worker
from v1.python.train_bridge import train_network_from_tensors
from v1.python.trajectory_buffer import TensorSelfPlayBatch


def _dist_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _dist_rank() -> int:
    if _dist_ready():
        return int(dist.get_rank())
    return 0


def _is_rank0() -> bool:
    return _dist_rank() == 0


def _print_rank0(message: str) -> None:
    if _is_rank0():
        print(message)


def _normalize_train_strategy(train_strategy: str) -> str:
    raw = str(train_strategy).strip().lower()
    if raw in {"dp", "data_parallel"}:
        return "data_parallel"
    if raw in {"none", "single"}:
        return "none"
    if raw == "ddp":
        return "ddp"
    raise ValueError(
        f"Unsupported train_strategy={train_strategy!r}; "
        "expected one of: none, data_parallel, ddp."
    )


def _maybe_init_process_group(train_strategy: str) -> bool:
    if train_strategy != "ddp":
        return False
    if _dist_ready():
        return False
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "train_strategy='ddp' requires torchrun environment variables (RANK/WORLD_SIZE)."
        )
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True


def _cleanup_process_group(initialized_here: bool) -> None:
    if initialized_here and _dist_ready():
        dist.destroy_process_group()


def _parse_device_list(primary_device: str, devices: Optional[str]) -> List[str]:
    raw = str(devices).strip() if devices is not None else ""
    tokens = [x.strip() for x in raw.split(",") if x.strip()] if raw else [str(primary_device).strip()]
    if not tokens:
        tokens = [str(primary_device).strip()]

    normalized: List[str] = []
    seen = set()
    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    for item in tokens:
        dev = torch.device(item)
        if dev.type == "cuda":
            if cuda_count <= 0:
                raise RuntimeError("CUDA devices were requested but torch.cuda.is_available() is False.")
            idx = 0 if dev.index is None else int(dev.index)
            if idx < 0 or idx >= cuda_count:
                print(
                    f"[v1.train] skip unavailable device {item!r} "
                    f"(visible cuda count={cuda_count})."
                )
                continue
            canon = f"cuda:{idx}"
        else:
            canon = str(dev)
        if canon in seen:
            continue
        seen.add(canon)
        normalized.append(canon)

    if not normalized:
        raise RuntimeError("No valid training devices resolved from --device/--devices.")
    return normalized


def _split_games(total_games: int, parts: int) -> List[int]:
    parts = max(1, int(parts))
    total_games = max(0, int(total_games))
    base = total_games // parts
    rem = total_games % parts
    out = [base + (1 if i < rem else 0) for i in range(parts)]
    return out


def _concat_self_play_batches_cpu(batches: List[TensorSelfPlayBatch]) -> TensorSelfPlayBatch:
    if not batches:
        empty_state = torch.empty(
            (0, NUM_INPUT_CHANNELS, GameState.BOARD_SIZE, GameState.BOARD_SIZE),
            dtype=torch.float32,
            device="cpu",
        )
        empty_action = torch.empty((0, TOTAL_ACTION_DIM), dtype=torch.float32, device="cpu")
        empty_mask = torch.empty((0, TOTAL_ACTION_DIM), dtype=torch.bool, device="cpu")
        empty_value = torch.empty((0,), dtype=torch.float32, device="cpu")
        return TensorSelfPlayBatch(
            state_tensors=empty_state,
            legal_masks=empty_mask,
            policy_targets=empty_action,
            value_targets=empty_value,
            soft_value_targets=empty_value.clone(),
        )
    if len(batches) == 1:
        return batches[0]
    return TensorSelfPlayBatch(
        state_tensors=torch.cat([b.state_tensors for b in batches], dim=0),
        legal_masks=torch.cat([b.legal_masks for b in batches], dim=0),
        policy_targets=torch.cat([b.policy_targets for b in batches], dim=0),
        value_targets=torch.cat([b.value_targets for b in batches], dim=0),
        soft_value_targets=torch.cat([b.soft_value_targets for b in batches], dim=0),
    )


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


def _merge_piece_delta_buckets(stats: List[SelfPlayV1Stats]) -> Dict[str, int]:
    merged: Dict[str, int] = {str(delta): 0 for delta in range(-18, 19)}
    for row in stats:
        buckets = _normalize_piece_delta_buckets(row.piece_delta_buckets)
        for key, value in buckets.items():
            merged[key] = int(merged.get(key, 0) + int(value))
    return merged


def _merge_self_play_stats(stats: List[SelfPlayV1Stats], elapsed_sec: float) -> SelfPlayV1Stats:
    total_games = int(sum(int(s.num_games) for s in stats))
    total_positions = int(sum(int(s.num_positions) for s in stats))
    black_wins = int(sum(int(s.black_wins) for s in stats))
    white_wins = int(sum(int(s.white_wins) for s in stats))
    draws = int(sum(int(s.draws) for s in stats))
    weighted_len_num = float(sum(float(s.avg_game_length) * float(s.num_games) for s in stats))
    avg_game_length = weighted_len_num / max(1.0, float(total_games))
    piece_delta_buckets = _merge_piece_delta_buckets(stats)

    step_ms: Dict[str, float] = {}
    step_ratio: Dict[str, float] = {}
    step_calls: Dict[str, int] = {}
    counter_sum: Dict[str, int] = {}
    for s in stats:
        for k, v in s.step_timing_ms.items():
            step_ms[k] = float(step_ms.get(k, 0.0) + float(v))
        for k, v in s.step_timing_calls.items():
            step_calls[k] = int(step_calls.get(k, 0) + int(v))
        for k, v in s.mcts_counters.items():
            counter_sum[k] = int(counter_sum.get(k, 0) + int(v))

    tracked_keys = ("root_puct_ms", "pack_writeback_ms", "self_play_step_ms", "finalize_ms")
    total_tracked = float(sum(float(step_ms.get(k, 0.0)) for k in tracked_keys))
    for k in tracked_keys:
        value = float(step_ms.get(k, 0.0))
        step_ratio[k] = value / total_tracked if total_tracked > 0.0 else 0.0
        step_calls.setdefault(k, 0)
        step_ms.setdefault(k, 0.0)

    elapsed = max(1e-9, float(elapsed_sec))
    return SelfPlayV1Stats(
        num_games=total_games,
        num_positions=total_positions,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        avg_game_length=float(avg_game_length),
        elapsed_sec=elapsed,
        positions_per_sec=float(total_positions / elapsed),
        games_per_sec=float(total_games / elapsed),
        step_timing_ms=step_ms,
        step_timing_ratio=step_ratio,
        step_timing_calls=step_calls,
        mcts_counters=counter_sum,
        piece_delta_buckets=piece_delta_buckets,
    )


def _compute_value_target_summary(samples: TensorSelfPlayBatch) -> Dict[str, Any]:
    values = samples.value_targets.detach().to("cpu", dtype=torch.float32).view(-1)
    total = int(values.numel())
    if total <= 0:
        return {
            "total": 0,
            "nonzero_count": 0,
            "zero_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "nonzero_ratio": 0.0,
        }
    positive = int(torch.count_nonzero(values > 0).item())
    negative = int(torch.count_nonzero(values < 0).item())
    nonzero = int(positive + negative)
    zero = int(total - nonzero)
    return {
        "total": total,
        "nonzero_count": nonzero,
        "zero_count": zero,
        "positive_count": positive,
        "negative_count": negative,
        "nonzero_ratio": float(nonzero / max(1, total)),
    }


def _build_self_play_report(
    *,
    stats: SelfPlayV1Stats,
    value_target_summary: Dict[str, Any],
) -> Dict[str, Any]:
    payload = stats.to_dict()
    games = max(1, int(stats.num_games))
    decisive = int(stats.black_wins + stats.white_wins)
    piece_delta_buckets = _normalize_piece_delta_buckets(stats.piece_delta_buckets)
    piece_delta_bucket_total = int(sum(int(v) for v in piece_delta_buckets.values()))
    payload["decisive_games"] = int(decisive)
    payload["decisive_game_ratio"] = float(decisive / games)
    payload["draw_game_ratio"] = float(int(stats.draws) / games)
    payload["piece_delta_buckets"] = piece_delta_buckets
    payload["piece_delta_bucket_total"] = int(piece_delta_bucket_total)
    payload["piece_delta_bucket_expected"] = int(stats.num_games)
    payload["piece_delta_bucket_coverage"] = float(piece_delta_bucket_total / games)
    payload["value_target_summary"] = dict(value_target_summary)
    return payload


def _print_self_play_summary(
    *,
    stats: SelfPlayV1Stats,
    value_target_summary: Dict[str, Any],
) -> None:
    games = max(1, int(stats.num_games))
    decisive = int(stats.black_wins + stats.white_wins)
    decisive_ratio = float(decisive / games)
    draw_ratio = float(int(stats.draws) / games)
    nonzero = int(value_target_summary.get("nonzero_count", 0))
    total = int(value_target_summary.get("total", 0))
    nonzero_ratio = float(value_target_summary.get("nonzero_ratio", 0.0))
    piece_delta_buckets = _normalize_piece_delta_buckets(stats.piece_delta_buckets)
    piece_delta_bucket_total = int(sum(int(v) for v in piece_delta_buckets.values()))
    nonzero_bucket_tokens: List[str] = []
    for delta in range(-18, 19):
        count = int(piece_delta_buckets.get(str(delta), 0))
        if count > 0:
            nonzero_bucket_tokens.append(f"{delta}:{count}")
    bucket_view = ",".join(nonzero_bucket_tokens) if nonzero_bucket_tokens else "none"
    _print_rank0(
        "[v1.train] selfplay outcomes "
        f"black_win={int(stats.black_wins)} white_win={int(stats.white_wins)} draw={int(stats.draws)} "
        f"decisive={decisive}/{games} ({decisive_ratio*100.0:.2f}%) draw_rate={draw_ratio*100.0:.2f}%"
    )
    _print_rank0(
        "[v1.train] selfplay piece_delta buckets "
        f"total={piece_delta_bucket_total}/{int(stats.num_games)} nonzero={{{bucket_view}}}"
    )
    _print_rank0(
        "[v1.train] selfplay value targets "
        f"nonzero={nonzero}/{total} ({nonzero_ratio*100.0:.2f}%) "
        f"pos={int(value_target_summary.get('positive_count', 0))} "
        f"neg={int(value_target_summary.get('negative_count', 0))}"
    )


def _run_self_play_shard(
    *,
    model_state_dict_cpu: Dict[str, torch.Tensor],
    shard_games: int,
    shard_device: str,
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
    seed: int,
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    torch.manual_seed(int(seed))
    shard_dev_obj = torch.device(shard_device)
    if shard_dev_obj.type == "cuda":
        torch.cuda.set_device(shard_dev_obj)
        torch.cuda.manual_seed(int(seed))

    local_model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    local_model.load_state_dict(model_state_dict_cpu, strict=True)
    local_model.to(shard_dev_obj)
    local_model.eval()

    shard_concurrent = max(1, min(int(shard_games), int(concurrent_games_per_device)))
    samples, stats = self_play_v1_gpu(
        model=local_model,
        num_games=int(shard_games),
        mcts_simulations=int(mcts_simulations),
        temperature_init=float(temperature_init),
        temperature_final=float(temperature_final),
        temperature_threshold=int(temperature_threshold),
        exploration_weight=float(exploration_weight),
        device=str(shard_device),
        add_dirichlet_noise=True,
        dirichlet_alpha=float(dirichlet_alpha),
        dirichlet_epsilon=float(dirichlet_epsilon),
        soft_value_k=float(soft_value_k),
        opening_random_moves=int(opening_random_moves),
        max_game_plies=int(max_game_plies),
        sample_moves=True,
        concurrent_games=shard_concurrent,
        verbose=False,
    )
    return samples.to("cpu"), stats


def _self_play_stats_from_payload(stats_payload: Dict[str, Any]) -> SelfPlayV1Stats:
    payload = stats_payload if isinstance(stats_payload, dict) else {}
    step_timing_ms_raw = payload.get("step_timing_ms")
    step_timing_ratio_raw = payload.get("step_timing_ratio")
    step_timing_calls_raw = payload.get("step_timing_calls")
    mcts_counters_raw = payload.get("mcts_counters")
    step_timing_ms = (
        {str(k): float(v) for k, v in step_timing_ms_raw.items()}
        if isinstance(step_timing_ms_raw, dict)
        else {}
    )
    step_timing_ratio = (
        {str(k): float(v) for k, v in step_timing_ratio_raw.items()}
        if isinstance(step_timing_ratio_raw, dict)
        else {}
    )
    step_timing_calls = (
        {str(k): int(v) for k, v in step_timing_calls_raw.items()}
        if isinstance(step_timing_calls_raw, dict)
        else {}
    )
    mcts_counters = (
        {str(k): int(v) for k, v in mcts_counters_raw.items()}
        if isinstance(mcts_counters_raw, dict)
        else {}
    )
    piece_delta_buckets = _normalize_piece_delta_buckets(payload.get("piece_delta_buckets"))
    elapsed = max(1e-9, float(payload.get("elapsed_sec", 0.0) or 0.0))
    num_games = int(payload.get("num_games", 0) or 0)
    num_positions = int(payload.get("num_positions", 0) or 0)
    return SelfPlayV1Stats(
        num_games=num_games,
        num_positions=num_positions,
        black_wins=int(payload.get("black_wins", 0) or 0),
        white_wins=int(payload.get("white_wins", 0) or 0),
        draws=int(payload.get("draws", 0) or 0),
        avg_game_length=float(payload.get("avg_game_length", 0.0) or 0.0),
        elapsed_sec=elapsed,
        positions_per_sec=float(payload.get("positions_per_sec", float(num_positions / elapsed))),
        games_per_sec=float(payload.get("games_per_sec", float(num_games / elapsed))),
        step_timing_ms=step_timing_ms,
        step_timing_ratio=step_timing_ratio,
        step_timing_calls=step_timing_calls,
        mcts_counters=mcts_counters,
        piece_delta_buckets=piece_delta_buckets,
    )


def _normalize_self_play_backend(backend: str) -> str:
    raw = str(backend).strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"thread", "threads"}:
        return "thread"
    if raw in {"process", "proc", "multiprocess", "mp"}:
        return "process"
    raise ValueError(
        f"Unsupported self-play backend={backend!r}; expected one of: auto, thread, process."
    )


def _resolve_self_play_backend(*, requested_backend: Optional[str], devices: List[str]) -> str:
    source = requested_backend
    if source is None:
        source = os.environ.get("V1_SELF_PLAY_BACKEND", "auto")
    mode = _normalize_self_play_backend(str(source))
    if mode != "auto":
        return mode
    if len(devices) <= 1:
        return "thread"
    all_cuda = all(torch.device(str(dev)).type == "cuda" for dev in devices)
    if all_cuda and sys.platform.startswith("linux"):
        return "process"
    return "thread"


def _prepare_self_play_workspace(
    *,
    shard_dir: Optional[str],
    iteration_seed: int,
) -> Tuple[str, bool]:
    tag = f"iter_{int(iteration_seed):06d}_pid_{os.getpid()}_{int(time.time() * 1000)}"
    if shard_dir:
        base = os.path.abspath(str(shard_dir))
        os.makedirs(base, exist_ok=True)
        root = os.path.join(base, tag)
        os.makedirs(root, exist_ok=True)
        return root, False
    root = tempfile.mkdtemp(prefix=f"v1_selfplay_{tag}_")
    return root, True


def _run_self_play_multi_device_thread(
    *,
    model: ChessNet,
    num_games: int,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    devices: List[str],
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    soft_value_k: float,
    opening_random_moves: int,
    max_game_plies: int,
    concurrent_games_per_device: int,
    iteration_seed: int,
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    shards = _split_games(int(num_games), len(devices))
    active: List[Tuple[int, str, int]] = [
        (idx, dev, g) for idx, (dev, g) in enumerate(zip(devices, shards)) if int(g) > 0
    ]
    if not active:
        raise RuntimeError("No self-play shard assigned after game split.")

    started = time.perf_counter()
    batches: List[TensorSelfPlayBatch] = []
    stats_list: List[SelfPlayV1Stats] = []
    finalize_graph_key = "V1_FINALIZE_GRAPH"
    had_finalize_graph_env = finalize_graph_key in os.environ
    prev_finalize_graph = os.environ.get(finalize_graph_key)
    if not had_finalize_graph_env:
        os.environ[finalize_graph_key] = "off"
        _print_rank0(
            "[v1.train] thread backend: default V1_FINALIZE_GRAPH=off for stability. "
            "Set V1_FINALIZE_GRAPH=on to force capture."
        )
    try:
        with ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="v1-sp") as pool:
            future_map = {}
            for worker_idx, shard_dev, shard_games in active:
                worker_seed = int(iteration_seed * 10007 + (worker_idx + 1) * 9973)
                future = pool.submit(
                    _run_self_play_shard,
                    model_state_dict_cpu=model_state_cpu,
                    shard_games=int(shard_games),
                    shard_device=str(shard_dev),
                    mcts_simulations=int(mcts_simulations),
                    temperature_init=float(temperature_init),
                    temperature_final=float(temperature_final),
                    temperature_threshold=int(temperature_threshold),
                    exploration_weight=float(exploration_weight),
                    dirichlet_alpha=float(dirichlet_alpha),
                    dirichlet_epsilon=float(dirichlet_epsilon),
                    soft_value_k=float(soft_value_k),
                    opening_random_moves=int(opening_random_moves),
                    max_game_plies=int(max_game_plies),
                    concurrent_games_per_device=int(concurrent_games_per_device),
                    seed=int(worker_seed),
                )
                future_map[future] = (worker_idx, shard_dev, shard_games)

            for future in as_completed(future_map):
                worker_idx, shard_dev, shard_games = future_map[future]
                try:
                    samples_cpu, stats = future.result()
                    batches.append(samples_cpu)
                    stats_list.append(stats)
                except Exception as exc:
                    raise RuntimeError(
                        f"v1 thread self-play failed on worker={worker_idx}, "
                        f"device={shard_dev}, games={shard_games}"
                    ) from exc
    finally:
        if not had_finalize_graph_env:
            if prev_finalize_graph is None:
                os.environ.pop(finalize_graph_key, None)
            else:
                os.environ[finalize_graph_key] = prev_finalize_graph

    merged_batch = _concat_self_play_batches_cpu(batches)
    merged_stats = _merge_self_play_stats(
        stats=stats_list,
        elapsed_sec=max(1e-9, time.perf_counter() - started),
    )
    return merged_batch, merged_stats


def _run_self_play_multi_device_process(
    *,
    model: ChessNet,
    num_games: int,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    devices: List[str],
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    soft_value_k: float,
    opening_random_moves: int,
    max_game_plies: int,
    concurrent_games_per_device: int,
    iteration_seed: int,
    shard_dir: Optional[str],
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    shards = _split_games(int(num_games), len(devices))
    active: List[Tuple[int, str, int]] = [
        (idx, dev, g) for idx, (dev, g) in enumerate(zip(devices, shards)) if int(g) > 0
    ]
    if not active:
        raise RuntimeError("No self-play shard assigned after game split.")

    workspace, auto_cleanup = _prepare_self_play_workspace(
        shard_dir=shard_dir,
        iteration_seed=int(iteration_seed),
    )
    state_path = os.path.join(workspace, "model_state_cpu.pt")
    torch.save(model_state_cpu, state_path)

    started = time.perf_counter()
    failed = False
    worker_rows: List[Dict[str, Any]] = []
    try:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(active),
            mp_context=mp_ctx,
        ) as pool:
            future_map = {}
            for worker_idx, shard_dev, shard_games in active:
                worker_seed = int(iteration_seed * 10007 + (worker_idx + 1) * 9973)
                shard_output = os.path.join(
                    workspace,
                    f"selfplay_shard_{int(iteration_seed):06d}_{int(worker_idx):02d}.pt",
                )
                future = pool.submit(
                    run_self_play_worker,
                    worker_idx=int(worker_idx),
                    shard_device=str(shard_dev),
                    shard_games=int(shard_games),
                    seed=int(worker_seed),
                    model_state_path=str(state_path),
                    output_path=str(shard_output),
                    mcts_simulations=int(mcts_simulations),
                    temperature_init=float(temperature_init),
                    temperature_final=float(temperature_final),
                    temperature_threshold=int(temperature_threshold),
                    exploration_weight=float(exploration_weight),
                    dirichlet_alpha=float(dirichlet_alpha),
                    dirichlet_epsilon=float(dirichlet_epsilon),
                    soft_value_k=float(soft_value_k),
                    opening_random_moves=int(opening_random_moves),
                    max_game_plies=int(max_game_plies),
                    concurrent_games_per_device=int(concurrent_games_per_device),
                )
                future_map[future] = (worker_idx, shard_dev, shard_games, shard_output)

            for future in as_completed(future_map):
                worker_idx, shard_dev, shard_games, shard_output = future_map[future]
                try:
                    row = future.result()
                    worker_rows.append(row)
                except Exception as exc:
                    raise RuntimeError(
                        "v1 process self-play failed "
                        f"on worker={worker_idx}, device={shard_dev}, games={shard_games}, "
                        f"shard_output={shard_output}"
                    ) from exc

        batches: List[TensorSelfPlayBatch] = []
        stats_list: List[SelfPlayV1Stats] = []
        for row in sorted(worker_rows, key=lambda x: int(x.get("worker_idx", 0))):
            shard_path = str(row.get("output_path", ""))
            if not shard_path:
                raise RuntimeError(f"Missing shard output path in worker row: {row}")
            batch_cpu, stats_payload, meta_payload = _load_self_play_payload(shard_path)
            if isinstance(meta_payload, dict) and meta_payload:
                _print_rank0(
                    "[v1.train] self-play shard merged "
                    f"worker={int(meta_payload.get('worker_idx', row.get('worker_idx', 0)))} "
                    f"device={meta_payload.get('device', '')} "
                    f"games={int(meta_payload.get('games', 0))} "
                    f"chunks={int(meta_payload.get('num_chunks', 1))} "
                    f"games_per_chunk={int(meta_payload.get('games_per_chunk', meta_payload.get('games', 0)))}"
                )
            batches.append(batch_cpu)
            stats_list.append(_self_play_stats_from_payload(stats_payload))

        merged_batch = _concat_self_play_batches_cpu(batches)
        merged_stats = _merge_self_play_stats(
            stats=stats_list,
            elapsed_sec=max(1e-9, time.perf_counter() - started),
        )
        return merged_batch, merged_stats
    except Exception:
        failed = True
        raise
    finally:
        if auto_cleanup and not failed:
            shutil.rmtree(workspace, ignore_errors=True)
        elif auto_cleanup and failed:
            _print_rank0(
                "[v1.train] process self-play failed; keep shard workspace for debug: "
                f"{workspace}"
            )


def _run_self_play_multi_device(
    *,
    model: ChessNet,
    num_games: int,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    devices: List[str],
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    soft_value_k: float,
    opening_random_moves: int,
    max_game_plies: int,
    concurrent_games_per_device: int,
    iteration_seed: int,
    self_play_backend: Optional[str],
    self_play_shard_dir: Optional[str],
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    if len(devices) <= 1:
        single_device = devices[0]
        shard_concurrent = max(1, min(int(num_games), int(concurrent_games_per_device)))
        return self_play_v1_gpu(
            model=model,
            num_games=int(num_games),
            mcts_simulations=int(mcts_simulations),
            temperature_init=float(temperature_init),
            temperature_final=float(temperature_final),
            temperature_threshold=int(temperature_threshold),
            exploration_weight=float(exploration_weight),
            device=str(single_device),
            add_dirichlet_noise=True,
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_epsilon=float(dirichlet_epsilon),
            soft_value_k=float(soft_value_k),
            opening_random_moves=int(opening_random_moves),
            max_game_plies=int(max_game_plies),
            sample_moves=True,
            concurrent_games=shard_concurrent,
            verbose=False,
        )

    backend = _resolve_self_play_backend(
        requested_backend=self_play_backend,
        devices=devices,
    )
    _print_rank0(f"[v1.train] self-play backend={backend} devices={devices}")
    if backend == "process":
        return _run_self_play_multi_device_process(
            model=model,
            num_games=int(num_games),
            mcts_simulations=int(mcts_simulations),
            temperature_init=float(temperature_init),
            temperature_final=float(temperature_final),
            temperature_threshold=int(temperature_threshold),
            exploration_weight=float(exploration_weight),
            devices=list(devices),
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_epsilon=float(dirichlet_epsilon),
            soft_value_k=float(soft_value_k),
            opening_random_moves=int(opening_random_moves),
            max_game_plies=int(max_game_plies),
            concurrent_games_per_device=int(concurrent_games_per_device),
            iteration_seed=int(iteration_seed),
            shard_dir=self_play_shard_dir,
        )
    return _run_self_play_multi_device_thread(
        model=model,
        num_games=int(num_games),
        mcts_simulations=int(mcts_simulations),
        temperature_init=float(temperature_init),
        temperature_final=float(temperature_final),
        temperature_threshold=int(temperature_threshold),
        exploration_weight=float(exploration_weight),
        devices=list(devices),
        dirichlet_alpha=float(dirichlet_alpha),
        dirichlet_epsilon=float(dirichlet_epsilon),
        soft_value_k=float(soft_value_k),
        opening_random_moves=int(opening_random_moves),
        max_game_plies=int(max_game_plies),
        concurrent_games_per_device=int(concurrent_games_per_device),
        iteration_seed=int(iteration_seed),
    )


def _build_model() -> ChessNet:
    return ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)


def _load_checkpoint_into_model(
    *,
    model: ChessNet,
    load_checkpoint: Optional[str],
    map_location: torch.device,
) -> None:
    if not load_checkpoint:
        return
    if not os.path.exists(load_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {load_checkpoint}")
    checkpoint = torch.load(load_checkpoint, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    _print_rank0(f"[v1.train] loaded checkpoint: {load_checkpoint}")


def _save_json(path: str, payload: Any) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_self_play_payload(
    *,
    path: str,
    samples: TensorSelfPlayBatch,
    stats: SelfPlayV1Stats,
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "state_tensors": samples.state_tensors.detach().cpu(),
        "legal_masks": samples.legal_masks.detach().cpu(),
        "policy_targets": samples.policy_targets.detach().cpu(),
        "value_targets": samples.value_targets.detach().cpu(),
        "soft_value_targets": samples.soft_value_targets.detach().cpu(),
        "stats": stats.to_dict(),
        "metadata": dict(metadata),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def _load_self_play_payload(path: str) -> Tuple[TensorSelfPlayBatch, Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Self-play payload not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, TensorSelfPlayBatch):
        return payload.to("cpu"), {}, {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported self-play payload format: {type(payload)!r}")

    required = [
        "state_tensors",
        "legal_masks",
        "policy_targets",
        "value_targets",
        "soft_value_targets",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"Missing keys in self-play payload {path}: {missing}")

    batch = TensorSelfPlayBatch(
        state_tensors=payload["state_tensors"].to("cpu"),
        legal_masks=payload["legal_masks"].to("cpu"),
        policy_targets=payload["policy_targets"].to("cpu"),
        value_targets=payload["value_targets"].to("cpu"),
        soft_value_targets=payload["soft_value_targets"].to("cpu"),
    )
    stats_payload = payload.get("stats")
    if not isinstance(stats_payload, dict):
        stats_payload = {}
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        metadata_payload = {}
    return batch, stats_payload, metadata_payload


def _resolve_primary_train_device(train_device_list: List[str], train_strategy: str) -> torch.device:
    if train_strategy == "ddp":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = int(torch.cuda.device_count())
        if local_rank < 0 or local_rank >= visible:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for visible cuda count={visible}."
            )
        return torch.device(f"cuda:{local_rank}")
    return torch.device(train_device_list[0])


def _run_inference_shard(
    *,
    model_state_dict_cpu: Dict[str, torch.Tensor],
    device: str,
    batch_size: int,
    warmup_iters: int,
    iters: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    dev = torch.device(device)
    model = _build_model()
    model.load_state_dict(model_state_dict_cpu, strict=True)
    model.to(dev)
    model.eval()

    bs = max(1, int(batch_size))
    loops = max(1, int(iters))
    warmups = max(0, int(warmup_iters))
    input_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    x = torch.randn(
        (bs, NUM_INPUT_CHANNELS, GameState.BOARD_SIZE, GameState.BOARD_SIZE),
        device=dev,
        dtype=input_dtype,
    )

    with torch.inference_mode():
        for _ in range(warmups):
            if dev.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        started = time.perf_counter()
        for _ in range(loops):
            if dev.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = max(1e-9, time.perf_counter() - started)

    samples = int(bs * loops)
    return {
        "device": str(dev),
        "batch_size": int(bs),
        "iters": int(loops),
        "warmup_iters": int(warmups),
        "elapsed_sec": float(elapsed),
        "samples_per_sec": float(samples / elapsed),
    }


def _run_inference_multi_device(
    *,
    model: ChessNet,
    devices: List[str],
    batch_size: int,
    warmup_iters: int,
    iters: int,
    iteration_seed: int,
) -> Dict[str, Any]:
    model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    active = [str(d) for d in devices]
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    if len(active) <= 1:
        rows.append(
            _run_inference_shard(
                model_state_dict_cpu=model_state_cpu,
                device=active[0],
                batch_size=batch_size,
                warmup_iters=warmup_iters,
                iters=iters,
                seed=iteration_seed,
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="v1-infer") as pool:
            future_map = {}
            for idx, dev in enumerate(active):
                future = pool.submit(
                    _run_inference_shard,
                    model_state_dict_cpu=model_state_cpu,
                    device=dev,
                    batch_size=int(batch_size),
                    warmup_iters=int(warmup_iters),
                    iters=int(iters),
                    seed=int(iteration_seed * 10007 + (idx + 1) * 7919),
                )
                future_map[future] = dev

            for future in as_completed(future_map):
                dev = future_map[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"v1 inference shard failed on device={dev}") from exc

    elapsed = max(1e-9, time.perf_counter() - started)
    rows_sorted = sorted(rows, key=lambda x: str(x.get("device", "")))
    return {
        "devices": active,
        "rows": rows_sorted,
        "batch_size": int(max(1, int(batch_size))),
        "iters": int(max(1, int(iters))),
        "warmup_iters": int(max(0, int(warmup_iters))),
        "elapsed_sec": float(elapsed),
        "aggregate_samples_per_sec": float(sum(float(r["samples_per_sec"]) for r in rows_sorted)),
    }


def train_pipeline_v1(
    iterations: int = 10,
    self_play_games: int = 8,
    mcts_simulations: int = 64,
    batch_size: int = 512,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    temperature_threshold: int = 10,
    exploration_weight: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    self_play_concurrent_games: int = 8,
    self_play_opening_random_moves: int = 0,
    self_play_backend: Optional[str] = None,
    self_play_shard_dir: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints_v1",
    device: str = "cpu",
    devices: Optional[str] = None,
    train_devices: Optional[str] = None,
    train_strategy: str = "data_parallel",
    soft_value_k: float = 2.0,
    max_game_plies: int = 512,
    load_checkpoint: Optional[str] = None,
    stage: str = "all",
    self_play_output: Optional[str] = None,
    self_play_input: Optional[str] = None,
    self_play_stats_json: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    metrics_output: Optional[str] = None,
    infer_devices: Optional[str] = None,
    infer_batch_size: int = 4096,
    infer_warmup_iters: int = 20,
    infer_iters: int = 100,
    infer_output: Optional[str] = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    stage_norm = str(stage).strip().lower()
    if stage_norm not in {"all", "selfplay", "train", "infer"}:
        raise ValueError(f"Unsupported stage={stage!r}; expected all/selfplay/train/infer.")

    train_strategy_norm = _normalize_train_strategy(train_strategy)
    if stage_norm == "all" and train_strategy_norm == "ddp":
        raise RuntimeError(
            "stage='all' with train_strategy='ddp' is not supported. "
            "Use staged mode: selfplay -> torchrun train -> infer."
        )
    if stage_norm != "train" and train_strategy_norm == "ddp":
        _print_rank0("[v1.train] train_strategy=ddp ignored for non-train stage; fallback to data_parallel.")
        train_strategy_norm = "data_parallel"

    init_pg_here = _maybe_init_process_group(train_strategy_norm)
    try:
        backend_arg = (
            _normalize_self_play_backend(str(self_play_backend))
            if self_play_backend is not None
            else None
        )
        shard_dir_arg = str(self_play_shard_dir).strip() if self_play_shard_dir is not None else ""
        shard_dir_arg = shard_dir_arg or None

        self_play_device_list = _parse_device_list(primary_device=device, devices=devices)
        if train_devices is None:
            train_device_list = _parse_device_list(primary_device=device, devices=str(device))
        else:
            train_device_list = _parse_device_list(primary_device=device, devices=train_devices)
        infer_device_list = _parse_device_list(
            primary_device=self_play_device_list[0],
            devices=(infer_devices if infer_devices is not None else devices),
        )

        primary_train_device = _resolve_primary_train_device(
            train_device_list=train_device_list,
            train_strategy=train_strategy_norm,
        )
        if len(self_play_device_list) > 1 and primary_train_device.type != "cuda":
            _print_rank0("[v1.train] multi-device requested on non-cuda primary; fallback to single-device mode.")
            self_play_device_list = [str(primary_train_device)]

        _print_rank0(f"[v1.train] stage={stage_norm} train_strategy={train_strategy_norm}")
        _print_rank0(f"[v1.train] self_play_devices={self_play_device_list}")
        _print_rank0(
            "[v1.train] self_play_backend="
            f"{backend_arg if backend_arg is not None else 'auto'} "
            f"self_play_shard_dir={shard_dir_arg or '<temp>'}"
        )
        _print_rank0(
            f"[v1.train] self_play_opening_random_moves={int(self_play_opening_random_moves)}"
        )
        _print_rank0(f"[v1.train] train_devices={train_device_list}")
        _print_rank0(f"[v1.train] infer_devices={infer_device_list}")
        if int(self_play_concurrent_games) > 256:
            _print_rank0(
                "[v1.train] warning: self_play_concurrent_games is very high; "
                "start with 32~128 for stability and throughput."
            )

        model = _build_model()
        _load_checkpoint_into_model(
            model=model,
            load_checkpoint=load_checkpoint,
            map_location=primary_train_device,
        )
        model.to(primary_train_device)

        if stage_norm == "selfplay":
            model.eval()
            samples, sp_stats = _run_self_play_multi_device(
                model=model,
                num_games=int(self_play_games),
                mcts_simulations=int(mcts_simulations),
                temperature_init=float(temperature_init),
                temperature_final=float(temperature_final),
                temperature_threshold=int(temperature_threshold),
                exploration_weight=float(exploration_weight),
                devices=self_play_device_list,
                dirichlet_alpha=float(dirichlet_alpha),
                dirichlet_epsilon=float(dirichlet_epsilon),
                soft_value_k=float(soft_value_k),
                opening_random_moves=int(self_play_opening_random_moves),
                max_game_plies=int(max_game_plies),
                concurrent_games_per_device=int(self_play_concurrent_games),
                iteration_seed=1,
                self_play_backend=backend_arg,
                self_play_shard_dir=shard_dir_arg,
            )
            value_target_summary = _compute_value_target_summary(samples)
            sp_report = _build_self_play_report(
                stats=sp_stats,
                value_target_summary=value_target_summary,
            )
            output_path = str(self_play_output or os.path.join(checkpoint_dir, "selfplay_batch_v1.pt"))
            _save_self_play_payload(
                path=output_path,
                samples=samples,
                stats=sp_stats,
                metadata={
                    "stage": "selfplay",
                    "self_play_devices": list(self_play_device_list),
                    "self_play_backend": backend_arg if backend_arg is not None else "auto",
                    "self_play_shard_dir": shard_dir_arg,
                    "mcts_simulations": int(mcts_simulations),
                    "self_play_games": int(self_play_games),
                    "self_play_concurrent_games": int(self_play_concurrent_games),
                    "self_play_opening_random_moves": int(self_play_opening_random_moves),
                    "value_target_summary": dict(value_target_summary),
                },
                )
            if self_play_stats_json:
                _save_json(str(self_play_stats_json), sp_report)
            _print_self_play_summary(
                stats=sp_stats,
                value_target_summary=value_target_summary,
            )
            _print_rank0(
                f"[v1.train] selfplay saved: {output_path} "
                f"(games={sp_stats.num_games}, positions={sp_stats.num_positions})"
            )
            return

        if stage_norm == "train":
            if not self_play_input:
                raise ValueError("stage='train' requires --self_play_input.")
            samples, sp_stats_payload, _sp_meta = _load_self_play_payload(str(self_play_input))
            train_start = time.perf_counter()
            model.train()
            parallel_devices = (
                train_device_list if (train_strategy_norm == "data_parallel" and len(train_device_list) > 1) else None
            )
            model, train_metrics = train_network_from_tensors(
                model=model,
                samples=samples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                soft_label_alpha=0.0,
                policy_draw_weight=1.0,
                device=str(primary_train_device),
                use_amp=(primary_train_device.type == "cuda"),
                parallel_devices=parallel_devices,
                parallel_strategy=train_strategy_norm,
            )
            train_elapsed = time.perf_counter() - train_start
            if _dist_ready():
                dist.barrier()
            if _is_rank0():
                ckpt_name = str(checkpoint_name or "model_iter_001.pt")
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                torch.save(
                    {
                        "iteration": 1,
                        "model_state_dict": model.state_dict(),
                        "board_size": GameState.BOARD_SIZE,
                        "num_input_channels": NUM_INPUT_CHANNELS,
                        "self_play_devices": list(self_play_device_list),
                        "train_devices": list(train_device_list),
                        "train_strategy": train_strategy_norm,
                        "stage": "train",
                        "self_play_input": str(self_play_input),
                    },
                    ckpt_path,
                )
                last_epoch = (train_metrics.get("epoch_stats") or [{}])[-1]
                entry: Dict[str, Any] = {
                    "stage": "train",
                    "self_play_input": str(self_play_input),
                    "self_play_games": sp_stats_payload.get("num_games"),
                    "self_play_positions": samples.num_samples,
                    "self_play_decisive_games": sp_stats_payload.get("decisive_games"),
                    "self_play_decisive_game_ratio": sp_stats_payload.get("decisive_game_ratio"),
                    "self_play_draw_game_ratio": sp_stats_payload.get("draw_game_ratio"),
                    "self_play_value_target_summary": sp_stats_payload.get("value_target_summary"),
                    "train_devices": list(train_device_list),
                    "train_strategy": train_strategy_norm,
                    "train_time_sec": float(train_elapsed),
                    "train_avg_loss": last_epoch.get("avg_loss"),
                    "train_avg_policy_loss": last_epoch.get("avg_policy_loss"),
                    "train_avg_value_loss": last_epoch.get("avg_value_loss"),
                    "checkpoint": ckpt_path,
                }
                metrics_path = str(metrics_output or os.path.join(checkpoint_dir, "training_metrics_v1.json"))
                _save_json(metrics_path, [entry])
                _print_rank0(
                    "[v1.train] train complete "
                    f"loss={float(last_epoch.get('avg_loss') or 0.0):.4f} "
                    f"time={train_elapsed:.2f}s checkpoint={ckpt_path}"
                )
                _print_rank0(f"[v1.train] metrics saved: {metrics_path}")
            if _dist_ready():
                dist.barrier()
            return

        if stage_norm == "infer":
            model.eval()
            report = _run_inference_multi_device(
                model=model,
                devices=infer_device_list,
                batch_size=int(infer_batch_size),
                warmup_iters=int(infer_warmup_iters),
                iters=int(infer_iters),
                iteration_seed=1,
            )
            report["stage"] = "infer"
            report["checkpoint"] = str(load_checkpoint) if load_checkpoint else None
            output_path = str(infer_output or os.path.join(checkpoint_dir, "inference_metrics_v1.json"))
            _save_json(output_path, report)
            _print_rank0(
                "[v1.train] infer complete "
                f"aggregate_samples_per_sec={float(report['aggregate_samples_per_sec']):.2f} "
                f"output={output_path}"
            )
            return

        metrics: List[Dict[str, Any]] = []
        for iteration in range(iterations):
            it_idx = iteration + 1
            _print_rank0(f"\n[v1.train] ===== Iteration {it_idx}/{iterations} =====")

            sp_start = time.perf_counter()
            model.eval()
            samples, sp_stats = _run_self_play_multi_device(
                model=model,
                num_games=int(self_play_games),
                mcts_simulations=int(mcts_simulations),
                temperature_init=float(temperature_init),
                temperature_final=float(temperature_final),
                temperature_threshold=int(temperature_threshold),
                exploration_weight=float(exploration_weight),
                devices=self_play_device_list,
                dirichlet_alpha=float(dirichlet_alpha),
                dirichlet_epsilon=float(dirichlet_epsilon),
                soft_value_k=float(soft_value_k),
                opening_random_moves=int(self_play_opening_random_moves),
                max_game_plies=int(max_game_plies),
                concurrent_games_per_device=int(self_play_concurrent_games),
                iteration_seed=int(it_idx),
                self_play_backend=backend_arg,
                self_play_shard_dir=shard_dir_arg,
            )
            value_target_summary = _compute_value_target_summary(samples)
            sp_elapsed = time.perf_counter() - sp_start

            train_start = time.perf_counter()
            model.train()
            parallel_devices = (
                train_device_list if (train_strategy_norm == "data_parallel" and len(train_device_list) > 1) else None
            )
            model, train_metrics = train_network_from_tensors(
                model=model,
                samples=samples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                soft_label_alpha=0.0,
                policy_draw_weight=1.0,
                device=str(primary_train_device),
                use_amp=(primary_train_device.type == "cuda"),
                parallel_devices=parallel_devices,
                parallel_strategy=train_strategy_norm,
            )
            train_elapsed = time.perf_counter() - train_start

            ckpt_path = os.path.join(checkpoint_dir, f"model_iter_{it_idx:03d}.pt")
            torch.save(
                {
                    "iteration": it_idx,
                    "model_state_dict": model.state_dict(),
                    "board_size": GameState.BOARD_SIZE,
                    "num_input_channels": NUM_INPUT_CHANNELS,
                    "self_play_devices": self_play_device_list,
                    "train_devices": train_device_list,
                    "train_strategy": train_strategy_norm,
                },
                ckpt_path,
            )

            last_epoch = (train_metrics.get("epoch_stats") or [{}])[-1]
            entry: Dict[str, Any] = {
                "iteration": it_idx,
                "self_play_devices": list(self_play_device_list),
                "train_devices": list(train_device_list),
                "train_strategy": train_strategy_norm,
                "self_play_games": sp_stats.num_games,
                "self_play_positions": sp_stats.num_positions,
                "self_play_time_sec": sp_elapsed,
                "self_play_games_per_sec": sp_stats.games_per_sec,
                "self_play_positions_per_sec": sp_stats.positions_per_sec,
                "black_wins": sp_stats.black_wins,
                "white_wins": sp_stats.white_wins,
                "draws": sp_stats.draws,
                "decisive_games": int(sp_stats.black_wins + sp_stats.white_wins),
                "decisive_game_ratio": float(
                    (sp_stats.black_wins + sp_stats.white_wins) / max(1, int(sp_stats.num_games))
                ),
                "draw_game_ratio": float(int(sp_stats.draws) / max(1, int(sp_stats.num_games))),
                "value_target_summary": dict(value_target_summary),
                "train_time_sec": train_elapsed,
                "train_avg_loss": last_epoch.get("avg_loss"),
                "train_avg_policy_loss": last_epoch.get("avg_policy_loss"),
                "train_avg_value_loss": last_epoch.get("avg_value_loss"),
                "checkpoint": ckpt_path,
            }
            metrics.append(entry)

            if self_play_output:
                save_path = str(self_play_output).format(iteration=it_idx, iter=it_idx)
                _save_self_play_payload(
                    path=save_path,
                    samples=samples,
                    stats=sp_stats,
                    metadata={
                        "stage": "all",
                        "iteration": int(it_idx),
                        "self_play_devices": list(self_play_device_list),
                        "self_play_backend": backend_arg if backend_arg is not None else "auto",
                        "self_play_shard_dir": shard_dir_arg,
                        "self_play_concurrent_games": int(self_play_concurrent_games),
                        "self_play_opening_random_moves": int(self_play_opening_random_moves),
                        "value_target_summary": dict(value_target_summary),
                    },
                )

            _print_self_play_summary(
                stats=sp_stats,
                value_target_summary=value_target_summary,
            )
            _print_rank0(
                "[v1.train] "
                f"games={sp_stats.num_games} positions={sp_stats.num_positions} "
                f"self_play={sp_elapsed:.2f}s train={train_elapsed:.2f}s "
                f"loss={float(last_epoch.get('avg_loss') or 0.0):.4f}"
            )

        metrics_path = str(metrics_output or os.path.join(checkpoint_dir, "training_metrics_v1.json"))
        _save_json(metrics_path, metrics)
        _print_rank0(f"[v1.train] metrics saved: {metrics_path}")
    finally:
        _cleanup_process_group(init_pg_here)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train with v1 GPU-first self-play pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "selfplay", "train", "infer"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--self_play_games", type=int, default=4)
    parser.add_argument("--mcts_simulations", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature_init", type=float, default=1.0)
    parser.add_argument("--temperature_final", type=float, default=0.1)
    parser.add_argument("--temperature_threshold", type=int, default=10)
    parser.add_argument("--exploration_weight", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--self_play_concurrent_games", type=int, default=8)
    parser.add_argument("--self_play_opening_random_moves", type=int, default=0)
    parser.add_argument(
        "--self_play_backend",
        type=str,
        default=None,
        choices=["auto", "thread", "process"],
        help="Optional v1 self-play backend override; default is auto.",
    )
    parser.add_argument(
        "--self_play_shard_dir",
        type=str,
        default=None,
        help="Optional directory for process-backend shard payloads (default uses temp dir).",
    )
    parser.add_argument("--soft_value_k", type=float, default=2.0)
    parser.add_argument("--max_game_plies", type=int, default=512)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_v1")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional comma-separated device list for v1 self-play sharding, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--train_devices",
        type=str,
        default=None,
        help="Optional comma-separated device list for training parallelism; default is single --device.",
    )
    parser.add_argument(
        "--train_strategy",
        type=str,
        default="data_parallel",
        choices=["none", "data_parallel", "ddp"],
        help="Training parallel strategy.",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument(
        "--self_play_output",
        type=str,
        default=None,
        help="Path to save self-play tensor payload (for stage=selfplay, optional for stage=all).",
    )
    parser.add_argument(
        "--self_play_input",
        type=str,
        default=None,
        help="Path to load self-play tensor payload (required for stage=train).",
    )
    parser.add_argument(
        "--self_play_stats_json",
        type=str,
        default=None,
        help="Optional JSON path for self-play stats summary.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Checkpoint filename used by stage=train.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Optional metrics output path.",
    )
    parser.add_argument(
        "--infer_devices",
        type=str,
        default=None,
        help="Optional comma-separated device list for stage=infer.",
    )
    parser.add_argument("--infer_batch_size", type=int, default=4096)
    parser.add_argument("--infer_warmup_iters", type=int, default=20)
    parser.add_argument("--infer_iters", type=int, default=100)
    parser.add_argument(
        "--infer_output",
        type=str,
        default=None,
        help="Optional output path for stage=infer report.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_pipeline_v1(
        stage=args.stage,
        iterations=args.iterations,
        self_play_games=args.self_play_games,
        mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature_init=args.temperature_init,
        temperature_final=args.temperature_final,
        temperature_threshold=args.temperature_threshold,
        exploration_weight=args.exploration_weight,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        self_play_concurrent_games=args.self_play_concurrent_games,
        self_play_opening_random_moves=args.self_play_opening_random_moves,
        self_play_backend=args.self_play_backend,
        self_play_shard_dir=args.self_play_shard_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        devices=args.devices,
        train_devices=args.train_devices,
        train_strategy=args.train_strategy,
        soft_value_k=args.soft_value_k,
        max_game_plies=args.max_game_plies,
        load_checkpoint=args.load_checkpoint,
        self_play_output=args.self_play_output,
        self_play_input=args.self_play_input,
        self_play_stats_json=args.self_play_stats_json,
        checkpoint_name=args.checkpoint_name,
        metrics_output=args.metrics_output,
        infer_devices=args.infer_devices,
        infer_batch_size=args.infer_batch_size,
        infer_warmup_iters=args.infer_warmup_iters,
        infer_iters=args.infer_iters,
        infer_output=args.infer_output,
    )
