"""v1 GPU-first self-play runner with tensor-native output."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Tuple

import torch
import v0_core

from .mcts_gpu import (
    GpuStateBatch,
    RootSearchBatchOutput,
    TOTAL_ACTION_DIM,
    V1RootMCTS,
    V1RootMCTSConfig,
)
from .self_play_types import SelfPlayV1Stats
from .trajectory_buffer import TensorSelfPlayBatch, TensorTrajectoryBuffer

_PIECE_DELTA_MIN = -18
_PIECE_DELTA_MAX = 18
_PIECE_DELTA_BUCKET_COUNT = (_PIECE_DELTA_MAX - _PIECE_DELTA_MIN) + 1


def _select_search_rows(
    search: RootSearchBatchOutput, indices: torch.Tensor
) -> RootSearchBatchOutput:
    return RootSearchBatchOutput(
        model_input=search.model_input.index_select(0, indices),
        legal_mask=search.legal_mask.index_select(0, indices),
        policy_dense=search.policy_dense.index_select(0, indices),
        root_value=search.root_value.index_select(0, indices),
        terminal_mask=search.terminal_mask.index_select(0, indices),
        chosen_action_indices=search.chosen_action_indices.index_select(0, indices),
        chosen_action_codes=search.chosen_action_codes.index_select(0, indices),
        chosen_valid_mask=search.chosen_valid_mask.index_select(0, indices),
    )


def _pad_state_batch(states: GpuStateBatch, target_size: int) -> GpuStateBatch:
    current = int(states.batch_size)
    target = int(target_size)
    if target < current:
        raise ValueError("target_size must not be smaller than the state batch")
    if target == current:
        return states
    padding = GpuStateBatch.initial(states.device, batch_size=target - current)
    return GpuStateBatch(
        **{
            field: torch.cat((getattr(states, field), getattr(padding, field)), dim=0)
            for field in states.__dataclass_fields__
        }
    )


def _power_of_two_bucket(size: int) -> int:
    value = max(1, int(size))
    return 1 << (value - 1).bit_length()


def self_play_v1_gpu(
    model,
    num_games: int,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    device: str,
    add_dirichlet_noise: bool = True,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    soft_value_k: float = 2.0,
    opening_random_moves: int = 0,
    max_game_plies: int = 512,
    sample_moves: bool = True,
    concurrent_games: int = 8,
    child_eval_mode: str = "value_only",
    sparse_ply: int = 1,
    sparse_top_k: int = 8,
    inference_engine=None,
    collect_step_timing: bool = False,
    verbose: bool = False,
    search_backend: str = "cuda_root",
    cuda_tree_precision: str = "amp_fp16",
    policy_target_temperature: float | None = None,
    policy_target_prior_pseudocount: float = 0.0,
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    """Run v1 self-play fully in tensor form and return training-ready tensors."""

    if num_games <= 0:
        raise ValueError("num_games must be positive.")
    dev = torch.device(device)
    max_plies = max(1, int(max_game_plies))
    opening_random_n = max(0, int(opening_random_moves))
    wave_size = max(1, min(int(concurrent_games), int(num_games)))

    mcts_cfg = V1RootMCTSConfig(
        num_simulations=max(1, int(mcts_simulations)),
        exploration_weight=float(exploration_weight),
        temperature=float(temperature_init),
        add_dirichlet_noise=bool(add_dirichlet_noise),
        dirichlet_alpha=float(dirichlet_alpha),
        dirichlet_epsilon=float(dirichlet_epsilon),
        sample_moves=bool(sample_moves),
        child_eval_mode=str(child_eval_mode),
        soft_value_k=float(soft_value_k),
        sparse_ply=max(1, int(sparse_ply)),
        sparse_top_k=max(1, int(sparse_top_k)),
    )
    collect_timing = bool(collect_step_timing)
    backend = str(search_backend).strip().lower()
    if backend == "cuda_tree":
        from .mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig

        if int(sparse_ply) != 1:
            raise ValueError("cuda_tree is full MCTS and does not use sparse_ply")
        mcts = CudaTreeMCTS(
            model=model,
            config=CudaTreeMCTSConfig(
                num_simulations=max(1, int(mcts_simulations)),
                exploration_weight=float(exploration_weight),
                temperature=float(temperature_init),
                add_dirichlet_noise=bool(add_dirichlet_noise),
                dirichlet_alpha=float(dirichlet_alpha),
                dirichlet_epsilon=float(dirichlet_epsilon),
                sample_moves=bool(sample_moves),
                precision=str(cuda_tree_precision),
                policy_target_temperature=policy_target_temperature,
                policy_target_prior_pseudocount=float(
                    policy_target_prior_pseudocount
                ),
            ),
            device=dev,
            inference_engine=inference_engine,
        )
    elif backend == "cuda_root":
        mcts = V1RootMCTS(
            model=model,
            config=mcts_cfg,
            device=dev,
            inference_engine=inference_engine,
            collect_timing=collect_timing,
        )
    else:
        raise ValueError(
            f"self_play_v1_gpu expected cuda_root or cuda_tree, got {search_backend!r}"
        )
    buffer = TensorTrajectoryBuffer(
        device=dev,
        action_dim=TOTAL_ACTION_DIM,
        max_steps_hint=max_plies,
        concurrent_games_hint=wave_size,
    )

    outcome_counts = torch.zeros((3,), dtype=torch.int64, device=dev)
    piece_delta_hist = torch.zeros((_PIECE_DELTA_BUCKET_COUNT,), dtype=torch.int64, device=dev)
    game_lengths = torch.zeros((int(num_games),), dtype=torch.int64, device=dev)
    step_timing_ms: Dict[str, float] = {
        "root_puct_ms": 0.0,
        "pack_writeback_ms": 0.0,
        "self_play_step_ms": 0.0,
        "finalize_ms": 0.0,
    }
    step_timing_calls: Dict[str, int] = {
        "root_puct_ms": 0,
        "pack_writeback_ms": 0,
        "self_play_step_ms": 0,
        "finalize_ms": 0,
    }
    timing_event_queue: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    timed_calls_since_drain = 0
    timing_drain_period = 256

    def _drain_timing_events(force: bool) -> None:
        nonlocal timing_event_queue
        if dev.type != "cuda" or not collect_timing or not timing_event_queue:
            return
        if force:
            torch.cuda.synchronize(dev)
        pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for key, start_evt, end_evt in timing_event_queue:
            if force or end_evt.query():
                elapsed_ms = float(start_evt.elapsed_time(end_evt))
                step_timing_ms[key] = float(step_timing_ms.get(key, 0.0) + elapsed_ms)
                step_timing_calls[key] = int(step_timing_calls.get(key, 0) + 1)
            else:
                pending.append((key, start_evt, end_evt))
        timing_event_queue = pending

    @contextmanager
    def _timed(name: str):
        nonlocal timed_calls_since_drain
        if not collect_timing:
            with nullcontext():
                yield
            return
        if dev.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            try:
                yield
            finally:
                end_evt.record()
                timing_event_queue.append((name, start_evt, end_evt))
                timed_calls_since_drain += 1
                if timed_calls_since_drain >= timing_drain_period:
                    timed_calls_since_drain = 0
                    _drain_timing_events(force=False)
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            step_timing_ms[name] = float(step_timing_ms.get(name, 0.0) + elapsed_ms)
            step_timing_calls[name] = int(step_timing_calls.get(name, 0) + 1)

    @contextmanager
    def _nvtx_range(name: str):
        if dev.type != "cuda":
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    started = time.perf_counter()

    wave_bases = (
        (0,) if backend == "cuda_tree" else range(0, int(num_games), wave_size)
    )
    for wave_base in wave_bases:
        wave_games = min(wave_size, int(num_games) - wave_base)
        states = GpuStateBatch.initial(dev, batch_size=wave_games)
        trajectory_games = int(num_games) if backend == "cuda_tree" else wave_games
        step_index_matrix = torch.full(
            (trajectory_games, max_plies), -1, dtype=torch.int64, device=dev
        )
        step_counts = torch.zeros((trajectory_games,), dtype=torch.int64, device=dev)
        plies = torch.zeros((wave_games,), dtype=torch.int64, device=dev)
        done = torch.zeros((wave_games,), dtype=torch.bool, device=dev)
        slot_game_ids = torch.arange(wave_games, dtype=torch.int64, device=dev)
        next_game_id = int(wave_games)
        completed_games = 0

        while True:
            active_idx = torch.where(~done)[0]
            if int(active_idx.numel()) == 0:
                break

            active_states = states.select(active_idx)
            active_plies = plies.index_select(0, active_idx)
            if backend == "cuda_tree":
                search_size = min(wave_games, _power_of_two_bucket(int(active_idx.numel())))
                search_states = (
                    states
                    if int(active_idx.numel()) == wave_games
                    else _pad_state_batch(active_states, search_size)
                )
                search_plies = active_plies
                if search_size > int(active_idx.numel()):
                    search_plies = torch.cat(
                        (
                            search_plies,
                            torch.full(
                                (search_size - int(active_idx.numel()),),
                                int(temperature_threshold),
                                dtype=search_plies.dtype,
                                device=dev,
                            ),
                        )
                    )
            else:
                search_states = active_states
                search_size = int(active_idx.numel())
                search_plies = active_plies
            search_temps = torch.where(
                search_plies < int(temperature_threshold),
                torch.full_like(search_plies, float(temperature_init), dtype=torch.float32),
                torch.full_like(search_plies, float(temperature_final), dtype=torch.float32),
            )
            search_force_uniform = (
                search_plies < opening_random_n if opening_random_n > 0 else None
            )
            with _nvtx_range("v1.search_batch"):
                search = mcts.search_batch(
                    search_states,
                    temperatures=search_temps,
                    add_dirichlet_noise=add_dirichlet_noise,
                    force_uniform_random_mask=search_force_uniform,
                )
                if backend == "cuda_tree" and search_size != int(active_idx.numel()):
                    search = _select_search_rows(
                        search,
                        torch.arange(
                            int(active_idx.numel()), dtype=torch.int64, device=dev
                        ),
                    )

            step_indices = buffer.append_steps(
                model_input=search.model_input,
                legal_mask=search.legal_mask,
                policy_dense=search.policy_dense,
                player_sign=active_states.current_player,
            )
            trajectory_ids = (
                slot_game_ids.index_select(0, active_idx)
                if backend == "cuda_tree"
                else active_idx
            )
            step_positions = step_counts.index_select(0, trajectory_ids)
            step_index_matrix[trajectory_ids, step_positions] = step_indices
            step_counts.index_add_(
                0,
                trajectory_ids,
                torch.ones((int(active_idx.numel()),), dtype=torch.int64, device=dev),
            )
            with _nvtx_range("v1.self_play_step_inplace"), _timed("self_play_step_ms"):
                finalize_slots, result_local, soft_local = v0_core.self_play_step_inplace(
                    states.board,
                    states.marks_black,
                    states.marks_white,
                    states.phase,
                    states.current_player,
                    states.pending_marks_required,
                    states.pending_marks_remaining,
                    states.pending_captures_required,
                    states.pending_captures_remaining,
                    states.forced_removals_done,
                    states.move_count,
                    states.moves_since_capture,
                    plies,
                    done,
                    active_idx,
                    search.chosen_action_codes,
                    search.terminal_mask,
                    search.chosen_valid_mask,
                    int(max_plies),
                    float(soft_value_k),
                )
            if int(finalize_slots.numel()) > 0:
                # Recompute terminal soft labels with v1 tan shaping so runtime behavior
                # is aligned even if extension binaries are stale.
                final_boards = states.board.index_select(0, finalize_slots)
                black_count = final_boards.eq(1).sum(dim=(1, 2)).to(torch.int64)
                white_count = final_boards.eq(-1).sum(dim=(1, 2)).to(torch.int64)
                piece_delta = black_count - white_count
                bucket_indices = torch.clamp(
                    piece_delta - int(_PIECE_DELTA_MIN),
                    min=0,
                    max=int(_PIECE_DELTA_BUCKET_COUNT - 1),
                )
                piece_delta_hist.add_(
                    torch.bincount(bucket_indices, minlength=int(_PIECE_DELTA_BUCKET_COUNT))
                )
                soft_local = V1RootMCTS._soft_tanh_from_board_black(
                    final_boards,
                    soft_value_k=float(soft_value_k),
                ).to(torch.float32)
                with _nvtx_range("v1.finalize_trajectory_inplace"), _timed("finalize_ms"):
                    trajectory_finalize_slots = (
                        slot_game_ids.index_select(0, finalize_slots)
                        if backend == "cuda_tree"
                        else finalize_slots
                    )
                    finalized_slots, finalized_lengths, outcome_delta = buffer.finalize_games_inplace(
                        step_index_matrix=step_index_matrix,
                        step_counts=step_counts,
                        slots=trajectory_finalize_slots,
                        result_from_black=result_local,
                        soft_value_from_black=soft_local,
                    )
                if int(finalized_slots.numel()) > 0:
                    global_slots = (
                        finalized_slots
                        if backend == "cuda_tree"
                        else finalized_slots + int(wave_base)
                    )
                    game_lengths.index_copy_(0, global_slots, finalized_lengths)
                outcome_counts.add_(outcome_delta)

                if backend == "cuda_tree":
                    finished_count = int(finalize_slots.numel())
                    completed_games += finished_count
                    refill_count = min(
                        finished_count, int(num_games) - int(next_game_id)
                    )
                    if refill_count > 0:
                        refill_slots = finalize_slots[:refill_count]
                        refill_ids = torch.arange(
                            next_game_id,
                            next_game_id + refill_count,
                            dtype=torch.int64,
                            device=dev,
                        )
                        fresh = GpuStateBatch.initial(dev, batch_size=refill_count)
                        for field in fresh.__dataclass_fields__:
                            getattr(states, field).index_copy_(
                                0, refill_slots, getattr(fresh, field)
                            )
                        slot_game_ids.index_copy_(0, refill_slots, refill_ids)
                        plies.index_fill_(0, refill_slots, 0)
                        done.index_fill_(0, refill_slots, False)
                        next_game_id += refill_count

        if verbose:
            completed = (
                completed_games
                if backend == "cuda_tree"
                else min(wave_base + wave_games, int(num_games))
            )
            black_wins = int(outcome_counts[0].item())
            white_wins = int(outcome_counts[1].item())
            draws = int(outcome_counts[2].item())
            print(
                f"[v1.self_play] games={completed}/{num_games} "
                f"W/L/D={black_wins}/{white_wins}/{draws}"
            )

    elapsed = max(1e-9, time.perf_counter() - started)
    _drain_timing_events(force=True)
    batch = buffer.build()
    if backend == "cuda_tree":
        cuda_tree_metrics = mcts.metrics()
        mcts_timing = {
            "timing_ms": {},
            "timing_calls": {},
            "counters": {
                key: int(value)
                for key, value in cuda_tree_metrics.items()
                if isinstance(value, int)
            },
        }
        mcts_metrics = {
            **cuda_tree_metrics,
            "search_backend": "cuda_tree",
            "leaf_evaluations_per_sec": float(
                int(cuda_tree_metrics.get("leaf_evaluations", 0)) / elapsed
            ),
        }
    else:
        mcts_timing = mcts.get_timing(reset=False)
        mcts_metrics = {"search_backend": "cuda_root"}
    for key, value in mcts_timing.get("timing_ms", {}).items():
        step_timing_ms[key] = float(step_timing_ms.get(key, 0.0) + float(value))
    for key, value in mcts_timing.get("timing_calls", {}).items():
        step_timing_calls[key] = int(step_timing_calls.get(key, 0) + int(value))
    tracked_keys = ("root_puct_ms", "pack_writeback_ms", "self_play_step_ms", "finalize_ms")
    total_tracked_ms = float(sum(float(step_timing_ms.get(k, 0.0)) for k in tracked_keys))
    step_timing_ratio = {
        k: (float(step_timing_ms.get(k, 0.0)) / total_tracked_ms if total_tracked_ms > 0.0 else 0.0)
        for k in tracked_keys
    }
    black_wins = int(outcome_counts[0].item())
    white_wins = int(outcome_counts[1].item())
    draws = int(outcome_counts[2].item())
    piece_delta_hist_cpu = piece_delta_hist.to("cpu")
    piece_delta_buckets = {
        str(delta): int(piece_delta_hist_cpu[delta - int(_PIECE_DELTA_MIN)].item())
        for delta in range(int(_PIECE_DELTA_MIN), int(_PIECE_DELTA_MAX) + 1)
    }
    avg_len = float(game_lengths.to(torch.float32).mean().item())
    stats = SelfPlayV1Stats(
        num_games=num_games,
        num_positions=batch.num_samples,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        avg_game_length=avg_len,
        elapsed_sec=elapsed,
        positions_per_sec=float(batch.num_samples / elapsed),
        games_per_sec=float(num_games / elapsed),
        step_timing_ms={k: float(step_timing_ms.get(k, 0.0)) for k in tracked_keys},
        step_timing_ratio={k: float(step_timing_ratio.get(k, 0.0)) for k in tracked_keys},
        step_timing_calls={k: int(step_timing_calls.get(k, 0)) for k in tracked_keys},
        mcts_counters={k: int(v) for k, v in mcts_timing.get("counters", {}).items()},
        piece_delta_buckets=piece_delta_buckets,
        mcts_metrics=mcts_metrics,
    )
    return batch, stats
