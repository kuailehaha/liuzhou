"""v1 GPU-first self-play runner with tensor-native output."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import v0_core

from .mcts_gpu import GpuStateBatch, TOTAL_ACTION_DIM, V1RootMCTS, V1RootMCTSConfig
from .trajectory_buffer import TensorSelfPlayBatch, TensorTrajectoryBuffer


@dataclass
class SelfPlayV1Stats:
    num_games: int
    num_positions: int
    black_wins: int
    white_wins: int
    draws: int
    avg_game_length: float
    elapsed_sec: float
    positions_per_sec: float
    games_per_sec: float
    step_timing_ms: Dict[str, float]
    step_timing_ratio: Dict[str, float]
    step_timing_calls: Dict[str, int]
    mcts_counters: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "num_games": float(self.num_games),
            "num_positions": float(self.num_positions),
            "black_wins": float(self.black_wins),
            "white_wins": float(self.white_wins),
            "draws": float(self.draws),
            "avg_game_length": float(self.avg_game_length),
            "elapsed_sec": float(self.elapsed_sec),
            "positions_per_sec": float(self.positions_per_sec),
            "games_per_sec": float(self.games_per_sec),
        }
        payload["step_timing_ms"] = {k: float(v) for k, v in self.step_timing_ms.items()}
        payload["step_timing_ratio"] = {k: float(v) for k, v in self.step_timing_ratio.items()}
        payload["step_timing_calls"] = {k: int(v) for k, v in self.step_timing_calls.items()}
        payload["mcts_counters"] = {k: int(v) for k, v in self.mcts_counters.items()}
        return payload

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
    inference_engine=None,
    collect_step_timing: bool = False,
    verbose: bool = False,
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
    )
    collect_timing = bool(collect_step_timing)
    mcts = V1RootMCTS(
        model=model,
        config=mcts_cfg,
        device=dev,
        inference_engine=inference_engine,
        collect_timing=collect_timing,
    )
    buffer = TensorTrajectoryBuffer(
        device=dev,
        action_dim=TOTAL_ACTION_DIM,
        max_steps_hint=max_plies,
        concurrent_games_hint=wave_size,
    )

    outcome_counts = torch.zeros((3,), dtype=torch.int64, device=dev)
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

    for wave_base in range(0, int(num_games), wave_size):
        wave_games = min(wave_size, int(num_games) - wave_base)
        states = GpuStateBatch.initial(dev, batch_size=wave_games)
        step_index_matrix = torch.full((wave_games, max_plies), -1, dtype=torch.int64, device=dev)
        step_counts = torch.zeros((wave_games,), dtype=torch.int64, device=dev)
        plies = torch.zeros((wave_games,), dtype=torch.int64, device=dev)
        done = torch.zeros((wave_games,), dtype=torch.bool, device=dev)

        while True:
            active_idx = torch.where(~done)[0]
            if int(active_idx.numel()) == 0:
                break

            active_states = states.select(active_idx)
            active_plies = plies.index_select(0, active_idx)
            active_temps = torch.where(
                active_plies < int(temperature_threshold),
                torch.full_like(active_plies, float(temperature_init), dtype=torch.float32),
                torch.full_like(active_plies, float(temperature_final), dtype=torch.float32),
            )
            active_force_uniform = (
                active_plies < opening_random_n if opening_random_n > 0 else None
            )
            with _nvtx_range("v1.search_batch"):
                search = mcts.search_batch(
                    active_states,
                    temperatures=active_temps,
                    add_dirichlet_noise=add_dirichlet_noise,
                    force_uniform_random_mask=active_force_uniform,
                )

            step_indices = buffer.append_steps(
                model_input=search.model_input,
                legal_mask=search.legal_mask,
                policy_dense=search.policy_dense,
                player_sign=active_states.current_player,
            )
            step_positions = step_counts.index_select(0, active_idx)
            step_index_matrix[active_idx, step_positions] = step_indices
            step_counts.index_add_(
                0,
                active_idx,
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
                soft_local = V1RootMCTS._soft_tan_from_board_black(
                    final_boards,
                    soft_value_k=float(soft_value_k),
                ).to(torch.float32)
                with _nvtx_range("v1.finalize_trajectory_inplace"), _timed("finalize_ms"):
                    finalized_slots, finalized_lengths, outcome_delta = buffer.finalize_games_inplace(
                        step_index_matrix=step_index_matrix,
                        step_counts=step_counts,
                        slots=finalize_slots,
                        result_from_black=result_local,
                        soft_value_from_black=soft_local,
                    )
                if int(finalized_slots.numel()) > 0:
                    global_slots = finalized_slots + int(wave_base)
                    game_lengths.index_copy_(0, global_slots, finalized_lengths)
                outcome_counts.add_(outcome_delta)

        if verbose:
            completed = min(wave_base + wave_games, int(num_games))
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
    mcts_timing = mcts.get_timing(reset=False)
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
    )
    return batch, stats
