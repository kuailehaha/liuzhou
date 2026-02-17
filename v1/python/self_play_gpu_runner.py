"""v1 GPU-first self-play runner with tensor-native output."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

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

    def to_dict(self) -> Dict[str, float]:
        return {
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
    max_game_plies: int = 512,
    sample_moves: bool = True,
    concurrent_games: int = 8,
    child_eval_mode: str = "value_only",
    inference_engine=None,
    verbose: bool = False,
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    """Run v1 self-play fully in tensor form and return training-ready tensors."""

    if num_games <= 0:
        raise ValueError("num_games must be positive.")
    dev = torch.device(device)
    max_plies = max(1, int(max_game_plies))
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
    )
    mcts = V1RootMCTS(
        model=model,
        config=mcts_cfg,
        device=dev,
        inference_engine=inference_engine,
    )
    buffer = TensorTrajectoryBuffer(
        device=dev,
        action_dim=TOTAL_ACTION_DIM,
        max_steps_hint=max_plies,
        concurrent_games_hint=wave_size,
    )

    black_wins = 0
    white_wins = 0
    draws = 0
    game_lengths = torch.zeros((int(num_games),), dtype=torch.int64, device=dev)

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
            search = mcts.search_batch(
                active_states,
                temperatures=active_temps,
                add_dirichlet_noise=add_dirichlet_noise,
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
                buffer.finalize_games(
                    step_index_matrix=step_index_matrix,
                    step_counts=step_counts,
                    slots=finalize_slots,
                    result_from_black=result_local,
                    soft_value_from_black=soft_local,
                )
                global_slots = finalize_slots + int(wave_base)
                game_lengths.index_copy_(0, global_slots, step_counts.index_select(0, finalize_slots))

                black_wins += int(result_local.gt(0).sum().item())
                white_wins += int(result_local.lt(0).sum().item())
                draws += int(result_local.eq(0).sum().item())

        if verbose:
            completed = min(wave_base + wave_games, int(num_games))
            print(
                f"[v1.self_play] games={completed}/{num_games} "
                f"W/L/D={black_wins}/{white_wins}/{draws}"
            )

    elapsed = max(1e-9, time.perf_counter() - started)
    batch = buffer.build()
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
    )
    return batch, stats
