"""v1 GPU-first self-play runner with tensor-native output."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from src.game_state import GameState, Phase

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


def _winner_from_board_row(board_row: torch.Tensor, phase_value: int) -> int:
    """Return winner sign from black perspective: +1 black, -1 white, 0 no winner."""

    if phase_value == int(Phase.PLACEMENT.value):
        return 0
    black_count = int((board_row == 1).sum().item())
    white_count = int((board_row == -1).sum().item())
    if black_count == 0:
        return -1
    if white_count == 0:
        return 1
    return 0


def _is_draw_limit_row(move_count: int, moves_since_capture: int) -> bool:
    return bool(
        int(move_count) >= int(GameState.MAX_MOVE_COUNT)
        or int(moves_since_capture) >= int(GameState.NO_CAPTURE_DRAW_LIMIT)
    )


def _soft_value_from_board_row(board_row: torch.Tensor, soft_value_k: float) -> float:
    black = float((board_row == 1).sum().item())
    white = float((board_row == -1).sum().item())
    material_delta = (black - white) / float(GameState.BOARD_SIZE * GameState.BOARD_SIZE)
    return float(math.tanh(float(soft_value_k) * material_delta))


def _index_copy_state(dst: GpuStateBatch, idx: torch.Tensor, src: GpuStateBatch) -> None:
    dst.board.index_copy_(0, idx, src.board)
    dst.marks_black.index_copy_(0, idx, src.marks_black)
    dst.marks_white.index_copy_(0, idx, src.marks_white)
    dst.phase.index_copy_(0, idx, src.phase)
    dst.current_player.index_copy_(0, idx, src.current_player)
    dst.pending_marks_required.index_copy_(0, idx, src.pending_marks_required)
    dst.pending_marks_remaining.index_copy_(0, idx, src.pending_marks_remaining)
    dst.pending_captures_required.index_copy_(0, idx, src.pending_captures_required)
    dst.pending_captures_remaining.index_copy_(0, idx, src.pending_captures_remaining)
    dst.forced_removals_done.index_copy_(0, idx, src.forced_removals_done)
    dst.move_count.index_copy_(0, idx, src.move_count)
    dst.moves_since_capture.index_copy_(0, idx, src.moves_since_capture)


def _finalize_game(
    *,
    buffer: TensorTrajectoryBuffer,
    step_indices: List[int],
    final_state: GpuStateBatch,
    final_local_index: int,
    result_from_black: float,
    soft_value_k: float,
) -> None:
    soft_value = _soft_value_from_board_row(final_state.board[final_local_index], soft_value_k)
    buffer.finalize_game(step_indices, result_from_black, soft_value)


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
    )
    mcts = V1RootMCTS(model=model, config=mcts_cfg, device=dev)
    buffer = TensorTrajectoryBuffer(device=dev, action_dim=TOTAL_ACTION_DIM)

    black_wins = 0
    white_wins = 0
    draws = 0
    game_lengths: List[int] = [0 for _ in range(int(num_games))]

    started = time.perf_counter()

    for wave_base in range(0, int(num_games), wave_size):
        wave_games = min(wave_size, int(num_games) - wave_base)
        states = GpuStateBatch.initial(dev, batch_size=wave_games)
        step_indices_by_slot: List[List[int]] = [[] for _ in range(wave_games)]
        plies = torch.zeros((wave_games,), dtype=torch.int64, device=dev)
        done = torch.zeros((wave_games,), dtype=torch.bool, device=dev)

        while not bool(done.all().item()):
            active_idx = torch.nonzero(~done, as_tuple=False).view(-1)
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

            # Persist one trajectory row for every active game at the current root.
            active_count = int(active_idx.numel())
            for local in range(active_count):
                slot = int(active_idx[local].item())
                player_sign = int(active_states.current_player[local].item())
                step_idx = buffer.append_step(
                    model_input=search.model_input[local],
                    legal_mask=search.legal_mask[local],
                    policy_dense=search.policy_dense[local],
                    player_sign=player_sign,
                )
                step_indices_by_slot[slot].append(step_idx)

            immediate_done_local = search.terminal_mask | (~search.chosen_valid_mask)
            immediate_done_idx = torch.nonzero(immediate_done_local, as_tuple=False).view(-1)
            if int(immediate_done_idx.numel()) > 0:
                for local in immediate_done_idx.tolist():
                    slot = int(active_idx[local].item())
                    current_player = int(active_states.current_player[local].item())
                    if bool(search.terminal_mask[local].item()):
                        result = -float(current_player)
                        if result > 0:
                            black_wins += 1
                        else:
                            white_wins += 1
                    else:
                        result = 0.0
                        draws += 1
                    _finalize_game(
                        buffer=buffer,
                        step_indices=step_indices_by_slot[slot],
                        final_state=active_states,
                        final_local_index=local,
                        result_from_black=result,
                        soft_value_k=soft_value_k,
                    )
                    game_lengths[wave_base + slot] = len(step_indices_by_slot[slot])
                    done[slot] = True

            valid_local = torch.nonzero(~immediate_done_local, as_tuple=False).view(-1)
            if int(valid_local.numel()) == 0:
                continue

            valid_slots = active_idx.index_select(0, valid_local)
            valid_states = active_states.select(valid_local)
            valid_action_codes = search.chosen_action_codes.index_select(0, valid_local)
            next_states = mcts.apply_action(valid_states, valid_action_codes)
            _index_copy_state(states, valid_slots, next_states)
            plies.index_add_(
                0,
                valid_slots,
                torch.ones((int(valid_slots.numel()),), dtype=torch.int64, device=dev),
            )

            for local_next in range(int(valid_slots.numel())):
                slot = int(valid_slots[local_next].item())
                winner = _winner_from_board_row(
                    next_states.board[local_next],
                    int(next_states.phase[local_next].item()),
                )
                hit_max_plies = int(plies[slot].item()) >= max_plies
                draw_limit = _is_draw_limit_row(
                    move_count=int(next_states.move_count[local_next].item()),
                    moves_since_capture=int(next_states.moves_since_capture[local_next].item()),
                )

                if winner != 0:
                    result = float(winner)
                    if winner > 0:
                        black_wins += 1
                    else:
                        white_wins += 1
                elif draw_limit or hit_max_plies:
                    result = 0.0
                    draws += 1
                else:
                    continue

                _finalize_game(
                    buffer=buffer,
                    step_indices=step_indices_by_slot[slot],
                    final_state=next_states,
                    final_local_index=local_next,
                    result_from_black=result,
                    soft_value_k=soft_value_k,
                )
                game_lengths[wave_base + slot] = len(step_indices_by_slot[slot])
                done[slot] = True

        if verbose:
            completed = min(wave_base + wave_games, int(num_games))
            print(
                f"[v1.self_play] games={completed}/{num_games} "
                f"W/L/D={black_wins}/{white_wins}/{draws}"
            )

    elapsed = max(1e-9, time.perf_counter() - started)
    batch = buffer.build()
    avg_len = float(sum(game_lengths) / len(game_lengths)) if game_lengths else 0.0
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
