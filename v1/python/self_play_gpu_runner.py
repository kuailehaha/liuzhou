"""v1 GPU-first self-play runner with tensor-native output."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

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


def _winner_from_state_tensor(state: GpuStateBatch) -> int:
    """Return winner sign from black perspective: +1 black, -1 white, 0 no winner."""

    phase_val = int(state.phase[0].item())
    if phase_val == int(Phase.PLACEMENT.value):
        return 0
    board = state.board[0]
    black_count = int((board == 1).sum().item())
    white_count = int((board == -1).sum().item())
    if black_count == 0:
        return -1
    if white_count == 0:
        return 1
    return 0


def _is_draw_limit(state: GpuStateBatch) -> bool:
    return bool(
        int(state.move_count[0].item()) >= int(GameState.MAX_MOVE_COUNT)
        or int(state.moves_since_capture[0].item()) >= int(GameState.NO_CAPTURE_DRAW_LIMIT)
    )


def _soft_value_from_tensor(state: GpuStateBatch, soft_value_k: float) -> float:
    board = state.board[0]
    black = float((board == 1).sum().item())
    white = float((board == -1).sum().item())
    material_delta = (black - white) / float(GameState.BOARD_SIZE * GameState.BOARD_SIZE)
    return float(math.tanh(float(soft_value_k) * material_delta))


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
    verbose: bool = False,
) -> Tuple[TensorSelfPlayBatch, SelfPlayV1Stats]:
    """Run v1 self-play fully in tensor form and return training-ready tensors."""

    if num_games <= 0:
        raise ValueError("num_games must be positive.")
    dev = torch.device(device)

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
    game_lengths = []

    started = time.perf_counter()

    for game_idx in range(num_games):
        state = GpuStateBatch.initial(dev, batch_size=1)
        step_indices = []
        plies = 0

        while plies < max(1, int(max_game_plies)):
            temperature = (
                float(temperature_init)
                if plies < int(temperature_threshold)
                else float(temperature_final)
            )
            search = mcts.search(
                state,
                temperature=temperature,
                add_dirichlet_noise=add_dirichlet_noise,
            )

            player_sign = int(state.current_player[0].item())
            step_idx = buffer.append_step(
                model_input=search.model_input,
                legal_mask=search.legal_mask,
                policy_dense=search.policy_dense,
                player_sign=player_sign,
            )
            step_indices.append(step_idx)

            if search.terminal:
                result = -float(player_sign)
                soft_value = _soft_value_from_tensor(state, soft_value_k)
                buffer.finalize_game(step_indices, result, soft_value)
                if result > 0:
                    black_wins += 1
                else:
                    white_wins += 1
                break

            if search.chosen_action_code is None:
                soft_value = _soft_value_from_tensor(state, soft_value_k)
                buffer.finalize_game(step_indices, 0.0, soft_value)
                draws += 1
                break

            state = mcts.apply_action(state, search.chosen_action_code.view(1, 4))
            plies += 1

            winner = _winner_from_state_tensor(state)
            if winner != 0:
                soft_value = _soft_value_from_tensor(state, soft_value_k)
                buffer.finalize_game(step_indices, float(winner), soft_value)
                if winner > 0:
                    black_wins += 1
                else:
                    white_wins += 1
                break

            if _is_draw_limit(state):
                soft_value = _soft_value_from_tensor(state, soft_value_k)
                buffer.finalize_game(step_indices, 0.0, soft_value)
                draws += 1
                break
        else:
            soft_value = _soft_value_from_tensor(state, soft_value_k)
            buffer.finalize_game(step_indices, 0.0, soft_value)
            draws += 1

        game_lengths.append(len(step_indices))
        if verbose:
            print(
                f"[v1.self_play] game={game_idx + 1}/{num_games} "
                f"steps={len(step_indices)} "
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

