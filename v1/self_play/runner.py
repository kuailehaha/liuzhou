"""
Self-play runner scaffold for the tensorized pipeline.

The runner orchestrates batched MCTS calls, action sampling, and rollout data
collection. Implementation will follow once vectorized MCTS and move encodings
are ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math

import torch

from src.game_state import GameState, Player
from src.move_generator import apply_move
from src.neural_network import state_to_tensor, NUM_INPUT_CHANNELS

from ..mcts.vectorized_mcts import VectorizedMCTS, VectorizedMCTSConfig
from ..game.state_batch import from_game_states
from ..game.move_encoder import ActionEncodingSpec, DEFAULT_ACTION_SPEC, decode_action_indices
from ..net.policy_decoder import sample_actions


@dataclass
class SelfPlayConfig:
    mcts: VectorizedMCTSConfig = field(default_factory=VectorizedMCTSConfig)
    temperature_init: float = 1.0
    temperature_final: float = 0.2
    temperature_threshold: int = 30
    max_moves: int = 200
    action_spec: ActionEncodingSpec = field(default_factory=lambda: DEFAULT_ACTION_SPEC)
    soft_value_k: float = 2.0


@dataclass
class SelfPlayBatchResult:
    states: torch.Tensor  # (B, T, C, H, W)
    policies: torch.Tensor  # (B, T, A)
    player_signs: torch.Tensor  # (B, T)
    legal_masks: torch.BoolTensor  # (B, T, A)
    mask: torch.BoolTensor  # (B, T) -> indicates valid timesteps
    results: torch.Tensor  # (B,)
    soft_values: torch.Tensor  # (B,)
    lengths: torch.LongTensor  # (B,)


def run_self_play(
    model,
    batch_size: int,
    device: str = "cpu",
    config: Optional[SelfPlayConfig] = None,
) -> SelfPlayBatchResult:
    """
    Execute a batch of self-play games using the vectorized components.
    """
    cfg = config or SelfPlayConfig()
    device = torch.device(device)
    vmcts = VectorizedMCTS(model=model, config=cfg.mcts, device=str(device))

    games: List[GameState] = [GameState() for _ in range(batch_size)]
    active = [True] * batch_size
    move_counts = [0] * batch_size
    results: List[float] = [0.0] * batch_size
    soft_values: List[float] = [0.0] * batch_size

    state_history: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    policy_history: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    legal_history: List[List[torch.BoolTensor]] = [[] for _ in range(batch_size)]
    sign_history: List[List[float]] = [[] for _ in range(batch_size)]

    board_size = GameState.BOARD_SIZE
    action_dim = cfg.action_spec.total_dim
    global_step = 0

    def compute_soft_value(state: GameState) -> float:
        black_count = state.count_player_pieces(Player.BLACK)
        white_count = state.count_player_pieces(Player.WHITE)
        material_delta = (black_count - white_count) / float(board_size * board_size)
        return math.tanh(cfg.soft_value_k * material_delta)

    while any(active):
        tensor_batch = from_game_states(games, device=device)
        policies, legal_mask = vmcts.search(tensor_batch)

        active_mask_tensor = torch.tensor(active, dtype=torch.bool, device=policies.device)
        legal_mask = legal_mask.clone()
        policies = policies.clone()
        legal_mask[~active_mask_tensor] = False
        policies[~active_mask_tensor] = 0.0

        # Handle players with no legal moves (loss condition -> game ends immediately)
        no_moves = (legal_mask.sum(dim=1) == 0) & active_mask_tensor
        if no_moves.any():
            indices = torch.nonzero(no_moves, as_tuple=False).squeeze(1).tolist()
            for idx in indices:
                state = games[idx]
                # The current player to act loses
                loser = state.current_player
                results[idx] = -1.0 if loser == Player.BLACK else 1.0
                soft_values[idx] = compute_soft_value(state)
                active[idx] = False
                vmcts._trees.pop(idx, None)
            if not any(active):
                break
            active_mask_tensor = torch.tensor(active, dtype=torch.bool, device=policies.device)

        temperature = (
            cfg.temperature_final if global_step >= cfg.temperature_threshold else cfg.temperature_init
        )

        action_indices = sample_actions(
            probs=policies,
            legal_mask=legal_mask,
            temperature=temperature,
            spec=cfg.action_spec,
            active_mask=active_mask_tensor,
        )

        # Record training samples for active games that selected an action
        for idx in range(batch_size):
            if not active[idx]:
                continue
            action_idx = int(action_indices[idx].item())
            if action_idx < 0:
                # No legal move chosen (should not occur since handled above)
                active[idx] = False
                vmcts._trees.pop(idx, None)
                continue
            state_tensor = state_to_tensor(games[idx], games[idx].current_player).squeeze(0).cpu()
            policy_tensor = policies[idx].detach().cpu()
            legal_tensor = legal_mask[idx].detach().cpu()
            state_history[idx].append(state_tensor)
            policy_history[idx].append(policy_tensor)
            legal_history[idx].append(legal_tensor)
            sign = 1.0 if games[idx].current_player == Player.BLACK else -1.0
            sign_history[idx].append(sign)

        vmcts.advance_roots(tensor_batch, action_indices)
        moves = decode_action_indices(action_indices.to("cpu"), tensor_batch, cfg.action_spec)

        for idx, move in enumerate(moves):
            if not active[idx]:
                continue
            action_idx = int(action_indices[idx].item())
            if action_idx < 0 or move is None:
                active[idx] = False
                vmcts._trees.pop(idx, None)
                continue
            try:
                games[idx] = apply_move(games[idx], move, quiet=True)
            except ValueError:
                active[idx] = False
                vmcts._trees.pop(idx, None)
                results[idx] = 0.0
                soft_values[idx] = compute_soft_value(games[idx])
                continue

            move_counts[idx] += 1
            winner = games[idx].get_winner()
            finished = False
            if winner is not None:
                results[idx] = 1.0 if winner == Player.BLACK else -1.0
                finished = True
            elif move_counts[idx] >= cfg.max_moves or games[idx].has_reached_move_limit():
                results[idx] = 0.0
                finished = True

            if finished:
                soft_values[idx] = compute_soft_value(games[idx])
                active[idx] = False
                vmcts._trees.pop(idx, None)

        global_step += 1

    # Prepare padded tensors
    lengths = torch.tensor([len(h) for h in state_history], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 0

    if max_len == 0:
        states_tensor = torch.empty(
            batch_size, 0, NUM_INPUT_CHANNELS, board_size, board_size, dtype=torch.float32
        )
        policies_tensor = torch.empty(batch_size, 0, action_dim, dtype=torch.float32)
        player_sign_tensor = torch.empty(batch_size, 0, dtype=torch.float32)
        legal_tensor = torch.zeros(batch_size, 0, action_dim, dtype=torch.bool)
        step_mask = torch.zeros(batch_size, 0, dtype=torch.bool)
    else:
        states_tensor = torch.zeros(
            batch_size, max_len, NUM_INPUT_CHANNELS, board_size, board_size, dtype=torch.float32
        )
        policies_tensor = torch.zeros(batch_size, max_len, action_dim, dtype=torch.float32)
        legal_tensor = torch.zeros(batch_size, max_len, action_dim, dtype=torch.bool)
        step_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        player_sign_tensor = torch.zeros(batch_size, max_len, dtype=torch.float32)
        for idx in range(batch_size):
            for t, (state_tensor, policy_tensor) in enumerate(zip(state_history[idx], policy_history[idx])):
                states_tensor[idx, t] = state_tensor
                policies_tensor[idx, t] = policy_tensor
                legal_tensor[idx, t] = legal_history[idx][t]
                step_mask[idx, t] = True
                player_sign_tensor[idx, t] = sign_history[idx][t]

    results_tensor = torch.tensor(results, dtype=torch.float32)
    soft_tensor = torch.tensor(soft_values, dtype=torch.float32)

    return SelfPlayBatchResult(
        states=states_tensor,
        policies=policies_tensor,
        player_signs=player_sign_tensor,
        legal_masks=legal_tensor,
        mask=step_mask,
        results=results_tensor,
        soft_values=soft_tensor,
        lengths=lengths,
    )
