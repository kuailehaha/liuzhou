"""Portable reproduction of the current fixed-one-step-value Root-PUCT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from src.game_state import GameState
from src.move_generator import MoveType, apply_move
from src.policy_batch import TOTAL_DIM, action_to_index

from .portable_mcts import PortableMCTS, PortableMCTSConfig, terminal_value


def allocate_fixed_q_visits(
    priors: torch.Tensor,
    q_values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_simulations: int,
    exploration_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate visits exactly like root-only PUCT, reusing fixed ``q_values``."""

    # Match the production Root-PUCT input/statistics dtype for semantic A/B.
    p = priors.to(torch.float32).view(-1)
    q_fixed = q_values.to(torch.float32).view(-1)
    valid = valid_mask.to(torch.bool).view(-1)
    if p.shape != q_fixed.shape or p.shape != valid.shape:
        raise ValueError("priors, q_values, and valid_mask must have identical flat shapes.")
    if not bool(valid.any().item()):
        return torch.zeros_like(p), torch.zeros_like(p)
    visits = torch.zeros_like(p)
    value_sum = torch.zeros_like(p)
    for _ in range(max(1, int(num_simulations))):
        total = float(visits.sum().item())
        mean = torch.where(visits.gt(0), value_sum / visits.clamp_min(1.0), torch.zeros_like(visits))
        score = mean + (
            float(exploration_weight) * p * math.sqrt(total + 1.0) / (1.0 + visits)
        )
        score = score.masked_fill(~valid, float("-inf"))
        chosen = int(torch.argmax(score).item())
        visits[chosen] += 1.0
        value_sum[chosen] += q_fixed[chosen]
    return visits.to(torch.float32), value_sum.to(torch.float32)


def policy_from_visits(
    visits: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = visits.to(torch.float32).view(-1)
    valid = valid_mask.to(torch.bool).view(-1)
    policy = torch.zeros_like(values)
    if not bool(valid.any().item()):
        return policy
    if float(temperature) <= 1e-6:
        masked = values.masked_fill(~valid, -1.0)
        policy[int(torch.argmax(masked).item())] = 1.0
        return policy
    logits = torch.full_like(values, float("-inf"))
    positive = valid.logical_and(values.gt(0))
    logits[positive] = torch.log(values[positive]) / max(float(temperature), 1e-6)
    if not bool(positive.any().item()):
        policy[valid] = 1.0 / int(valid.sum().item())
    else:
        policy = torch.softmax(logits, dim=0)
    policy *= valid.to(torch.float32)
    return policy / policy.sum().clamp_min(1e-8)


@dataclass(frozen=True)
class PortableRootPUCTOutput:
    model_input: torch.Tensor
    legal_mask: torch.Tensor
    priors: torch.Tensor
    q_values: torch.Tensor
    visits: torch.Tensor
    policy_dense: torch.Tensor
    root_value: float
    chosen_action_index: Optional[int]
    chosen_move: Optional[MoveType]


class PortableRootPUCT:
    """One-step child evaluation plus fixed-q PUCT visit allocation."""

    def __init__(
        self,
        model,
        *,
        device: str,
        num_simulations: int,
        exploration_weight: float = 1.0,
        temperature: float = 1.0,
        sample_moves: bool = False,
    ) -> None:
        self.evaluator = PortableMCTS(
            model=model,
            config=PortableMCTSConfig(
                num_simulations=1,
                exploration_weight=exploration_weight,
                temperature=temperature,
                add_dirichlet_noise=False,
                sample_moves=sample_moves,
            ),
            device=device,
        )
        self.num_simulations = max(1, int(num_simulations))
        self.exploration_weight = float(exploration_weight)
        self.temperature = float(temperature)
        self.sample_moves = bool(sample_moves)

    def search(self, state: GameState) -> PortableRootPUCTOutput:
        root_eval = self.evaluator.evaluate_states([state])
        legal_mask = root_eval.legal_masks[0]
        priors = root_eval.priors[0]
        legal_moves = root_eval.legal_moves[0]
        q_values = torch.zeros((TOTAL_DIM,), dtype=torch.float32)
        move_by_index: Dict[int, MoveType] = {}
        if legal_moves:
            child_states = []
            child_indices = []
            for move in legal_moves:
                index = action_to_index(move, GameState.BOARD_SIZE)
                if index is None:
                    raise ValueError(f"Could not encode legal move: {move}")
                child_indices.append(int(index))
                move_by_index[int(index)] = move
                child_states.append(apply_move(state, move, quiet=True))
            child_eval = self.evaluator.evaluate_states(child_states)
            for row, (index, child_state) in enumerate(zip(child_indices, child_states)):
                child_value = (
                    terminal_value(child_state)
                    if child_state.is_game_over()
                    else float(child_eval.values[row].item())
                )
                if child_state.current_player != state.current_player:
                    child_value = -child_value
                q_values[index] = child_value

        visits, value_sum = allocate_fixed_q_visits(
            priors,
            q_values,
            legal_mask,
            num_simulations=self.num_simulations,
            exploration_weight=self.exploration_weight,
        )
        policy = policy_from_visits(visits, legal_mask, self.temperature)
        if bool(legal_mask.any().item()):
            if self.sample_moves:
                chosen_index = int(torch.multinomial(policy, 1).item())
            else:
                chosen_index = int(torch.argmax(policy).item())
            chosen_move = move_by_index[chosen_index]
        else:
            chosen_index = None
            chosen_move = None
        total_visits = float(visits.sum().item())
        root_value = float(value_sum.sum().item() / total_visits) if total_visits > 0 else 0.0
        return PortableRootPUCTOutput(
            model_input=root_eval.model_inputs[0],
            legal_mask=legal_mask,
            priors=priors,
            q_values=q_values,
            visits=visits,
            policy_dense=policy,
            root_value=root_value,
            chosen_action_index=chosen_index,
            chosen_move=chosen_move,
        )
