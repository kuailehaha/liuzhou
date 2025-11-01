"""
Vectorized MCTS implementation backed by tensorized neural evaluations.

Exposes a batched interface compatible with the legacy `src.mcts.MCTS` API
while keeping all neural inference inside the v1 tensor pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.distributions import Dirichlet

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves

from ..game.move_encoder import (
    ActionEncodingSpec,
    DEFAULT_ACTION_SPEC,
    action_to_index,
    decode_action_indices,
    encode_actions,
)
from ..game.state_batch import TensorStateBatch, from_game_states, to_game_states
from ..net.encoding import project_policy_logits, states_to_model_input


def _move_to_key(move):
    if isinstance(move, dict):
        return tuple(sorted((k, _move_to_key(v)) for k, v in move.items()))
    if isinstance(move, (list, tuple)):
        return tuple(_move_to_key(v) for v in move)
    return move


def _state_signature(state: GameState) -> Tuple:
    board_sig = tuple(tuple(row) for row in state.board)
    marks_black = tuple(sorted(state.marked_black))
    marks_white = tuple(sorted(state.marked_white))
    return (
        state.move_count,
        state.phase.value,
        state.current_player.value,
        board_sig,
        marks_black,
        marks_white,
        state.pending_marks_required,
        state.pending_marks_remaining,
        state.pending_captures_required,
        state.pending_captures_remaining,
        state.forced_removals_done,
    )


class _MCTSNode:
    __slots__ = (
        "state",
        "player",
        "parent",
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "children_by_index",
        "move",
        "action_index",
        "terminal_value",
        "state_signature",
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional["_MCTSNode"] = None,
        prior: float = 1.0,
        move: Optional[dict] = None,
        action_index: Optional[int] = None,
    ) -> None:
        self.state = state
        self.player = state.current_player
        self.parent = parent
        self.prior = float(prior)
        self.visit_count = 0.0
        self.value_sum = 0.0
        self.children: Dict[Tuple, _MCTSNode] = {}
        self.children_by_index: Dict[int, _MCTSNode] = {}
        self.move = move
        self.action_index = action_index
        self.terminal_value = self._evaluate_terminal()
        self.state_signature = _state_signature(state)

    def _evaluate_terminal(self) -> Optional[float]:
        if self.state.is_game_over():
            winner = self.state.get_winner()
            if winner is None:
                return 0.0
            return 1.0 if winner == self.player else -1.0
        return None

    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    def is_expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        legal_moves: Sequence[dict],
        priors: Sequence[float],
        spec: ActionEncodingSpec,
        apply_dirichlet: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        if self.children:
            return

        items: List[Tuple[int, _MCTSNode]] = []
        for move, prior in zip(legal_moves, priors):
            idx = action_to_index(move, self.state.BOARD_SIZE, spec)
            if idx is None:
                continue
            child_state = apply_move(self.state, move, quiet=True)
            child = _MCTSNode(
                state=child_state,
                parent=self,
                prior=float(prior),
                move=move,
                action_index=idx,
            )
            key = _move_to_key(move)
            self.children[key] = child
            self.children_by_index[idx] = child
            items.append((idx, child))

        if not items:
            return

        total_prior = sum(child.prior for _, child in items)
        if total_prior <= 0:
            uniform = 1.0 / len(items)
            for _, child in items:
                child.prior = uniform
        else:
            inv = 1.0 / total_prior
            for _, child in items:
                child.prior *= inv

        if apply_dirichlet and len(items) > 0:
            concentration = torch.full((len(items),), float(dirichlet_alpha), dtype=torch.float32)
            noise = Dirichlet(concentration).sample().tolist()
            eps = float(dirichlet_epsilon)
            for noise_val, (_, child) in zip(noise, items):
                child.prior = (1.0 - eps) * child.prior + eps * float(noise_val)
            total = sum(child.prior for _, child in items)
            if total > 0:
                inv = 1.0 / total
                for _, child in items:
                    child.prior *= inv

    def select_child(self, exploration_weight: float) -> Optional["_MCTSNode"]:
        if not self.children_by_index:
            return None
        best_score = -float("inf")
        best_child: Optional[_MCTSNode] = None
        sqrt_total = math.sqrt(self.visit_count + 1.0)

         # 遍历顺序稳定化：按 action_index 排序，避免同分随机抖动
        for _, child in sorted(self.children_by_index.items()):
            # 关键修正：child.value() 是以 child.player 视角存的，父节点需要取反
            q = -child.value()
            u = exploration_weight * child.prior * sqrt_total / (1.0 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def make_terminal(self, value: float) -> None:
        self.terminal_value = float(value)


def _backpropagate(path: Sequence[_MCTSNode], value: float) -> None:
    current = float(value)
    for node in reversed(path):
        node.visit_count += 1.0
        node.value_sum += current
        current = -current


@dataclass
class VectorizedMCTSConfig:
    num_simulations: int = 128
    exploration_weight: float = 1.0
    temperature: float = 1.0
    action_spec: ActionEncodingSpec = field(default_factory=lambda: DEFAULT_ACTION_SPEC)
    virtual_loss_weight: float = 0.0
    batch_leaves: int = 16
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class VectorizedMCTS:
    """
    Batched Monte Carlo Tree Search using tensorized neural evaluations.
    """

    def __init__(self, model, config: Optional[VectorizedMCTSConfig] = None, device: Optional[str] = None):
        self.model = model
        self.config = config or VectorizedMCTSConfig()
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._roots: Dict[int, _MCTSNode] = {}

    def search(self, batch: TensorStateBatch):
        """
        Run batched simulations for the provided root states.

        Returns
        -------
        policy : torch.Tensor
            Shape (B, action_dim) probability over encoded actions.
        legal_mask : torch.BoolTensor
            Same shape mask indicating legal actions for each sample.
        """
        spec = self.config.action_spec
        legal_mask = encode_actions(batch, spec).to(self.device)
        policies = torch.zeros(
            (batch.batch_size, spec.total_dim), dtype=torch.float32, device=self.device
        )

        states = to_game_states(batch)

        self._roots = {idx: root for idx, root in self._roots.items() if idx < batch.batch_size}

        roots: List[Optional[_MCTSNode]] = []
        for idx, state in enumerate(states):
            if batch.mask_alive is not None and not bool(batch.mask_alive[idx]):
                self._roots.pop(idx, None)
                roots.append(None)
                continue

            sig = _state_signature(state)
            root = self._roots.get(idx)
            if root is None or root.state_signature != sig:
                root = _MCTSNode(state)
                self._roots[idx] = root
            roots.append(root)

        active_indices = [i for i, root in enumerate(roots) if root is not None and legal_mask[i].any().item()]
        if not active_indices:
            return policies, legal_mask

        num_sims = self.config.num_simulations
        batch_leaves = max(1, self.config.batch_leaves)
        sims_done = 0
        while sims_done < num_sims:
            leaves: List[Tuple[int, _MCTSNode, List[_MCTSNode], List[dict]]] = []

            for idx in active_indices:
                root = roots[idx]
                if root is None:
                    continue

                node = root
                path = [node]

                while node.is_expanded() and not node.is_terminal():
                    child = node.select_child(self.config.exploration_weight)
                    if child is None:
                        break
                    node = child
                    path.append(node)

                if node.is_terminal():
                    _backpropagate(path, node.terminal_value)
                    continue

                legal_moves = generate_all_legal_moves(node.state)
                if not legal_moves:
                    node.make_terminal(-1.0)
                    _backpropagate(path, -1.0)
                    continue

                leaves.append((idx, node, path, legal_moves))
                if len(leaves) >= batch_leaves:
                    break

            if not leaves:
                break

            eval_states = [leaf[1].state for leaf in leaves]
            tensor_batch = from_game_states(eval_states, device=self.device)
            inputs = states_to_model_input(tensor_batch)
            with torch.no_grad():
                log_p1, log_p2, log_pmc, values = self.model(inputs)
            values = values.squeeze(1).clamp(-1.0, 1.0)
            legal_eval_mask = encode_actions(tensor_batch, spec)
            probs, _ = project_policy_logits((log_p1, log_p2, log_pmc), legal_eval_mask, spec)

            for bi, (root_idx, node, path, legal_moves) in enumerate(leaves):
                priors = []
                probs_row = probs[bi]
                for move in legal_moves:
                    action_idx = action_to_index(move, node.state.BOARD_SIZE, spec)
                    priors.append(float(probs_row[action_idx].item()) if action_idx is not None else 0.0)

                node.expand(
                    legal_moves,
                    priors,
                    spec,
                    apply_dirichlet=node.parent is None and self.config.add_dirichlet_noise,
                    dirichlet_alpha=self.config.dirichlet_alpha,
                    dirichlet_epsilon=self.config.dirichlet_epsilon,
                )
                value = float(values[bi].item())
                _backpropagate(path, value)

            sims_done += len(leaves)

        for idx in active_indices:
            root = roots[idx]
            if root is None:
                continue
            mask_row = legal_mask[idx]
            row = policies[idx]
            legal_indices = mask_row.nonzero(as_tuple=False).flatten()
            if not legal_indices.numel():
                continue
            row.zero_()

            if not root.children_by_index:
                 # 没有子节点时，不再用均匀分布；回退到“带 mask 的先验”更稳
                with torch.no_grad():
                    tb = from_game_states([root.state], device=self.device)
                    inputs = states_to_model_input(tb)
                    log_p1, log_p2, log_pmc, _ = self.model(inputs)
                    eval_mask = encode_actions(tb, spec).to(self.device)
                    probs_row, _ = project_policy_logits((log_p1, log_p2, log_pmc), eval_mask, spec)
                    row.copy_(probs_row[0])
                continue

            total_visits = sum(child.visit_count for child in root.children_by_index.values())
            if total_visits > 0:
                items = sorted(root.children_by_index.items())
                visits = torch.tensor(
                    [child.visit_count for _, child in items],
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.config.temperature <= 1e-6:
                    best = int(torch.argmax(visits).item())
                    row[items[best][0]] = 1.0
                else:
                    adjusted = visits.pow(1.0 / self.config.temperature)
                    if adjusted.sum().item() <= 0:
                        adjusted = torch.ones_like(adjusted)
                    adjusted = adjusted / adjusted.sum()
                    for (action_idx, _), prob in zip(items, adjusted.tolist()):
                        row[action_idx] = prob
            else:
                items = sorted(root.children_by_index.items())
                priors_tensor = torch.tensor(
                    [child.prior for _, child in items],
                    dtype=torch.float32,
                    device=self.device,
                )
                if priors_tensor.sum().item() <= 0:
                    row[legal_indices] = 1.0 / legal_indices.numel()
                elif self.config.temperature <= 1e-6:
                    best = int(torch.argmax(priors_tensor).item())
                    row[items[best][0]] = 1.0
                else:
                    adjusted = priors_tensor.pow(1.0 / self.config.temperature)
                    if adjusted.sum().item() <= 0:
                        adjusted = torch.ones_like(adjusted)
                    adjusted = adjusted / adjusted.sum()
                    for (action_idx, _), prob in zip(items, adjusted.tolist()):
                        row[action_idx] = prob

            if row.sum().item() <= 0:
                row[legal_indices] = 1.0 / legal_indices.numel()

        return policies, legal_mask

    def advance_roots(self, batch: TensorStateBatch, action_indices: torch.Tensor):
        """
        Update cached roots after external callers apply sampled actions.
        """
        if not self._roots:
            return None

        if action_indices.device != torch.device("cpu"):
            indices_cpu = action_indices.to("cpu")
        else:
            indices_cpu = action_indices

        decoded_moves = decode_action_indices(indices_cpu, batch, self.config.action_spec)

        for idx, move in enumerate(decoded_moves):
            root = self._roots.get(idx)
            if root is None:
                continue
            action_idx = int(indices_cpu[idx].item())
            if move is None or action_idx < 0:
                self._roots.pop(idx, None)
                continue

            child = root.children_by_index.get(action_idx)
            if child is None:
                # Fallback: rebuild from scratch
                try:
                    new_state = apply_move(root.state, move, quiet=True)
                except Exception:
                    self._roots.pop(idx, None)
                    continue
                self._roots[idx] = _MCTSNode(new_state)
            else:
                child.parent = None
                self._roots[idx] = child

        return None
