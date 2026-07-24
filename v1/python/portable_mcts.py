"""Correctness-first full-tree MCTS with CPU rules and PyTorch inference.

The tree and all rule transitions stay in ordinary Python/CPU objects. Network
evaluation is batched across trees and may run on CPU, MPS, or CUDA without
importing ``v0_core`` or any CUDA-only helper.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.game_state import GameState, Player
from src.move_generator import MoveType, apply_move, generate_all_legal_moves
from src.neural_network import bucket_logits_to_scalar, state_to_tensor
from src.policy_batch import TOTAL_DIM, action_to_index, build_combined_logits

from .portable_device import PortableDeviceResolution, resolve_portable_device


@dataclass
class PortableMCTSConfig:
    num_simulations: int = 128
    exploration_weight: float = 1.0
    temperature: float = 1.0
    policy_target_temperature: Optional[float] = None
    policy_target_prior_pseudocount: float = 0.0
    add_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    sample_moves: bool = True


@dataclass
class PortableNode:
    state: GameState
    parent: Optional["PortableNode"] = None
    children: Dict[int, "PortableNode"] = field(default_factory=dict)
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    current_player: Player = field(init=False)
    terminal: bool = field(init=False)
    action_index: Optional[int] = None
    move: Optional[MoveType] = None
    expanded: bool = False
    no_legal_terminal: bool = False
    initial_value: float = 0.0
    model_input: Optional[torch.Tensor] = None
    legal_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.current_player = self.state.current_player
        self.terminal = bool(self.state.is_game_over())

    @property
    def mean_value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return float(self.value_sum / self.visit_count)


class PortableTree:
    """Persistent tree arena supporting reuse after a real move."""

    def __init__(self, state: GameState) -> None:
        self.root = PortableNode(state=state.copy())
        self.nodes: List[PortableNode] = [self.root]

    def advance_root(self, action_index: int) -> bool:
        child = self.root.children.get(int(action_index))
        if child is None:
            return False
        child.parent = None
        self.root = child
        reachable: List[PortableNode] = []
        stack = [child]
        while stack:
            node = stack.pop()
            reachable.append(node)
            stack.extend(node.children.values())
        self.nodes = reachable
        return True


@dataclass(frozen=True)
class PortableEvaluationBatch:
    model_inputs: torch.Tensor
    legal_masks: torch.Tensor
    priors: torch.Tensor
    values: torch.Tensor
    legal_moves: Tuple[Tuple[MoveType, ...], ...]
    fallback_count: int
    fallback_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class PortableSearchOutput:
    model_input: torch.Tensor
    legal_mask: torch.Tensor
    policy_dense: torch.Tensor
    selection_policy_dense: torch.Tensor
    root_priors: torch.Tensor
    root_action_values: torch.Tensor
    root_value: float
    terminal: bool
    chosen_action_index: Optional[int]
    chosen_move: Optional[MoveType]
    visit_counts: Dict[int, int]


def value_for_parent(parent: PortableNode, child: PortableNode) -> float:
    """Return a child's stored mean value in its parent's player perspective."""

    value = child.mean_value
    return value if parent.current_player == child.current_player else -value


def backup_path(path: Sequence[PortableNode], leaf_value: float) -> None:
    """Back up a leaf-current-player value, flipping only on an actual turn change."""

    if not path:
        return
    value = float(leaf_value)
    if not math.isfinite(value):
        raise ValueError(f"Cannot back up non-finite value: {value}")
    for offset in range(len(path) - 1, -1, -1):
        node = path[offset]
        node.visit_count += 1
        node.value_sum += value
        if offset > 0:
            parent = path[offset - 1]
            if parent.current_player != node.current_player:
                value = -value


def terminal_value(state: GameState) -> float:
    """Exact W/D/L utility from ``state.current_player`` perspective."""

    winner = state.get_winner()
    if winner is None:
        return 0.0
    return 1.0 if winner == state.current_player else -1.0


def policy_from_visits_and_priors(
    visits: torch.Tensor,
    priors: torch.Tensor,
    *,
    temperature: float,
    prior_pseudocount: float = 0.0,
) -> torch.Tensor:
    """Create a stable policy and keep beta=0 bit-compatible with visit targets."""

    visit_values = torch.as_tensor(visits, dtype=torch.float32).view(-1)
    prior_values = torch.as_tensor(priors, dtype=torch.float32).view(-1)
    if visit_values.shape != prior_values.shape:
        raise ValueError(
            "visits and priors must have the same shape, "
            f"got {tuple(visit_values.shape)} and {tuple(prior_values.shape)}"
        )
    if not bool(torch.isfinite(visit_values).all().item()):
        raise ValueError("visit counts contain NaN/Inf")
    if bool((visit_values < 0).any().item()):
        raise ValueError("visit counts must be non-negative")
    beta = float(prior_pseudocount)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("prior_pseudocount must be finite and non-negative")

    scores = visit_values
    if beta > 0.0:
        if not bool(torch.isfinite(prior_values).all().item()):
            raise ValueError("root priors contain NaN/Inf")
        normalized_priors = prior_values.clamp_min(1e-8)
        prior_sum = float(normalized_priors.sum().item())
        if not math.isfinite(prior_sum) or prior_sum <= 0.0:
            normalized_priors.fill_(
                1.0 / max(1, int(normalized_priors.numel()))
            )
        else:
            normalized_priors = normalized_priors / prior_sum
        scores = visit_values + (beta * normalized_priors)

    total = float(scores.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("Expanded non-terminal root has no policy mass after search.")
    temp = float(temperature)
    if not math.isfinite(temp) or temp < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if temp <= 1e-6:
        probabilities = torch.zeros_like(scores)
        probabilities[int(torch.argmax(scores).item())] = 1.0
        return probabilities

    logits = torch.full_like(scores, float("-inf"))
    positive = scores.gt(0)
    logits[positive] = torch.log(scores[positive]) / max(temp, 1e-6)
    probabilities = torch.softmax(logits, dim=0)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise RuntimeError("Policy construction produced NaN/Inf.")
    return probabilities


class PortableMCTS:
    """Full PUCT tree search, batched over one selected leaf per tree."""

    def __init__(
        self,
        model,
        config: PortableMCTSConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device_resolution: PortableDeviceResolution = resolve_portable_device(device)
        self.device = self.device_resolution.device
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.inference_batches = 0
        self.timing_seconds: Dict[str, float] = {}
        self.timing_calls: Dict[str, int] = {}

    def record_timing(self, name: str, elapsed_sec: float) -> None:
        key = str(name)
        self.timing_seconds[key] = self.timing_seconds.get(key, 0.0) + max(
            0.0, float(elapsed_sec)
        )
        self.timing_calls[key] = self.timing_calls.get(key, 0) + 1

    def timing_snapshot(
        self, total_elapsed_sec: float
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        denominator = max(1e-9, float(total_elapsed_sec))
        timing_ms = {
            key: float(value * 1000.0) for key, value in self.timing_seconds.items()
        }
        timing_ratio = {
            key: min(1.0, max(0.0, float(value / denominator)))
            for key, value in self.timing_seconds.items()
        }
        return timing_ms, timing_ratio, dict(self.timing_calls)

    def _add_root_noise(self, node: PortableNode) -> None:
        if len(node.children) <= 1:
            return
        indices = sorted(node.children)
        priors = torch.tensor(
            [node.children[index].prior for index in indices], dtype=torch.float32
        )
        alpha = max(float(self.config.dirichlet_alpha), 1e-8)
        noise = torch.distributions.Dirichlet(
            torch.full((len(indices),), alpha, dtype=torch.float32)
        ).sample()
        epsilon = min(max(float(self.config.dirichlet_epsilon), 0.0), 1.0)
        mixed = ((1.0 - epsilon) * priors) + (epsilon * noise)
        mixed.div_(mixed.sum().clamp_min(1e-8))
        for offset, index in enumerate(indices):
            node.children[index].prior = float(mixed[offset].item())

    def evaluate_states(self, states: Sequence[GameState]) -> PortableEvaluationBatch:
        """Batch model inference while generating legal masks with CPU rules."""

        if not states:
            return PortableEvaluationBatch(
                model_inputs=torch.empty((0, 11, 6, 6), dtype=torch.float32),
                legal_masks=torch.empty((0, TOTAL_DIM), dtype=torch.bool),
                priors=torch.empty((0, TOTAL_DIM), dtype=torch.float32),
                values=torch.empty((0,), dtype=torch.float32),
                legal_moves=(),
                fallback_count=self.device_resolution.fallback_count,
                fallback_reasons=self.device_resolution.fallback_reasons,
            )

        started = time.perf_counter()
        model_inputs = torch.cat(
            [state_to_tensor(state, state.current_player) for state in states], dim=0
        ).to(torch.float32)
        self.record_timing("state_encode", time.perf_counter() - started)
        if tuple(model_inputs.shape[1:]) != (11, 6, 6):
            raise ValueError(f"Unexpected model input shape: {tuple(model_inputs.shape)}")

        started = time.perf_counter()
        legal_masks = torch.zeros((len(states), TOTAL_DIM), dtype=torch.bool)
        moves_by_row: List[Tuple[MoveType, ...]] = []
        for row, state in enumerate(states):
            legal_moves = tuple(generate_all_legal_moves(state))
            moves_by_row.append(legal_moves)
            for move in legal_moves:
                action_index = action_to_index(move, GameState.BOARD_SIZE)
                if action_index is None or not 0 <= int(action_index) < TOTAL_DIM:
                    raise ValueError(f"Legal move has no valid 220-d action index: {move}")
                if bool(legal_masks[row, int(action_index)].item()):
                    raise ValueError(
                        f"Two legal moves map to action index {action_index} in state row {row}."
                    )
                legal_masks[row, int(action_index)] = True
        self.record_timing("legal_encode", time.perf_counter() - started)

        started = time.perf_counter()
        device_inputs = model_inputs.to(self.device, dtype=torch.float32)
        device_masks = legal_masks.to(self.device)
        try:
            with torch.inference_mode():
                log_p1, log_p2, log_pmc, raw_values = self.model(device_inputs)
                combined = build_combined_logits(
                    log_p1.view(len(states), -1),
                    log_p2.view(len(states), -1),
                    log_pmc.view(len(states), -1),
                    board_size=GameState.BOARD_SIZE,
                ).to(torch.float32)
                masked = combined.masked_fill(~device_masks, float("-inf"))
                priors_device = torch.zeros_like(masked, dtype=torch.float32)
                valid_rows = device_masks.any(dim=1)
                if bool(valid_rows.any().item()):
                    priors_device[valid_rows] = torch.softmax(masked[valid_rows], dim=1)
                if raw_values.dim() == 2 and int(raw_values.size(1)) == 1:
                    values_device = raw_values[:, 0].to(torch.float32)
                else:
                    values_device = bucket_logits_to_scalar(
                        raw_values, num_bins=int(raw_values.size(1))
                    ).to(torch.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Portable inference failed on device={self.device}; no device fallback was attempted."
            ) from exc

        self.inference_batches += 1
        priors = priors_device.to("cpu")
        values = values_device.view(-1).to("cpu")
        self.record_timing("device_inference", time.perf_counter() - started)
        started = time.perf_counter()
        if tuple(priors.shape) != (len(states), TOTAL_DIM):
            raise ValueError(f"Unexpected prior shape: {tuple(priors.shape)}")
        if not bool(torch.isfinite(priors).all().item()):
            raise ValueError("Portable policy inference produced NaN/Inf.")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("Portable value inference produced NaN/Inf.")
        legal_prob_sum = (priors * legal_masks.to(torch.float32)).sum(dim=1)
        has_legal = legal_masks.any(dim=1)
        if bool(has_legal.any().item()) and not torch.allclose(
            legal_prob_sum[has_legal],
            torch.ones_like(legal_prob_sum[has_legal]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("Portable policy is not normalized over legal actions.")
        self.record_timing("eval_validate", time.perf_counter() - started)

        return PortableEvaluationBatch(
            model_inputs=model_inputs.detach().cpu(),
            legal_masks=legal_masks,
            priors=priors,
            values=values,
            legal_moves=tuple(moves_by_row),
            fallback_count=self.device_resolution.fallback_count,
            fallback_reasons=self.device_resolution.fallback_reasons,
        )

    def _expand_evaluated_node(
        self,
        tree: PortableTree,
        node: PortableNode,
        evaluation: PortableEvaluationBatch,
        row: int,
        *,
        is_root: bool,
        add_noise: bool,
    ) -> float:
        node.model_input = evaluation.model_inputs[row].clone()
        node.legal_mask = evaluation.legal_masks[row].clone()
        node.initial_value = float(evaluation.values[row].item())
        moves = evaluation.legal_moves[row]
        if not moves:
            node.expanded = True
            node.terminal = True
            node.no_legal_terminal = not node.state.is_game_over()
            node.initial_value = (
                -1.0 if node.no_legal_terminal else terminal_value(node.state)
            )
            return node.initial_value

        indexed_moves: List[Tuple[int, MoveType, float]] = []
        for move in moves:
            action_index = action_to_index(move, GameState.BOARD_SIZE)
            if action_index is None:
                raise ValueError(f"Could not encode legal move: {move}")
            indexed_moves.append(
                (int(action_index), move, float(evaluation.priors[row, int(action_index)].item()))
            )
        indexed_moves.sort(key=lambda item: item[0])

        priors = torch.tensor([item[2] for item in indexed_moves], dtype=torch.float32)
        if is_root and add_noise and len(indexed_moves) > 1:
            alpha = max(float(self.config.dirichlet_alpha), 1e-8)
            noise = torch.distributions.Dirichlet(
                torch.full((len(indexed_moves),), alpha, dtype=torch.float32)
            ).sample()
            epsilon = min(max(float(self.config.dirichlet_epsilon), 0.0), 1.0)
            priors = ((1.0 - epsilon) * priors) + (epsilon * noise)
        prior_sum = float(priors.sum().item())
        if not math.isfinite(prior_sum) or prior_sum <= 0.0:
            priors.fill_(1.0 / len(indexed_moves))
        else:
            priors.div_(prior_sum)

        node.children.clear()
        for offset, (action_index, move, _prior) in enumerate(indexed_moves):
            child_state = apply_move(node.state, move, quiet=True)
            child = PortableNode(
                state=child_state,
                parent=node,
                prior=float(priors[offset].item()),
                action_index=action_index,
                move=move,
            )
            node.children[action_index] = child
            tree.nodes.append(child)
        node.expanded = True
        return node.initial_value

    def _select_path(self, root: PortableNode) -> List[PortableNode]:
        path = [root]
        node = root
        while node.expanded and node.children and not node.terminal:
            sqrt_total = math.sqrt(max(1, node.visit_count))
            best_score = -math.inf
            best_index = -1
            best_child: Optional[PortableNode] = None
            for action_index in sorted(node.children):
                child = node.children[action_index]
                q = value_for_parent(node, child) if child.visit_count > 0 else 0.0
                u = (
                    float(self.config.exploration_weight)
                    * float(child.prior)
                    * sqrt_total
                    / (1.0 + child.visit_count)
                )
                score = q + u
                if score > best_score or (score == best_score and action_index < best_index):
                    best_score = score
                    best_index = action_index
                    best_child = child
            if best_child is None:
                break
            node = best_child
            path.append(node)
        return path

    @staticmethod
    def _normalize_temperatures(
        temperatures: Optional[float | Sequence[float]],
        count: int,
        default: float,
    ) -> List[float]:
        if temperatures is None:
            return [float(default)] * count
        if isinstance(temperatures, (float, int)):
            return [float(temperatures)] * count
        values = [float(value) for value in temperatures]
        if len(values) != count:
            raise ValueError(f"Expected {count} temperatures, got {len(values)}.")
        return values

    def _policy_from_visits(
        self,
        root: PortableNode,
        temperature: float,
        prior_pseudocount: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        policy = torch.zeros((TOTAL_DIM,), dtype=torch.float32)
        visit_counts = {
            int(action_index): int(child.visit_count)
            for action_index, child in root.children.items()
        }
        if not visit_counts:
            return policy, visit_counts
        indices = sorted(visit_counts)
        visits = torch.tensor([visit_counts[index] for index in indices], dtype=torch.float32)
        priors = torch.tensor(
            [root.children[index].prior for index in indices],
            dtype=torch.float32,
        )
        probs = policy_from_visits_and_priors(
            visits,
            priors,
            temperature=float(temperature),
            prior_pseudocount=float(prior_pseudocount),
        )
        policy[torch.tensor(indices, dtype=torch.int64)] = probs
        return policy, visit_counts

    @staticmethod
    def _root_priors(root: PortableNode) -> torch.Tensor:
        priors = torch.zeros((TOTAL_DIM,), dtype=torch.float32)
        if root.children:
            indices = sorted(root.children)
            priors[torch.tensor(indices, dtype=torch.int64)] = torch.tensor(
                [root.children[index].prior for index in indices],
                dtype=torch.float32,
            )
        return priors

    @staticmethod
    def _root_action_values(root: PortableNode) -> torch.Tensor:
        values = torch.zeros((TOTAL_DIM,), dtype=torch.float32)
        for action_index, child in root.children.items():
            if child.visit_count > 0:
                values[int(action_index)] = float(value_for_parent(root, child))
        return values

    def search_batch(
        self,
        trees: Sequence[PortableTree],
        *,
        temperatures: Optional[float | Sequence[float]] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random: Optional[Sequence[bool]] = None,
    ) -> List[PortableSearchOutput]:
        if not trees:
            return []
        add_noise = (
            bool(self.config.add_dirichlet_noise)
            if add_dirichlet_noise is None
            else bool(add_dirichlet_noise)
        )
        temp_values = self._normalize_temperatures(
            temperatures, len(trees), float(self.config.temperature)
        )
        force_uniform = (
            [False] * len(trees)
            if force_uniform_random is None
            else [bool(value) for value in force_uniform_random]
        )
        if len(force_uniform) != len(trees):
            raise ValueError(
                f"Expected {len(trees)} force-uniform flags, got {len(force_uniform)}."
            )

        roots_to_expand: List[Tuple[int, PortableTree, PortableNode]] = []
        for row, tree in enumerate(trees):
            root = tree.root
            if root.state.is_game_over():
                root.terminal = True
            elif not root.expanded:
                roots_to_expand.append((row, tree, root))
        if roots_to_expand:
            root_eval = self.evaluate_states([item[2].state for item in roots_to_expand])
            started = time.perf_counter()
            for eval_row, (_output_row, tree, node) in enumerate(roots_to_expand):
                self._expand_evaluated_node(
                    tree,
                    node,
                    root_eval,
                    eval_row,
                    is_root=True,
                    add_noise=add_noise,
                )
            self.record_timing("tree_expand_backup", time.perf_counter() - started)
        if add_noise:
            newly_expanded = {id(item[2]) for item in roots_to_expand}
            for tree in trees:
                if id(tree.root) not in newly_expanded and tree.root.expanded:
                    self._add_root_noise(tree.root)

        simulations = max(1, int(self.config.num_simulations))
        for _simulation in range(simulations):
            started = time.perf_counter()
            pending: List[Tuple[PortableTree, List[PortableNode]]] = []
            for tree in trees:
                root = tree.root
                if root.terminal:
                    continue
                path = self._select_path(root)
                leaf = path[-1]
                if leaf.terminal:
                    leaf_result = -1.0 if leaf.no_legal_terminal else terminal_value(leaf.state)
                    backup_path(path, leaf_result)
                elif leaf.expanded and not leaf.children:
                    leaf.terminal = True
                    leaf.no_legal_terminal = True
                    backup_path(path, -1.0)
                else:
                    pending.append((tree, path))
            self.record_timing("tree_select", time.perf_counter() - started)
            if pending:
                evaluation = self.evaluate_states([path[-1].state for _tree, path in pending])
                started = time.perf_counter()
                for row, (tree, path) in enumerate(pending):
                    leaf = path[-1]
                    value = self._expand_evaluated_node(
                        tree,
                        leaf,
                        evaluation,
                        row,
                        is_root=False,
                        add_noise=False,
                    )
                    backup_path(path, value)
                self.record_timing("tree_expand_backup", time.perf_counter() - started)

        started = time.perf_counter()
        outputs: List[PortableSearchOutput] = []
        for row, tree in enumerate(trees):
            root = tree.root
            if root.model_input is None:
                root.model_input = state_to_tensor(root.state, root.current_player)[0]
            if root.legal_mask is None:
                root.legal_mask = torch.zeros((TOTAL_DIM,), dtype=torch.bool)
            if root.terminal or not root.children:
                outputs.append(
                    PortableSearchOutput(
                        model_input=root.model_input.clone(),
                        legal_mask=root.legal_mask.clone(),
                        policy_dense=torch.zeros((TOTAL_DIM,), dtype=torch.float32),
                        selection_policy_dense=torch.zeros(
                            (TOTAL_DIM,), dtype=torch.float32
                        ),
                        root_priors=self._root_priors(root),
                        root_action_values=self._root_action_values(root),
                        root_value=(
                            -1.0 if root.no_legal_terminal else terminal_value(root.state)
                        ),
                        terminal=True,
                        chosen_action_index=None,
                        chosen_move=None,
                        visit_counts={},
                    )
                )
                continue

            selection_policy, visit_counts = self._policy_from_visits(
                root, temp_values[row]
            )
            target_temperature = self.config.policy_target_temperature
            if target_temperature is None:
                target_temperature = temp_values[row]
            policy, _ = self._policy_from_visits(
                root,
                float(target_temperature),
                float(self.config.policy_target_prior_pseudocount),
            )
            legal_mask = root.legal_mask.to(torch.bool)
            policy = policy * legal_mask.to(torch.float32)
            policy_sum = float(policy.sum().item())
            if not math.isfinite(policy_sum) or abs(policy_sum - 1.0) > 1e-5:
                raise RuntimeError(f"Invalid portable root policy sum: {policy_sum}")
            if int(policy[~legal_mask].count_nonzero().item()) != 0:
                raise RuntimeError("Portable root policy assigned mass to an illegal action.")

            if force_uniform[row]:
                legal_indices = torch.where(legal_mask)[0]
                selected_local = int(torch.randint(int(legal_indices.numel()), (1,)).item())
                chosen_index = int(legal_indices[selected_local].item())
            elif self.config.sample_moves:
                chosen_index = int(
                    torch.multinomial(selection_policy, num_samples=1).item()
                )
            else:
                chosen_index = int(torch.argmax(selection_policy).item())
            child = root.children.get(chosen_index)
            if child is None or child.move is None:
                raise RuntimeError(f"Chosen action {chosen_index} is missing from the root tree.")
            outputs.append(
                PortableSearchOutput(
                    model_input=root.model_input.clone(),
                    legal_mask=legal_mask.clone(),
                    policy_dense=policy.clone(),
                    selection_policy_dense=selection_policy.clone(),
                    root_priors=self._root_priors(root),
                    root_action_values=self._root_action_values(root),
                    root_value=(root.mean_value if root.visit_count > 0 else root.initial_value),
                    terminal=False,
                    chosen_action_index=chosen_index,
                    chosen_move=child.move,
                    visit_counts=visit_counts,
                )
            )
        self.record_timing("policy_select", time.perf_counter() - started)
        return outputs

    def search(
        self,
        tree: PortableTree,
        *,
        temperature: Optional[float] = None,
        add_dirichlet_noise: Optional[bool] = None,
    ) -> PortableSearchOutput:
        return self.search_batch(
            [tree],
            temperatures=(self.config.temperature if temperature is None else temperature),
            add_dirichlet_noise=add_dirichlet_noise,
        )[0]
