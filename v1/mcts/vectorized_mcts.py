"""
Vectorized MCTS implementation backed by tensorized neural evaluations.

Exposes a batched interface compatible with the legacy `src.mcts.MCTS` API
while keeping all neural inference inside the v1 tensor pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Set
import time
from contextlib import nullcontext

import torch
from torch.distributions import Dirichlet

from src.game_state import GameState, Phase
from src.move_generator import apply_move

from ..game.move_encoder import (
    ActionEncodingSpec,
    DEFAULT_ACTION_SPEC,
    action_to_index,
    decode_action_indices,
    encode_actions,
    DIRECTIONS,
)
from ..game.state_batch import TensorStateBatch, from_game_states, to_game_states
from ..net.encoding import project_policy_logits, states_to_model_input

ACTION_KIND_INVALID = 0
ACTION_KIND_PLACEMENT = 1
ACTION_KIND_MOVEMENT = 2
ACTION_KIND_MARK_SELECTION = 3
ACTION_KIND_CAPTURE_SELECTION = 4
ACTION_KIND_FORCED_REMOVAL_SELECTION = 5
ACTION_KIND_COUNTER_REMOVAL_SELECTION = 6
ACTION_KIND_NO_MOVES_REMOVAL_SELECTION = 7
ACTION_KIND_PROCESS_REMOVAL = 8


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


def _indices_to_moves(
    indices: torch.Tensor,
    state: GameState,
    spec: ActionEncodingSpec,
) -> List[dict]:
    """
    Convert flat action indices into structured move dictionaries for a single state.
    """
    moves: List[dict] = []
    if indices.numel() == 0:
        return moves

    cpu_indices = indices.detach().to("cpu")

    board_size = state.BOARD_SIZE
    placement_dim = spec.placement_dim
    movement_dim = spec.movement_dim
    selection_dim = spec.selection_dim

    placement_end = placement_dim
    movement_end = placement_end + movement_dim
    selection_end = movement_end + selection_dim
    phase = state.phase

    dirs = len(DIRECTIONS)

    for raw in cpu_indices.tolist():
        action = int(raw)
        if action < placement_end:
            cell = action
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.PLACEMENT,
                "action_type": "place",
                "position": (r, c),
            }
        elif action < movement_end:
            rel = action - placement_end
            cell_idx, dir_idx = divmod(rel, dirs)
            r_from = cell_idx // board_size
            c_from = cell_idx % board_size
            dr, dc = DIRECTIONS[dir_idx]
            move = {
                "phase": Phase.MOVEMENT,
                "action_type": "move",
                "from_position": (r_from, c_from),
                "to_position": (r_from + dr, c_from + dc),
            }
        elif action < selection_end:
            rel = action - movement_end
            r = rel // board_size
            c = rel % board_size
            if phase == Phase.MARK_SELECTION:
                action_type = "mark"
                target_phase = Phase.MARK_SELECTION
            elif phase == Phase.CAPTURE_SELECTION:
                action_type = "capture"
                target_phase = Phase.CAPTURE_SELECTION
            elif phase == Phase.FORCED_REMOVAL:
                action_type = "remove"
                target_phase = Phase.FORCED_REMOVAL
            elif phase == Phase.COUNTER_REMOVAL:
                action_type = "counter_remove"
                target_phase = Phase.COUNTER_REMOVAL
            elif phase == Phase.MOVEMENT:
                action_type = "no_moves_remove"
                target_phase = Phase.MOVEMENT
            else:
                action_type = "select"
                target_phase = phase
            move = {
                "phase": target_phase,
                "action_type": action_type,
                "position": (r, c),
            }
        else:
            aux_index = action - selection_end
            if aux_index == 0:
                move = {
                    "phase": Phase.REMOVAL,
                    "action_type": "process_removal",
                }
            else:
                continue
        moves.append(move)

    return moves


def _metadata_to_moves(
    metadata: torch.Tensor,
    state: GameState,
    spec: ActionEncodingSpec,
) -> List[dict]:
    """
    Convert fast encode metadata into structured move dictionaries.
    """
    if metadata.numel() == 0:
        return []

    board_size = state.BOARD_SIZE
    moves: List[dict] = []

    entries = metadata.tolist()
    for kind, primary, secondary, extra in entries:
        if kind == ACTION_KIND_PLACEMENT:
            cell = int(primary)
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.PLACEMENT,
                "action_type": "place",
                "position": (r, c),
            }
        elif kind == ACTION_KIND_MOVEMENT:
            from_idx = int(primary)
            dir_idx = int(secondary)
            if dir_idx < 0 or dir_idx >= len(DIRECTIONS):
                continue
            r_from = from_idx // board_size
            c_from = from_idx % board_size
            dr, dc = DIRECTIONS[dir_idx]
            move = {
                "phase": Phase.MOVEMENT,
                "action_type": "move",
                "from_position": (r_from, c_from),
                "to_position": (r_from + dr, c_from + dc),
            }
        elif kind == ACTION_KIND_MARK_SELECTION:
            cell = int(primary)
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.MARK_SELECTION,
                "action_type": "mark",
                "position": (r, c),
            }
        elif kind == ACTION_KIND_CAPTURE_SELECTION:
            cell = int(primary)
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.CAPTURE_SELECTION,
                "action_type": "capture",
                "position": (r, c),
            }
        elif kind == ACTION_KIND_FORCED_REMOVAL_SELECTION:
            cell = int(primary)
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.FORCED_REMOVAL,
                "action_type": "remove",
                "position": (r, c),
            }
        elif kind == ACTION_KIND_COUNTER_REMOVAL_SELECTION:
            cell = int(primary)
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.COUNTER_REMOVAL,
                "action_type": "counter_remove",
                "position": (r, c),
            }
        elif kind == ACTION_KIND_NO_MOVES_REMOVAL_SELECTION:
            cell = int(primary)
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.MOVEMENT,
                "action_type": "no_moves_remove",
                "position": (r, c),
            }
        elif kind == ACTION_KIND_PROCESS_REMOVAL:
            move = {
                "phase": Phase.REMOVAL,
                "action_type": "process_removal",
            }
        else:
            continue
        moves.append(move)

    return moves


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
        "virtual_loss",
        "cached_legal_indices",   # torch.Tensor | None  (shape: [L], dtype: long)
        "cached_legal_moves",     # List[dict] | None
        "cached_legal_metadata",  # torch.Tensor | None
        "cached_children",        # Dict[int, Tuple[GameState, dict]] | None
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
        self.virtual_loss = 0.0
        self.cached_legal_indices = None
        self.cached_legal_moves = None
        self.cached_legal_metadata = None
        self.cached_children = None

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
        # match legacy: sqrt(max(1, N_parent))
        sqrt_total = math.sqrt(max(1.0, self.visit_count))
        # deterministic tie-break by action index
        for _, child in sorted(self.children_by_index.items()):
            q = child.value()
            u = exploration_weight * child.prior * sqrt_total / (1.0 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def get_best_child_excluding(
        self,
        exploration_weight: float,
        banned_child_ids: Optional[Set[int]] = None,
    ) -> Optional["_MCTSNode"]:
        if not self.children_by_index:
            return None
        banned_child_ids = banned_child_ids or set()
        best_score = -float("inf")
        best_child: Optional[_MCTSNode] = None
        # keep consistent with select_child()/legacy
        sqrt_total = math.sqrt(max(1.0, self.visit_count))

        for _, child in sorted(self.children_by_index.items()):
            if id(child) in banned_child_ids:
                continue
            q = child.value()
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
    log_stats: bool = False


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
        self.profile = bool(getattr(self.config, "log_stats", False))

    def _apply_virtual_loss(self, path: Sequence[_MCTSNode]) -> None:
        loss = float(self.config.virtual_loss_weight)
        if loss <= 0.0:
            return
        for node in path:
            node.virtual_loss += loss
            node.visit_count += loss
            node.value_sum -= loss

    def _revert_virtual_loss(self, path: Sequence[_MCTSNode]) -> None:
        loss = float(self.config.virtual_loss_weight)
        if loss <= 0.0:
            return
        for node in reversed(path):
            if node.virtual_loss <= 0.0:
                continue
            delta = min(loss, node.virtual_loss)
            node.virtual_loss -= delta
            node.visit_count = max(0.0, node.visit_count - delta)
            node.value_sum += delta

    def _select_one_leaf_serial(
        self,
        root: Optional[_MCTSNode],
        exploration_weight: float,
    ) -> Tuple[Optional[_MCTSNode], List[_MCTSNode]]:
        """
        Descend a single root to select exactly one leaf while keeping the
        selection path strictly serial.
        """
        if root is None:
            return None, []
        node = root
        path: List[_MCTSNode] = [node]
        backtrack_steps = 0
        MAX_BACKTRACK_STEPS = 128

        while True:
            if not node.is_expanded() or node.is_terminal():
                return node, path

            child = node.select_child(exploration_weight)
            if child is None:
                if node.parent is None or not path:
                    return node, path
                path.pop()
                node = node.parent
                backtrack_steps += 1
                if backtrack_steps > MAX_BACKTRACK_STEPS:
                    return node, path
                continue

            node = child
            path.append(node)

    # ---------- NEW: Batch expand helper ----------
    def _expand_nodes(
        self,
        nodes: Sequence[_MCTSNode],
        spec: ActionEncodingSpec,
        profile_acc: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, int]:
        """
        Expand a list of nodes in a single forward pass through the model.
        """
        if not nodes:
            return 0, 0

        pending_nodes: List[_MCTSNode] = []
        eval_states: List[GameState] = []

        for node in nodes:
            if node.is_terminal() or node.is_expanded():
                continue
            eval_states.append(node.state)
            pending_nodes.append(node)

        if not pending_nodes:
            return 0, 0

        profiling = self.profile and profile_acc is not None

        encode_start = time.perf_counter() if profiling else None
        tensor_batch = from_game_states(eval_states, device=self.device)
        inputs = states_to_model_input(tensor_batch)
        if encode_start is not None:
            profile_acc["encode"] += time.perf_counter() - encode_start

        fwd_start = time.perf_counter() if profiling else None
        inference_ctx = torch.inference_mode()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        )
        with inference_ctx:
            with autocast_ctx:
                log_p1, log_p2, log_pmc, values = self.model(inputs)
        if fwd_start is not None:
            profile_acc["fwd"] += time.perf_counter() - fwd_start
        values = values.squeeze(1).clamp(-1.0, 1.0)

        pri_mask_start = time.perf_counter() if profiling else None
        legal_eval_mask, action_metadata = encode_actions(
            tensor_batch, spec, return_metadata=True
        )
        if pri_mask_start is not None:
            profile_acc["pri_encode_actions"] += time.perf_counter() - pri_mask_start

        pri_project_start = time.perf_counter() if profiling else None
        probs, _ = project_policy_logits((log_p1, log_p2, log_pmc), legal_eval_mask, spec)
        probs_cpu = probs.to("cpu")
        if pri_project_start is not None:
            profile_acc["pri_project_policy_logits"] += time.perf_counter() - pri_project_start

        apply_start = time.perf_counter() if profiling else None
        for bi, node in enumerate(pending_nodes):
            mask_row = legal_eval_mask[bi]
            legal_indices_gpu = mask_row.nonzero(as_tuple=False).view(-1)
            if legal_indices_gpu.numel() == 0:
                node.cached_legal_indices = torch.empty(0, dtype=torch.long, device=log_p1.device)
                node.cached_legal_moves = []
                node.cached_legal_metadata = None
                node.make_terminal(-1.0)
                continue

            legal_indices = legal_indices_gpu.to(log_p1.device)
            legal_indices_cpu = legal_indices_gpu.to("cpu")

            if action_metadata is not None:
                meta_row = action_metadata[bi]
                legal_meta = meta_row.index_select(0, legal_indices_cpu).contiguous()
            else:
                legal_meta = None

            if legal_meta is not None:
                legal_moves = _metadata_to_moves(legal_meta, eval_states[bi], spec)
            else:
                legal_moves = _indices_to_moves(legal_indices_cpu, eval_states[bi], spec)

            node.cached_legal_indices = legal_indices
            node.cached_legal_moves = legal_moves
            node.cached_legal_metadata = legal_meta

            priors_t = probs_cpu[bi, legal_indices_cpu]
            priors = priors_t.tolist()

            node.expand(
                legal_moves,
                priors,
                spec,
                apply_dirichlet=node.parent is None and self.config.add_dirichlet_noise,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
            )
        if apply_start is not None:
            profile_acc["apply"] += time.perf_counter() - apply_start

        return 1, len(eval_states)

    def _run_simulations_multi_root(
        self,
        roots: Sequence[Tuple[int, _MCTSNode]],
        spec: ActionEncodingSpec,
        profile_acc: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, int, int]:
        """
        Run simulations for multiple roots while keeping each individual root
        strictly serial (one leaf per simulation).
        """
        num_sims = int(self.config.num_simulations)
        if num_sims <= 0 or not roots:
            return 0, 0, 0

        sims_done = [0] * len(roots)
        exploration_weight = float(self.config.exploration_weight)
        nn_calls = 0
        total_b = 0
        waves = 0
        profiling = self.profile and profile_acc is not None

        while True:
            eval_states: List[GameState] = []
            eval_info: List[Tuple[int, _MCTSNode, List[_MCTSNode]]] = []
            progress = False

            for ridx, (_, root_node) in enumerate(roots):
                if sims_done[ridx] >= num_sims:
                    continue

                sel_start = time.perf_counter() if profiling else None
                leaf, path = self._select_one_leaf_serial(root_node, exploration_weight)
                if sel_start is not None:
                    profile_acc["sel"] += time.perf_counter() - sel_start
                if leaf is None:
                    sims_done[ridx] = num_sims
                    continue

                self._apply_virtual_loss(path)

                if leaf.is_terminal():
                    self._revert_virtual_loss(path)
                    _backpropagate(path, leaf.terminal_value)
                    sims_done[ridx] += 1
                    progress = True
                    continue

                eval_states.append(leaf.state)
                eval_info.append((ridx, leaf, path))
                sims_done[ridx] += 1
                progress = True

            if eval_info:
                encode_start = time.perf_counter() if profiling else None
                tensor_batch = from_game_states(eval_states, device=self.device)
                inputs = states_to_model_input(tensor_batch)
                if encode_start is not None:
                    profile_acc["encode"] += time.perf_counter() - encode_start

                fwd_start = time.perf_counter() if profiling else None
                inference_ctx = torch.inference_mode()
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with inference_ctx:
                    with autocast_ctx:
                        log_p1, log_p2, log_pmc, values = self.model(inputs)
                if fwd_start is not None:
                    profile_acc["fwd"] += time.perf_counter() - fwd_start
                values = values.squeeze(1).clamp(-1.0, 1.0)
                nn_calls += 1
                total_b += len(eval_info)

                pri_mask_start = time.perf_counter() if profiling else None
                legal_eval_mask, action_metadata = encode_actions(
                    tensor_batch, spec, return_metadata=True
                )
                if pri_mask_start is not None:
                    profile_acc["pri_encode_actions"] += time.perf_counter() - pri_mask_start

                pri_project_start = time.perf_counter() if profiling else None
                probs, _ = project_policy_logits((log_p1, log_p2, log_pmc), legal_eval_mask, spec)
                probs_cpu = probs.to("cpu")
                if pri_project_start is not None:
                    profile_acc["pri_project_policy_logits"] += time.perf_counter() - pri_project_start

                apply_start = time.perf_counter() if profiling else None
                for bi, (ridx, leaf, path) in enumerate(eval_info):
                    mask_row = legal_eval_mask[bi]
                    legal_indices_gpu = mask_row.nonzero(as_tuple=False).view(-1)
                    if legal_indices_gpu.numel() == 0:
                        leaf.cached_legal_indices = torch.empty(
                            0, dtype=torch.long, device=log_p1.device
                        )
                        leaf.cached_legal_moves = []
                        leaf.cached_legal_metadata = None
                        self._revert_virtual_loss(path)
                        leaf.make_terminal(-1.0)
                        _backpropagate(path, -1.0)
                        continue

                    legal_indices = legal_indices_gpu.to(log_p1.device)
                    legal_indices_cpu = legal_indices_gpu.to("cpu")

                    if action_metadata is not None:
                        meta_row = action_metadata[bi]
                        legal_meta = meta_row.index_select(0, legal_indices_cpu).contiguous()
                    else:
                        legal_meta = None

                    if legal_meta is not None:
                        legal_moves = _metadata_to_moves(legal_meta, eval_states[bi], spec)
                    else:
                        legal_moves = _indices_to_moves(legal_indices_cpu, eval_states[bi], spec)

                    leaf.cached_legal_indices = legal_indices
                    leaf.cached_legal_moves = legal_moves
                    leaf.cached_legal_metadata = legal_meta

                    priors_t = probs_cpu[bi, legal_indices_cpu]
                    priors = priors_t.tolist()

                    self._revert_virtual_loss(path)
                    leaf.expand(
                        legal_moves,
                        priors,
                        spec,
                        apply_dirichlet=leaf.parent is None and self.config.add_dirichlet_noise,
                        dirichlet_alpha=self.config.dirichlet_alpha,
                        dirichlet_epsilon=self.config.dirichlet_epsilon,
                    )
                    value = float(values[bi].item())
                    _backpropagate(path, value)
                if apply_start is not None:
                    profile_acc["apply"] += time.perf_counter() - apply_start

                if profiling:
                    waves += 1
            elif not progress:
                break

            if all(done >= num_sims for done in sims_done):
                break

        return nn_calls, total_b, waves

    def _run_simulations_for_root(self, idx: int, root: _MCTSNode, spec: ActionEncodingSpec) -> None:
        """
        Legacy helper kept for compatibility. Delegates to the multi-root scheduler
        with a single entry.
        """
        if root is None:
            return
        self._run_simulations_multi_root([(idx, root)], spec)

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
            if self.config.log_stats:
                print("[VectorizedMCTS] nn_calls=0 total_B=0 avg_B=0.00")
            return policies, legal_mask

        nn_calls = 0
        total_b = 0
        profile_acc = (
            {
                "sel": 0.0,
                "moves": 0.0,
                "encode": 0.0,
                "fwd": 0.0,
                "pri_encode_actions": 0.0,
                "pri_project_policy_logits": 0.0,
                "apply": 0.0,
            }
            if self.profile
            else None
        )
        profile_waves = 0

        # ---------- NEW: Pre-expand not-yet-expanded, non-terminal roots ----------
        pending_roots = [
            roots[i]
            for i in active_indices
            if roots[i] is not None and not roots[i].is_terminal() and not roots[i].is_expanded()
        ]
        expand_calls, expand_b = self._expand_nodes(pending_roots, spec, profile_acc)
        nn_calls += expand_calls
        total_b += expand_b

        multi_roots = [(idx, roots[idx]) for idx in active_indices if roots[idx] is not None]
        if multi_roots:
            sim_calls, sim_b, sim_waves = self._run_simulations_multi_root(multi_roots, spec, profile_acc)
            nn_calls += sim_calls
            total_b += sim_b
            profile_waves += sim_waves

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
                row[legal_indices] = 1.0 / legal_indices.numel()
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

        if self.config.log_stats:
            avg_b = (total_b / nn_calls) if nn_calls else 0.0
            print(f"[VectorizedMCTS] nn_calls={nn_calls} total_B={total_b} avg_B={avg_b:.2f}")

        if self.profile and profile_acc is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            total_time = sum(profile_acc.values())
            if total_time > 0.0:
                print(
                    "[VMCTS-profile] "
                    f"waves={profile_waves} "
                    f"sel={profile_acc['sel']/total_time:.2%} "
                    f"moves={profile_acc['moves']/total_time:.2%} "
                    f"encode={profile_acc['encode']/total_time:.2%} "
                    f"fwd={profile_acc['fwd']/total_time:.2%} "
                    f"pri:encode_actions={profile_acc['pri_encode_actions']/total_time:.2%} "
                    f"pri:project_policy_logits={profile_acc['pri_project_policy_logits']/total_time:.2%} "
                    f"apply={profile_acc['apply']/total_time:.2%}"
                )

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



"""
python -m tools.cross_check_mcts --states 1000 --max-random-moves 40 --num-simulations 64 --device cuda --batch-leaves 16 > check_log.txt


[Debug] State 245: phase=Phase.MARK_SELECTION player=Player.BLACK
  Mismatched probabilities:
    idx  192: legacy=0.203125 tensorized=0.093750 diff=0.109375 move={'phase': <Phase.MARK_SELECTION: 2>, 'action_type': 'mark', 'position': (2, 0)}
    idx  213: legacy=0.078125 tensorized=0.187500 diff=0.109375 move={'phase': <Phase.MARK_SELECTION: 2>, 'action_type': 'mark', 'position': (5, 3)}
  GameState snapshot:
    0 1 2 3 4 5
   +-----------+
 0 |○ ○ ○ · ● ○|
 1 |○ ○ ○ ● · ○|
 2 |○ ● ● ● ● ●|
 3 |● ● ○ · ● ●|
 4 |B ○ ○ · ○ ○|
 5 |● · ● ○ ● ●|
   +-----------+
Phase: Phase.MARK_SELECTION, Current Player: Player.BLACK
Marked Black: {(4, 0)}
Marked White: set()
Forced Removals Done: 0
Pending Marks: 1/1
Pending Captures: 0/0
Move Count: 32


Summary:
  Mean L1: 0.000219
  Max L1:  0.218750
  Mean max|diff|: 0.000109
  Max max|diff|:  0.109375
  Legacy search avg: 146.448 ms  (total 146.448s)
  Vectorized search avg: 223.240 ms  (batch total 223.240s)
  Speedup (legacy/vectorized): 0.66x


"""
