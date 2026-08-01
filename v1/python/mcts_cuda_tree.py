"""GPU-resident full-tree MCTS for the V1 CUDA self-play path.

The tree arena, game states, PUCT statistics, legal masks, action metadata and
trajectory-facing outputs stay on CUDA.  A simulation remains sequential
within each tree, while all trees in the batch advance together.  This mirrors
the portable full-MCTS reference without virtual loss or root-only allocation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import v0_core

from .mcts_gpu import (
    PHASE_CAPTURE_SELECTION,
    PHASE_COUNTER_REMOVAL,
    PHASE_MOVEMENT,
    TOTAL_ACTION_DIM,
    GpuStateBatch,
    RootSearchBatchOutput,
    V1RootMCTS,
    V1RootMCTSConfig,
    batch_apply_moves_compat,
)


_STATE_FIELDS = (
    "board",
    "marks_black",
    "marks_white",
    "phase",
    "current_player",
    "pending_marks_required",
    "pending_marks_remaining",
    "pending_captures_required",
    "pending_captures_remaining",
    "forced_removals_done",
    "move_count",
    "moves_since_capture",
)


@dataclass(frozen=True)
class CudaTreeMCTSConfig:
    num_simulations: int = 64
    exploration_weight: float = 1.0
    temperature: float = 1.0
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    sample_moves: bool = True
    precision: str = "amp_fp16"
    policy_target_temperature: Optional[float] = None
    policy_target_prior_pseudocount: float = 0.0


class _CudaStateArena:
    def __init__(self, roots: GpuStateBatch, max_nodes: int) -> None:
        self.batch_size = int(roots.batch_size)
        self.max_nodes = int(max_nodes)
        self.device = roots.device
        if self.device.type != "cuda":
            raise ValueError("cuda_tree requires CUDA game states")
        if self.max_nodes < 2:
            raise ValueError("max_nodes must be at least two")
        for field in _STATE_FIELDS:
            root = getattr(roots, field)
            shape = (self.batch_size, self.max_nodes, *root.shape[1:])
            tensor = torch.zeros(shape, dtype=root.dtype, device=self.device)
            tensor[:, 0].copy_(root)
            setattr(self, field, tensor)

    def flat(self) -> GpuStateBatch:
        values = {}
        for field in _STATE_FIELDS:
            tensor = getattr(self, field)
            values[field] = tensor.view(
                self.batch_size * self.max_nodes, *tensor.shape[2:]
            )
        return GpuStateBatch(**values)

    def gather(self, nodes: torch.Tensor) -> GpuStateBatch:
        indices = torch.arange(self.batch_size, device=self.device, dtype=torch.int64)
        indices = indices * self.max_nodes + nodes.to(torch.int64)
        flat = self.flat()
        return GpuStateBatch(
            **{field: getattr(flat, field).index_select(0, indices) for field in _STATE_FIELDS}
        )

    def write(self, nodes: torch.Tensor, values: GpuStateBatch, mask: torch.Tensor) -> None:
        indices = torch.arange(self.batch_size, device=self.device, dtype=torch.int64)
        indices = indices * self.max_nodes + nodes.to(torch.int64)
        flat = self.flat()
        row_mask = mask.to(torch.bool)
        for field in _STATE_FIELDS:
            target = getattr(flat, field)
            incoming = getattr(values, field)
            old = target.index_select(0, indices)
            shape = (self.batch_size,) + (1,) * (incoming.dim() - 1)
            merged = torch.where(row_mask.view(shape), incoming, old)
            target.index_copy_(0, indices, merged)


class _CudaTreeArena:
    """Fixed-capacity SoA tree storage whose rows survive real game moves."""

    def __init__(self, roots: GpuStateBatch, max_nodes: int) -> None:
        self.batch_size = int(roots.batch_size)
        self.max_nodes = int(max_nodes)
        self.device = roots.device
        self.states = _CudaStateArena(roots, max_nodes=max_nodes)
        node_shape = (self.batch_size, self.max_nodes)
        edge_shape = (self.batch_size, self.max_nodes, TOTAL_ACTION_DIM)
        self.priors = torch.zeros(edge_shape, dtype=torch.float32, device=self.device)
        self.legal = torch.zeros(edge_shape, dtype=torch.bool, device=self.device)
        self.metadata = torch.zeros(
            (*edge_shape, 4), dtype=torch.int32, device=self.device
        )
        self.children = torch.full(
            edge_shape, -1, dtype=torch.int32, device=self.device
        )
        self.visits = torch.zeros(node_shape, dtype=torch.int32, device=self.device)
        self.value_sum = torch.zeros(
            node_shape, dtype=torch.float64, device=self.device
        )
        self.initial_value = torch.zeros(
            node_shape, dtype=torch.float64, device=self.device
        )
        self.terminal_value = torch.zeros(
            node_shape, dtype=torch.float64, device=self.device
        )
        self.player = torch.zeros(node_shape, dtype=torch.int8, device=self.device)
        self.expanded = torch.zeros(node_shape, dtype=torch.bool, device=self.device)
        self.terminal = torch.zeros(node_shape, dtype=torch.bool, device=self.device)
        self.parent = torch.full(
            node_shape, -1, dtype=torch.int32, device=self.device
        )
        self.node_action = torch.full(
            node_shape, -1, dtype=torch.int32, device=self.device
        )
        self.node_count = torch.ones(
            (self.batch_size,), dtype=torch.int32, device=self.device
        )
        self.needs_init = torch.ones(
            (self.batch_size,), dtype=torch.bool, device=self.device
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.priors,
            self.legal,
            self.metadata,
            self.children,
            self.visits,
            self.value_sum,
            self.initial_value,
            self.terminal_value,
            self.player,
            self.expanded,
            self.terminal,
            self.parent,
            self.node_action,
            self.node_count,
            self.needs_init,
            *(getattr(self.states, field) for field in _STATE_FIELDS),
        )

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclass
class _ForwardGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class _SearchGraphEntry:
    graph: torch.cuda.CUDAGraph
    states: GpuStateBatch
    temperatures: torch.Tensor
    force_uniform_random_mask: torch.Tensor
    arena: _CudaTreeArena
    outputs: RootSearchBatchOutput


@dataclass
class _AdvanceGraphEntry:
    graph: torch.cuda.CUDAGraph
    actions: torch.Tensor
    arena: _CudaTreeArena


class _CudaModelGraphRunner:
    """Fixed-shape CUDA Graph cache for the inference-only model replica."""

    def __init__(self, model, device: torch.device, precision: str) -> None:
        replica = copy.deepcopy(model).eval().to(device)
        if precision == "amp_fp16":
            replica = replica.half()
            self.input_dtype = torch.float16
        else:
            replica = replica.float()
            self.input_dtype = torch.float32
        self.model = replica.to(memory_format=torch.channels_last)
        self.device = device
        self.batch_size = 0
        self.entries: dict[int, _ForwardGraphEntry] = {}
        self.capture_count = 0
        self.replay_count = 0
        self.optimized = False

    def _capture(self, inputs: torch.Tensor) -> _ForwardGraphEntry:
        batch = int(inputs.size(0))
        static_input = torch.empty(
            tuple(inputs.shape),
            dtype=self.input_dtype,
            device=self.device,
            memory_format=torch.channels_last,
        )
        static_input.copy_(inputs)
        if not self.optimized:
            with torch.inference_mode():
                traced = torch.jit.trace(self.model, static_input, strict=False)
                traced = torch.jit.freeze(traced)
                self.model = torch.jit.optimize_for_inference(traced)
            self.optimized = True
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(8):
                self.model(static_input)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            outputs = self.model(static_input)
        entry = _ForwardGraphEntry(
            graph=graph,
            static_input=static_input,
            outputs=outputs,
        )
        self.entries[batch] = entry
        self.capture_count += 1
        return entry

    def forward(self, inputs: torch.Tensor, n_valid: int):
        batch = int(inputs.size(0))
        if int(n_valid) != batch:
            raise ValueError("cuda_tree graph runner requires an unpadded fixed batch")
        # A model-only graph cannot be replayed while a caller is capturing the
        # complete MCTS search.  In that case execute the inference replica
        # eagerly: its kernels become part of the outer graph and the fixed
        # precision/layout conversion is itself capture-safe.
        if torch.cuda.is_current_stream_capturing():
            graph_inputs = inputs.to(dtype=self.input_dtype).contiguous(
                memory_format=torch.channels_last
            )
            return self.model(graph_inputs)
        entry = self.entries.get(batch)
        if entry is None:
            entry = self._capture(inputs)
        else:
            entry.static_input.copy_(inputs)
        entry.graph.replay()
        self.replay_count += 1
        return entry.outputs


class CudaTreeMCTS:
    """Sequential-per-tree, batched-across-trees full PUCT search on CUDA."""

    def __init__(
        self,
        model,
        config: CudaTreeMCTSConfig,
        device: torch.device | str,
        inference_engine=None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("cuda_tree is only available on CUDA")
        precision = str(config.precision).strip().lower()
        if precision not in {"fp32", "amp_fp16"}:
            raise ValueError("cuda_tree precision must be 'fp32' or 'amp_fp16'")
        if int(config.num_simulations) <= 0:
            raise ValueError("num_simulations must be positive")
        if not hasattr(v0_core, "cuda_tree_select") or not hasattr(v0_core, "cuda_tree_backup"):
            raise RuntimeError(
                "v0_core was built without cuda_tree kernels; rebuild with BUILD_CUDA_KERNELS=ON"
            )
        root_config = V1RootMCTSConfig(
            num_simulations=int(config.num_simulations),
            exploration_weight=float(config.exploration_weight),
            temperature=float(config.temperature),
            add_dirichlet_noise=bool(config.add_dirichlet_noise),
            dirichlet_alpha=float(config.dirichlet_alpha),
            dirichlet_epsilon=float(config.dirichlet_epsilon),
            sample_moves=bool(config.sample_moves),
            autocast_dtype=("float32" if precision == "fp32" else "float16"),
            child_eval_mode="full",
        )
        self._graph_runner = (
            _CudaModelGraphRunner(model, self.device, precision)
            if inference_engine is None
            else None
        )
        self._evaluator = V1RootMCTS(
            model=model,
            config=root_config,
            device=self.device,
            inference_engine=(
                inference_engine if inference_engine is not None else self._graph_runner
            ),
        )
        self._counters = {
            "leaf_evaluations": 0,
            "inference_batches": 0,
            "arena_overflow": 0,
            "illegal_actions": 0,
            "non_finite": 0,
            "fallback": 0,
            "graph_capture": 0,
            "graph_replay": 0,
            "subtree_compactions": 0,
            "subtree_reuse_hits": 0,
            "subtree_reuse_misses": 0,
            "subtree_retained_nodes": 0,
            "subtree_max_retained_nodes": 0,
            "subtree_layout_migrations": 0,
            "subtree_capacity_resets": 0,
            "subtree_state_mismatches": 0,
            "subtree_graph_capture": 0,
            "subtree_graph_replay": 0,
        }
        self._device_overflow = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_illegal = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_non_finite = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_reuse_hits = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_reuse_misses = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_retained_nodes = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_max_retained_nodes = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self._device_capacity_resets = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_state_mismatches = torch.zeros((), dtype=torch.int64, device=self.device)
        self._search_graphs: dict[tuple[int, bool], _SearchGraphEntry] = {}
        self._search_graph_warmed: set[tuple[int, bool]] = set()
        self._advance_graphs: dict[int, _AdvanceGraphEntry] = {}
        self._advance_graph_warmed: set[int] = set()
        self._arenas: dict[int, _CudaTreeArena] = {}
        self._tree_ids: dict[int, torch.Tensor] = {}
        self._active_batch_size: Optional[int] = None
        self._peak_arena_bytes = 0
        self.last_root_visits: Optional[torch.Tensor] = None
        self.last_root_priors: Optional[torch.Tensor] = None
        self.last_root_action_values: Optional[torch.Tensor] = None

    @staticmethod
    def _terminal_mask_and_value(states: GpuStateBatch) -> tuple[torch.Tensor, torch.Tensor]:
        post_movement = (
            states.phase.eq(PHASE_MOVEMENT)
            | states.phase.eq(PHASE_CAPTURE_SELECTION)
            | states.phase.eq(PHASE_COUNTER_REMOVAL)
        )
        black = states.board.eq(1).sum(dim=(1, 2))
        white = states.board.eq(-1).sum(dim=(1, 2))
        white_wins = post_movement & black.lt(4)
        black_wins = post_movement & ~white_wins & white.lt(4)
        winner = white_wins | black_wins
        draw = ~winner & (states.move_count.ge(144) | states.moves_since_capture.ge(36))
        terminal = winner | draw
        black_value = torch.where(
            black_wins,
            torch.ones_like(black, dtype=torch.float64),
            torch.where(
                white_wins,
                -torch.ones_like(black, dtype=torch.float64),
                torch.zeros_like(black, dtype=torch.float64),
            ),
        )
        player_value = black_value * states.current_player.to(torch.float64)
        return terminal, player_value

    @staticmethod
    def _vectorized_deterministic_action(
        visits: torch.Tensor,
        values: torch.Tensor,
        priors: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        min_int = torch.iinfo(torch.int32).min
        legal_visits = torch.where(legal, visits, torch.full_like(visits, min_int))
        max_visits = legal_visits.max(dim=1, keepdim=True).values
        candidates = legal & visits.eq(max_visits)
        finite_values = torch.where(
            torch.isfinite(values), values, torch.full_like(values, float("-inf"))
        )
        max_values = finite_values.masked_fill(~candidates, float("-inf")).max(
            dim=1, keepdim=True
        ).values
        candidates &= torch.isclose(finite_values, max_values, atol=1e-6, rtol=0.0)
        finite_priors = torch.where(
            torch.isfinite(priors), priors, torch.full_like(priors, float("-inf"))
        )
        max_priors = finite_priors.masked_fill(~candidates, float("-inf")).max(
            dim=1, keepdim=True
        ).values
        candidates &= torch.isclose(finite_priors, max_priors, atol=1e-8, rtol=0.0)
        indices = torch.arange(visits.size(1), device=visits.device, dtype=torch.int64)
        return torch.where(
            candidates,
            indices.view(1, -1),
            torch.full_like(indices.view(1, -1), visits.size(1)),
        ).min(dim=1).values

    @staticmethod
    def _policy(
        visits: torch.Tensor,
        legal: torch.Tensor,
        temperatures: torch.Tensor,
        priors: Optional[torch.Tensor] = None,
        prior_pseudocount: float = 0.0,
    ) -> torch.Tensor:
        scores = visits.to(torch.float32)
        if priors is not None and float(prior_pseudocount) > 0.0:
            scores = scores + float(prior_pseudocount) * priors.to(torch.float32)
        safe_temperatures = temperatures.to(torch.float32).clamp_min(1e-6).view(-1, 1)
        positive = legal & scores.gt(0.0)
        logits = torch.full_like(scores, float("-inf"))
        logits = torch.where(positive, torch.log(scores.clamp_min(1e-30)) / safe_temperatures, logits)
        probabilities = torch.softmax(logits, dim=1)
        legal_float = legal.to(torch.float32)
        uniform = legal_float / legal_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        has_positive = positive.any(dim=1, keepdim=True)
        probabilities = torch.where(has_positive, probabilities, uniform)

        # Portable semantics use an exact argmax distribution at temperature
        # zero, with the lowest action index winning a score tie.
        cold = temperatures.to(torch.float32).view(-1, 1).le(1e-6)
        masked_scores = scores.masked_fill(~legal, float("-inf"))
        cold_actions = masked_scores.argmax(dim=1)
        cold_policy = torch.nn.functional.one_hot(
            cold_actions, num_classes=scores.size(1)
        ).to(torch.float32)
        probabilities = torch.where(cold, cold_policy, probabilities)
        return probabilities * legal_float

    def _evaluate(self, states: GpuStateBatch):
        if self._graph_runner is not None:
            self._graph_runner.batch_size = int(states.batch_size)
        inputs, legal, metadata, priors, values = self._evaluator._evaluate_batch(states)
        finite_priors = torch.isfinite(priors)
        finite_values = torch.isfinite(values)
        self._device_non_finite.add_(
            (~finite_priors).sum(dtype=torch.int64)
            + (~finite_values).sum(dtype=torch.int64)
        )
        priors = torch.where(finite_priors, priors, torch.zeros_like(priors))
        values = torch.where(finite_values, values, torch.zeros_like(values))
        return inputs, legal, metadata, priors, values

    def _search_batch_impl(
        self,
        states: GpuStateBatch,
        arena: _CudaTreeArena,
        temperatures: Optional[float | torch.Tensor | list] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random_mask: Optional[torch.Tensor] = None,
    ) -> RootSearchBatchOutput:
        if states.device != self.device:
            states = states.to(self.device)
        batch_size = int(states.batch_size)
        simulations = int(self.config.num_simulations)
        max_nodes = int(arena.max_nodes)
        action_dim = TOTAL_ACTION_DIM
        state_arena = arena.states
        priors = arena.priors
        legal = arena.legal
        metadata = arena.metadata
        children = arena.children
        visits = arena.visits
        value_sum = arena.value_sum
        initial_value = arena.initial_value
        terminal_value = arena.terminal_value
        player = arena.player
        expanded = arena.expanded
        terminal = arena.terminal
        parent = arena.parent
        node_action = arena.node_action
        node_count = arena.node_count
        tree_indices = torch.arange(batch_size, device=self.device, dtype=torch.int64)

        root_inputs, root_legal, root_metadata, root_priors, root_network_values = self._evaluate(states)
        root_state_terminal, root_terminal_values = self._terminal_mask_and_value(states)
        root_no_legal = ~root_legal.any(dim=1) & ~root_state_terminal
        root_terminal = root_state_terminal | root_no_legal
        root_values = torch.where(
            root_state_terminal,
            root_terminal_values,
            torch.where(
                root_no_legal,
                -torch.ones_like(root_terminal_values),
                root_network_values.to(torch.float64),
            ),
        )
        root_matches = torch.ones((batch_size,), dtype=torch.bool, device=self.device)
        for field in _STATE_FIELDS:
            stored = getattr(state_arena, field)[:, 0]
            incoming = getattr(states, field)
            root_matches &= stored.eq(incoming).view(batch_size, -1).all(dim=1)
        state_mismatch = ~arena.needs_init & ~root_matches
        self._device_state_mismatches.add_(state_mismatch.sum(dtype=torch.int64))
        capacity_reset = node_count.gt(max_nodes - simulations)
        self._device_capacity_resets.add_(capacity_reset.sum(dtype=torch.int64))
        initialize = arena.needs_init | capacity_reset | state_mismatch
        effective_root_legal = root_legal.to(torch.bool) & ~root_terminal.view(-1, 1)
        init_edges = initialize.view(-1, 1)
        priors[:, 0].copy_(
            torch.where(
                init_edges,
                root_priors.to(torch.float32) * effective_root_legal,
                priors[:, 0],
            )
        )
        legal[:, 0].copy_(torch.where(init_edges, effective_root_legal, legal[:, 0]))
        metadata[:, 0].copy_(
            torch.where(initialize.view(-1, 1, 1), root_metadata.to(torch.int32), metadata[:, 0])
        )
        children[:, 0].copy_(
            torch.where(init_edges, torch.full_like(children[:, 0], -1), children[:, 0])
        )
        visits[:, 0].copy_(torch.where(initialize, torch.zeros_like(visits[:, 0]), visits[:, 0]))
        value_sum[:, 0].copy_(
            torch.where(initialize, torch.zeros_like(value_sum[:, 0]), value_sum[:, 0])
        )
        player[:, 0].copy_(
            torch.where(initialize, states.current_player.to(torch.int8), player[:, 0])
        )
        expanded[:, 0].copy_(torch.where(initialize, torch.ones_like(expanded[:, 0]), expanded[:, 0]))
        terminal[:, 0].copy_(torch.where(initialize, root_terminal, terminal[:, 0]))
        initial_value[:, 0].copy_(
            torch.where(initialize, root_values, initial_value[:, 0])
        )
        terminal_value[:, 0].copy_(
            torch.where(initialize, root_values, terminal_value[:, 0])
        )
        parent[:, 0].fill_(-1)
        node_action[:, 0].copy_(
            torch.where(initialize, torch.full_like(node_action[:, 0], -1), node_action[:, 0])
        )
        node_count.copy_(torch.where(initialize, torch.ones_like(node_count), node_count))
        for field in _STATE_FIELDS:
            root_state = getattr(state_arena, field)[:, 0]
            incoming = getattr(states, field)
            shape = (batch_size,) + (1,) * (incoming.dim() - 1)
            root_state.copy_(torch.where(initialize.view(shape), incoming, root_state))
        arena.needs_init.fill_(False)

        use_noise = self.config.add_dirichlet_noise if add_dirichlet_noise is None else bool(add_dirichlet_noise)
        if use_noise:
            alpha = max(float(self.config.dirichlet_alpha), 1e-8)
            epsilon = min(max(float(self.config.dirichlet_epsilon), 0.0), 1.0)
            gamma = torch._standard_gamma(torch.full_like(priors[:, 0], alpha))
            noise = gamma * effective_root_legal.to(torch.float32)
            noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1e-12)
            mixed = (1.0 - epsilon) * priors[:, 0] + epsilon * noise
            priors[:, 0].copy_(mixed * effective_root_legal)

        for _ in range(simulations):
            leaf, paths, path_lengths, selected_actions, allocated, overflow = v0_core.cuda_tree_select(
                priors,
                legal,
                children,
                visits,
                value_sum,
                player,
                expanded,
                terminal,
                parent,
                node_action,
                node_count,
                float(self.config.exploration_weight),
                2 * simulations + 1,
            )
            leaf_long = leaf.to(torch.int64)
            parent_nodes = parent[tree_indices, leaf_long].to(torch.int64).clamp_min(0)
            action_long = selected_actions.to(torch.int64).clamp(min=0, max=action_dim - 1)
            action_codes = metadata[tree_indices, parent_nodes, action_long]
            flat_parent = tree_indices * max_nodes + parent_nodes
            candidate_states = batch_apply_moves_compat(
                state_arena.flat(), action_codes, flat_parent
            )
            state_arena.write(leaf_long, candidate_states, allocated)
            leaf_states = state_arena.gather(leaf_long)
            player[tree_indices, leaf_long] = leaf_states.current_player.to(torch.int8)

            _inputs, leaf_legal, leaf_metadata, leaf_priors, network_values = self._evaluate(leaf_states)
            state_terminal, state_terminal_values = self._terminal_mask_and_value(leaf_states)
            no_legal = ~leaf_legal.any(dim=1)
            is_terminal = state_terminal | no_legal
            evaluated_values = torch.where(
                state_terminal,
                state_terminal_values,
                torch.where(
                    no_legal,
                    -torch.ones_like(state_terminal_values),
                    network_values.to(torch.float64),
                ),
            )
            needs_expand = ~expanded[tree_indices, leaf_long] & path_lengths.gt(0)
            old_leaf_values = terminal_value[tree_indices, leaf_long]
            leaf_values = torch.where(needs_expand, evaluated_values, old_leaf_values)
            effective_legal = leaf_legal.to(torch.bool) & ~is_terminal.view(-1, 1)
            global_leaf = tree_indices * max_nodes + leaf_long
            edge_flat = priors.view(batch_size * max_nodes, action_dim)
            legal_flat = legal.view(batch_size * max_nodes, action_dim)
            metadata_flat = metadata.view(batch_size * max_nodes, action_dim, 4)
            old_prior = edge_flat.index_select(0, global_leaf)
            old_legal = legal_flat.index_select(0, global_leaf)
            old_metadata = metadata_flat.index_select(0, global_leaf)
            edge_flat.index_copy_(
                0,
                global_leaf,
                torch.where(needs_expand.view(-1, 1), leaf_priors.to(torch.float32) * effective_legal, old_prior),
            )
            legal_flat.index_copy_(
                0,
                global_leaf,
                torch.where(needs_expand.view(-1, 1), effective_legal, old_legal),
            )
            metadata_flat.index_copy_(
                0,
                global_leaf,
                torch.where(needs_expand.view(-1, 1, 1), leaf_metadata.to(torch.int32), old_metadata),
            )
            expanded[tree_indices, leaf_long] = (
                expanded[tree_indices, leaf_long] | needs_expand
            )
            terminal[tree_indices, leaf_long] = torch.where(
                needs_expand, is_terminal, terminal[tree_indices, leaf_long]
            )
            initial_value[tree_indices, leaf_long] = torch.where(
                needs_expand, evaluated_values, initial_value[tree_indices, leaf_long]
            )
            terminal_value[tree_indices, leaf_long] = torch.where(
                needs_expand, evaluated_values, terminal_value[tree_indices, leaf_long]
            )
            v0_core.cuda_tree_backup(
                paths,
                path_lengths,
                player,
                leaf_values.to(torch.float64),
                visits,
                value_sum,
            )
            self._device_overflow.add_(overflow.sum(dtype=torch.int64))

        root_legal = legal[:, 0]
        root_priors = priors[:, 0]
        root_children = children[:, 0].to(torch.int64)
        valid_child = root_children.ge(0)
        safe_child = root_children.clamp_min(0)
        child_visits = visits.gather(1, safe_child) * valid_child.to(torch.int32)
        child_sum = value_sum.gather(1, safe_child)
        child_mean = child_sum / child_visits.clamp_min(1).to(torch.float64)
        child_players = player.gather(1, safe_child)
        root_players = player[:, :1].expand_as(child_players)
        action_values = torch.where(
            valid_child & child_visits.gt(0),
            torch.where(root_players.eq(child_players), child_mean, -child_mean),
            torch.zeros_like(child_mean),
        ).to(torch.float32)
        self.last_root_visits = child_visits.detach().clone()
        self.last_root_priors = root_priors.detach().clone()
        self.last_root_action_values = action_values.detach().clone()
        root_visits = visits[:, 0]
        root_values_out = torch.where(
            root_visits.gt(0),
            value_sum[:, 0] / root_visits.clamp_min(1).to(torch.float64),
            initial_value[:, 0],
        ).to(torch.float32)
        temperature_values = V1RootMCTS._normalize_temperatures(
            temperatures,
            batch_size,
            float(self.config.temperature),
            self.device,
        )
        selection_policy = self._policy(child_visits, root_legal, temperature_values)
        target_temperature = (
            temperature_values
            if self.config.policy_target_temperature is None
            else torch.full_like(temperature_values, float(self.config.policy_target_temperature))
        )
        policy_dense = self._policy(
            child_visits,
            root_legal,
            target_temperature,
            priors=root_priors,
            prior_pseudocount=float(self.config.policy_target_prior_pseudocount),
        )
        valid_roots = root_legal.any(dim=1) & ~terminal[:, 0]
        safe_selection = torch.where(
            valid_roots.view(-1, 1),
            selection_policy,
            torch.nn.functional.one_hot(
                torch.zeros(batch_size, dtype=torch.int64, device=self.device),
                num_classes=action_dim,
            ).to(torch.float32),
        )
        if force_uniform_random_mask is not None:
            force_uniform = force_uniform_random_mask.to(self.device, dtype=torch.bool).view(-1)
            uniform = root_legal.to(torch.float32)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            safe_selection = torch.where(force_uniform.view(-1, 1), uniform, safe_selection)
        if self.config.sample_moves:
            chosen = torch.multinomial(safe_selection, num_samples=1).view(-1)
        else:
            chosen = self._vectorized_deterministic_action(
                child_visits, action_values, root_priors, root_legal
            )
            chosen = torch.where(valid_roots, chosen, torch.zeros_like(chosen))
        chosen_codes = metadata[:, 0][tree_indices, chosen]
        chosen_indices = torch.where(valid_roots, chosen, torch.full_like(chosen, -1))
        chosen_is_legal = root_legal[tree_indices, chosen] & valid_roots
        self._device_illegal.add_((valid_roots & ~chosen_is_legal).sum(dtype=torch.int64))

        return RootSearchBatchOutput(
            model_input=root_inputs.detach(),
            legal_mask=root_legal.detach().clone(),
            policy_dense=policy_dense.detach(),
            root_value=root_values_out.detach(),
            terminal_mask=terminal[:, 0].detach().clone(),
            chosen_action_indices=chosen_indices.detach(),
            chosen_action_codes=chosen_codes.detach(),
            chosen_valid_mask=valid_roots.detach(),
        )

    @staticmethod
    def _copy_states(destination: GpuStateBatch, source: GpuStateBatch) -> None:
        for field in _STATE_FIELDS:
            getattr(destination, field).copy_(getattr(source, field))

    @staticmethod
    def _empty_states_like(source: GpuStateBatch) -> GpuStateBatch:
        return GpuStateBatch(
            **{field: torch.empty_like(getattr(source, field)) for field in _STATE_FIELDS}
        )

    def _new_arena(self, states: GpuStateBatch) -> _CudaTreeArena:
        # One search can allocate at most ``num_simulations`` nodes.  Five
        # additional budgets retain the selected descendants observed in
        # sustained sim64 play; rows that still outgrow this fixed bound are
        # reset explicitly and counted before another search starts.
        max_nodes = 6 * int(self.config.num_simulations) + 1
        arena = _CudaTreeArena(states, max_nodes=max_nodes)
        self._peak_arena_bytes = max(self._peak_arena_bytes, arena.nbytes)
        return arena

    @staticmethod
    def _copy_arena_rows(
        destination: _CudaTreeArena,
        source: _CudaTreeArena,
        source_rows: torch.Tensor,
        found: torch.Tensor,
    ) -> None:
        row_mask = found.to(torch.bool)
        for destination_tensor, source_tensor in zip(
            destination.tensors(), source.tensors()
        ):
            gathered = source_tensor.index_select(0, source_rows)
            shape = (destination.batch_size,) + (1,) * (gathered.dim() - 1)
            destination_tensor.copy_(
                torch.where(row_mask.view(shape), gathered, destination_tensor)
            )

    def _drop_batch_arena(self, batch_size: int) -> None:
        self._arenas.pop(int(batch_size), None)
        self._tree_ids.pop(int(batch_size), None)
        for key in [key for key in self._search_graphs if key[0] == int(batch_size)]:
            del self._search_graphs[key]
        self._search_graph_warmed = {
            key for key in self._search_graph_warmed if key[0] != int(batch_size)
        }
        self._advance_graphs.pop(int(batch_size), None)
        self._advance_graph_warmed.discard(int(batch_size))

    def _prepare_arena_layout(
        self,
        states: GpuStateBatch,
        tree_ids: torch.Tensor,
        layout_changed: bool,
    ) -> _CudaTreeArena:
        batch_size = int(states.batch_size)
        existing = self._arenas.get(batch_size)
        if existing is None:
            existing = self._new_arena(states)
            source_batch = self._active_batch_size
            source = (
                self._arenas.get(int(source_batch))
                if source_batch is not None
                else None
            )
            source_ids = (
                self._tree_ids.get(int(source_batch))
                if source_batch is not None
                else None
            )
            if layout_changed and source is not None and source_ids is not None:
                matches = (
                    tree_ids.view(-1, 1).eq(source_ids.view(1, -1))
                    & tree_ids.view(-1, 1).ge(0)
                )
                found = matches.any(dim=1)
                source_rows = matches.to(torch.int64).argmax(dim=1)
                self._copy_arena_rows(existing, source, source_rows, found)
                self._counters["subtree_layout_migrations"] += 1
            self._arenas[batch_size] = existing
            self._tree_ids[batch_size] = tree_ids.clone()
            if (
                layout_changed
                and source_batch is not None
                and int(source_batch) != batch_size
            ):
                self._drop_batch_arena(int(source_batch))
            self._active_batch_size = batch_size
            return existing
        if not layout_changed:
            self._active_batch_size = batch_size
            return existing

        source_batch = self._active_batch_size
        source = self._arenas.get(int(source_batch)) if source_batch is not None else None
        source_ids = self._tree_ids.get(int(source_batch)) if source_batch is not None else None
        replacement = self._new_arena(states)
        if source is not None and source_ids is not None:
            matches = tree_ids.view(-1, 1).eq(source_ids.view(1, -1)) & tree_ids.view(-1, 1).ge(0)
            found = matches.any(dim=1)
            source_rows = matches.to(torch.int64).argmax(dim=1)
            self._copy_arena_rows(replacement, source, source_rows, found)
        self._arenas[batch_size] = replacement
        self._tree_ids[batch_size] = tree_ids.clone()
        self._active_batch_size = batch_size
        for key in [key for key in self._search_graphs if key[0] == batch_size]:
            del self._search_graphs[key]
        self._search_graph_warmed = {
            key for key in self._search_graph_warmed if key[0] != batch_size
        }
        self._advance_graphs.pop(batch_size, None)
        self._advance_graph_warmed.discard(batch_size)
        if source_batch is not None and int(source_batch) != batch_size:
            self._drop_batch_arena(int(source_batch))
        self._counters["subtree_layout_migrations"] += 1
        return replacement

    @staticmethod
    def _gather_compacted_nodes(
        tensor: torch.Tensor,
        new_to_old: torch.Tensor,
        valid_destinations: torch.Tensor,
        fill_value: int | float | bool,
    ) -> torch.Tensor:
        shape = (*new_to_old.shape, *((1,) * (tensor.dim() - 2)))
        gather_index = new_to_old.view(shape).expand_as(tensor)
        gathered = torch.gather(tensor, 1, gather_index)
        mask_shape = (*valid_destinations.shape, *((1,) * (tensor.dim() - 2)))
        return torch.where(
            valid_destinations.view(mask_shape),
            gathered,
            torch.full_like(gathered, fill_value),
        )

    def _advance_roots_impl(
        self, arena: _CudaTreeArena, actions: torch.Tensor
    ) -> None:
        batch_size = int(arena.batch_size)
        tree_indices = torch.arange(batch_size, device=self.device, dtype=torch.int64)
        valid_action = actions.ge(0) & actions.lt(TOTAL_ACTION_DIM)
        safe_actions = actions.clamp(min=0, max=TOTAL_ACTION_DIM - 1)
        selected = arena.children[:, 0][tree_indices, safe_actions].to(torch.int64)
        valid = valid_action & selected.ge(0)

        node_ids = torch.arange(
            arena.max_nodes, device=self.device, dtype=torch.int64
        ).view(1, -1).expand(batch_size, -1)
        up = arena.parent.to(torch.int64).clamp_min(0)
        reachable = node_ids.eq(selected.view(-1, 1))
        rounds = max(1, int(arena.max_nodes - 1).bit_length())
        for _ in range(rounds):
            reachable = reachable | torch.gather(reachable, 1, up)
            up = torch.gather(up, 1, up)
        reachable &= valid.view(-1, 1)
        reachable &= node_ids.lt(arena.node_count.to(torch.int64).view(-1, 1))
        retained = reachable.sum(dim=1, dtype=torch.int64)
        valid &= retained.gt(0)

        sort_keys = torch.where(reachable, node_ids, node_ids + arena.max_nodes)
        new_to_old = torch.argsort(sort_keys, dim=1)
        new_positions = torch.arange(
            arena.max_nodes, device=self.device, dtype=torch.int64
        ).view(1, -1)
        valid_destinations = new_positions.lt(retained.view(-1, 1)) & valid.view(-1, 1)
        ranks = reachable.to(torch.int64).cumsum(dim=1) - 1
        old_to_new = torch.where(reachable, ranks, torch.full_like(ranks, -1))

        for tensor, fill in (
            (arena.priors, 0.0),
            (arena.legal, False),
            (arena.metadata, 0),
            (arena.visits, 0),
            (arena.value_sum, 0.0),
            (arena.initial_value, 0.0),
            (arena.terminal_value, 0.0),
            (arena.player, 0),
            (arena.expanded, False),
            (arena.terminal, False),
            (arena.node_action, -1),
        ):
            tensor.copy_(
                self._gather_compacted_nodes(
                    tensor, new_to_old, valid_destinations, fill
                )
            )

        old_children = self._gather_compacted_nodes(
            arena.children, new_to_old, valid_destinations, -1
        )
        mapped_children = torch.gather(
            old_to_new,
            1,
            old_children.to(torch.int64).clamp_min(0).view(batch_size, -1),
        ).view_as(old_children)
        arena.children.copy_(
            torch.where(old_children.ge(0), mapped_children.to(torch.int32), old_children)
        )
        old_parent = self._gather_compacted_nodes(
            arena.parent, new_to_old, valid_destinations, -1
        )
        mapped_parent = torch.gather(
            old_to_new, 1, old_parent.to(torch.int64).clamp_min(0)
        )
        arena.parent.copy_(
            torch.where(old_parent.ge(0), mapped_parent.to(torch.int32), old_parent)
        )
        arena.parent[:, 0].fill_(-1)
        arena.node_action[:, 0].fill_(-1)
        for field in _STATE_FIELDS:
            tensor = getattr(arena.states, field)
            tensor.copy_(
                self._gather_compacted_nodes(
                    tensor, new_to_old, valid_destinations, False
                )
            )
        arena.node_count.copy_(
            torch.where(valid, retained.to(torch.int32), torch.ones_like(arena.node_count))
        )
        arena.needs_init.copy_(~valid)
        self._device_reuse_hits.add_(valid.sum(dtype=torch.int64))
        self._device_reuse_misses.add_((~valid).sum(dtype=torch.int64))
        self._device_retained_nodes.add_(
            torch.where(valid, retained, torch.zeros_like(retained)).sum(dtype=torch.int64)
        )
        self._device_max_retained_nodes.copy_(
            torch.maximum(
                self._device_max_retained_nodes,
                torch.where(valid, retained, torch.zeros_like(retained)).max(),
            )
        )

    def advance_roots(self, actions: torch.Tensor) -> None:
        """Keep every node reachable from the selected child on the GPU."""
        if self._active_batch_size is None:
            raise RuntimeError("advance_roots requires a preceding search_batch call")
        batch_size = int(self._active_batch_size)
        arena = self._arenas[batch_size]
        actions = actions.to(self.device, dtype=torch.int64).view(-1)
        if int(actions.numel()) != batch_size:
            raise ValueError("advance_roots actions must match the active tree batch")
        self._counters["subtree_compactions"] += 1
        entry = self._advance_graphs.get(batch_size)
        if entry is not None and entry.arena is arena:
            entry.actions.copy_(actions)
            entry.graph.replay()
            self._counters["subtree_graph_replay"] += 1
            return
        if batch_size not in self._advance_graph_warmed:
            self._advance_graph_warmed.add(batch_size)
            self._advance_roots_impl(arena, actions)
            return

        static_actions = actions.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            self._advance_roots_impl(arena, static_actions)
        self._advance_graphs[batch_size] = _AdvanceGraphEntry(
            graph=graph,
            actions=static_actions,
            arena=arena,
        )
        self._counters["subtree_graph_capture"] += 1
        graph.replay()
        self._counters["subtree_graph_replay"] += 1

    def reset_tree_rows(self, rows: torch.Tensor) -> None:
        """Invalidate refilled fixed slots; their next external state becomes root."""
        if self._active_batch_size is None:
            return
        arena = self._arenas[int(self._active_batch_size)]
        rows = rows.to(self.device, dtype=torch.int64).view(-1)
        if int(rows.numel()) == 0:
            return
        if bool(rows.lt(0).any().item()) or bool(rows.ge(arena.batch_size).any().item()):
            raise ValueError("reset_tree_rows row is out of range")
        arena.needs_init.index_fill_(0, rows, True)
        arena.node_count.index_fill_(0, rows, 1)

    def search_batch(
        self,
        states: GpuStateBatch,
        temperatures: Optional[float | torch.Tensor | list] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random_mask: Optional[torch.Tensor] = None,
        tree_ids: Optional[torch.Tensor] = None,
        tree_layout_changed: bool = False,
    ) -> RootSearchBatchOutput:
        if states.device != self.device:
            states = states.to(self.device)
        batch_size = int(states.batch_size)
        use_noise = (
            bool(self.config.add_dirichlet_noise)
            if add_dirichlet_noise is None
            else bool(add_dirichlet_noise)
        )
        temperature_values = V1RootMCTS._normalize_temperatures(
            temperatures,
            batch_size,
            float(self.config.temperature),
            self.device,
        )
        force_uniform = (
            torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
            if force_uniform_random_mask is None
            else force_uniform_random_mask.to(self.device, dtype=torch.bool).view(-1)
        )
        if int(force_uniform.numel()) != batch_size:
            raise ValueError("force_uniform_random_mask must match the state batch")
        normalized_tree_ids = (
            torch.arange(batch_size, dtype=torch.int64, device=self.device)
            if tree_ids is None
            else tree_ids.to(self.device, dtype=torch.int64).view(-1)
        )
        if int(normalized_tree_ids.numel()) != batch_size:
            raise ValueError("tree_ids must match the state batch")
        arena = self._prepare_arena_layout(
            states,
            normalized_tree_ids,
            bool(tree_layout_changed),
        )

        key = (batch_size, use_noise)
        self._counters["leaf_evaluations"] += batch_size * (
            int(self.config.num_simulations) + 1
        )
        self._counters["inference_batches"] += int(self.config.num_simulations) + 1
        entry = self._search_graphs.get(key)
        if entry is not None:
            self._copy_states(entry.states, states)
            entry.temperatures.copy_(temperature_values)
            entry.force_uniform_random_mask.copy_(force_uniform)
            entry.graph.replay()
            self._counters["graph_replay"] += 1
            return entry.outputs

        # Let one eager search initialize all model/library kernels and allocator
        # paths before capturing this fixed batch shape on its next occurrence.
        if key not in self._search_graph_warmed:
            self._search_graph_warmed.add(key)
            return self._search_batch_impl(
                states,
                arena,
                temperatures=temperature_values,
                add_dirichlet_noise=use_noise,
                force_uniform_random_mask=force_uniform,
            )

        static_states = self._empty_states_like(states)
        self._copy_states(static_states, states)
        static_temperatures = temperature_values.clone()
        static_force_uniform = force_uniform.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            outputs = self._search_batch_impl(
                static_states,
                arena,
                temperatures=static_temperatures,
                add_dirichlet_noise=use_noise,
                force_uniform_random_mask=static_force_uniform,
            )
        self._search_graphs[key] = _SearchGraphEntry(
            graph=graph,
            states=static_states,
            temperatures=static_temperatures,
            force_uniform_random_mask=static_force_uniform,
            arena=arena,
            outputs=outputs,
        )
        self._counters["graph_capture"] += 1
        # Capture records kernels but its side effects are not a supported
        # substitute for replay.  Replay once so this call returns results for
        # the current static inputs (not allocator-initialized arena contents).
        graph.replay()
        self._counters["graph_replay"] += 1
        return outputs

    def metrics(self) -> dict[str, int | str]:
        self._counters["arena_overflow"] = int(self._device_overflow.item())
        self._counters["illegal_actions"] = int(self._device_illegal.item())
        self._counters["non_finite"] = int(self._device_non_finite.item())
        self._counters["subtree_reuse_hits"] = int(self._device_reuse_hits.item())
        self._counters["subtree_reuse_misses"] = int(self._device_reuse_misses.item())
        self._counters["subtree_retained_nodes"] = int(
            self._device_retained_nodes.item()
        )
        self._counters["subtree_max_retained_nodes"] = int(
            self._device_max_retained_nodes.item()
        )
        self._counters["subtree_capacity_resets"] = int(
            self._device_capacity_resets.item()
        )
        self._counters["subtree_state_mismatches"] = int(
            self._device_state_mismatches.item()
        )
        return {
            **self._counters,
            "arena_peak_bytes": int(self._peak_arena_bytes),
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.device)
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self.device)
            ),
            "precision": str(self.config.precision),
        }
