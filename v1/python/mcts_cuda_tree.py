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
    outputs: RootSearchBatchOutput


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
        }
        self._device_overflow = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_illegal = torch.zeros((), dtype=torch.int64, device=self.device)
        self._device_non_finite = torch.zeros((), dtype=torch.int64, device=self.device)
        self._search_graphs: dict[tuple[int, bool], _SearchGraphEntry] = {}
        self._search_graph_warmed: set[tuple[int, bool]] = set()
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
        temperatures: Optional[float | torch.Tensor | list] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random_mask: Optional[torch.Tensor] = None,
    ) -> RootSearchBatchOutput:
        if states.device != self.device:
            states = states.to(self.device)
        batch_size = int(states.batch_size)
        simulations = int(self.config.num_simulations)
        max_nodes = simulations + 1
        action_dim = TOTAL_ACTION_DIM
        state_arena = _CudaStateArena(states, max_nodes=max_nodes)
        shape_nodes = (batch_size, max_nodes)
        shape_edges = (batch_size, max_nodes, action_dim)
        priors = torch.zeros(shape_edges, dtype=torch.float32, device=self.device)
        legal = torch.zeros(shape_edges, dtype=torch.bool, device=self.device)
        metadata = torch.zeros((*shape_edges, 4), dtype=torch.int32, device=self.device)
        children = torch.full(shape_edges, -1, dtype=torch.int32, device=self.device)
        visits = torch.zeros(shape_nodes, dtype=torch.int32, device=self.device)
        value_sum = torch.zeros(shape_nodes, dtype=torch.float64, device=self.device)
        initial_value = torch.zeros(shape_nodes, dtype=torch.float64, device=self.device)
        terminal_value = torch.zeros(shape_nodes, dtype=torch.float64, device=self.device)
        player = torch.zeros(shape_nodes, dtype=torch.int8, device=self.device)
        expanded = torch.zeros(shape_nodes, dtype=torch.bool, device=self.device)
        terminal = torch.zeros(shape_nodes, dtype=torch.bool, device=self.device)
        parent = torch.full(shape_nodes, -1, dtype=torch.int32, device=self.device)
        node_action = torch.full(shape_nodes, -1, dtype=torch.int32, device=self.device)
        node_count = torch.ones((batch_size,), dtype=torch.int32, device=self.device)
        tree_indices = torch.arange(batch_size, device=self.device, dtype=torch.int64)

        arena_tensors = (
            priors,
            legal,
            metadata,
            children,
            visits,
            value_sum,
            initial_value,
            terminal_value,
            player,
            expanded,
            terminal,
            parent,
            node_action,
            node_count,
            *(getattr(state_arena, field) for field in _STATE_FIELDS),
        )
        self._peak_arena_bytes = max(
            self._peak_arena_bytes,
            sum(t.numel() * t.element_size() for t in arena_tensors),
        )

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
        effective_root_legal = root_legal.to(torch.bool) & ~root_terminal.view(-1, 1)
        priors[:, 0].copy_(root_priors.to(torch.float32) * effective_root_legal)
        legal[:, 0].copy_(effective_root_legal)
        metadata[:, 0].copy_(root_metadata.to(torch.int32))
        player[:, 0].copy_(states.current_player.to(torch.int8))
        expanded[:, 0] = True
        terminal[:, 0].copy_(root_terminal)
        initial_value[:, 0].copy_(root_values)
        terminal_value[:, 0].copy_(root_values)

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
        self.last_root_visits = child_visits.detach()
        self.last_root_priors = root_priors.detach()
        self.last_root_action_values = action_values.detach()
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
            legal_mask=root_legal.detach(),
            policy_dense=policy_dense.detach(),
            root_value=root_values_out.detach(),
            terminal_mask=terminal[:, 0].detach(),
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

    def search_batch(
        self,
        states: GpuStateBatch,
        temperatures: Optional[float | torch.Tensor | list] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random_mask: Optional[torch.Tensor] = None,
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
                temperatures=static_temperatures,
                add_dirichlet_noise=use_noise,
                force_uniform_random_mask=static_force_uniform,
            )
        self._search_graphs[key] = _SearchGraphEntry(
            graph=graph,
            states=static_states,
            temperatures=static_temperatures,
            force_uniform_random_mask=static_force_uniform,
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
        return {
            **self._counters,
            "arena_peak_bytes": int(self._peak_arena_bytes),
            "precision": str(self.config.precision),
        }
