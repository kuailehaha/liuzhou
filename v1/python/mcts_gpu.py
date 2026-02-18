"""v1 GPU-first root-search MCTS utilities.

This module keeps the hot path on GPU:
- legal-mask generation via ``v0_core.encode_actions_fast``
- policy projection via ``v0_core.project_policy_logits_fast``
- state transition via ``v0_core.batch_apply_moves``

The current implementation is a root-PUCT search. It is intentionally isolated
from v0 to allow incremental migration without changing the v0 path.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

import v0_core
from v0.python.move_encoder import DEFAULT_ACTION_SPEC

PLACEMENT_DIM = int(DEFAULT_ACTION_SPEC.placement_dim)
MOVEMENT_DIM = int(DEFAULT_ACTION_SPEC.movement_dim)
SELECTION_DIM = int(DEFAULT_ACTION_SPEC.selection_dim)
AUXILIARY_DIM = int(DEFAULT_ACTION_SPEC.auxiliary_dim)
TOTAL_ACTION_DIM = int(DEFAULT_ACTION_SPEC.total_dim)

@dataclass
class GpuStateBatch:
    """Tensor-native game state batch for v1 GPU self-play."""

    board: torch.Tensor
    marks_black: torch.Tensor
    marks_white: torch.Tensor
    phase: torch.Tensor
    current_player: torch.Tensor
    pending_marks_required: torch.Tensor
    pending_marks_remaining: torch.Tensor
    pending_captures_required: torch.Tensor
    pending_captures_remaining: torch.Tensor
    forced_removals_done: torch.Tensor
    move_count: torch.Tensor
    moves_since_capture: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.board.device

    @property
    def batch_size(self) -> int:
        return int(self.board.shape[0])

    def to(self, device: torch.device | str) -> "GpuStateBatch":
        dev = torch.device(device)
        return GpuStateBatch(
            board=self.board.to(dev),
            marks_black=self.marks_black.to(dev),
            marks_white=self.marks_white.to(dev),
            phase=self.phase.to(dev),
            current_player=self.current_player.to(dev),
            pending_marks_required=self.pending_marks_required.to(dev),
            pending_marks_remaining=self.pending_marks_remaining.to(dev),
            pending_captures_required=self.pending_captures_required.to(dev),
            pending_captures_remaining=self.pending_captures_remaining.to(dev),
            forced_removals_done=self.forced_removals_done.to(dev),
            move_count=self.move_count.to(dev),
            moves_since_capture=self.moves_since_capture.to(dev),
        )

    def slice(self, index: int) -> "GpuStateBatch":
        sl = slice(index, index + 1)
        return GpuStateBatch(
            board=self.board[sl],
            marks_black=self.marks_black[sl],
            marks_white=self.marks_white[sl],
            phase=self.phase[sl],
            current_player=self.current_player[sl],
            pending_marks_required=self.pending_marks_required[sl],
            pending_marks_remaining=self.pending_marks_remaining[sl],
            pending_captures_required=self.pending_captures_required[sl],
            pending_captures_remaining=self.pending_captures_remaining[sl],
            forced_removals_done=self.forced_removals_done[sl],
            move_count=self.move_count[sl],
            moves_since_capture=self.moves_since_capture[sl],
        )

    def select(self, indices: torch.Tensor | list[int]) -> "GpuStateBatch":
        if isinstance(indices, list):
            if not indices:
                raise ValueError("indices must not be empty.")
            idx = torch.tensor(indices, dtype=torch.int64, device=self.device)
        else:
            idx = indices.to(device=self.device, dtype=torch.int64).view(-1)
            if int(idx.numel()) == 0:
                raise ValueError("indices must not be empty.")
        return GpuStateBatch(
            board=self.board.index_select(0, idx),
            marks_black=self.marks_black.index_select(0, idx),
            marks_white=self.marks_white.index_select(0, idx),
            phase=self.phase.index_select(0, idx),
            current_player=self.current_player.index_select(0, idx),
            pending_marks_required=self.pending_marks_required.index_select(0, idx),
            pending_marks_remaining=self.pending_marks_remaining.index_select(0, idx),
            pending_captures_required=self.pending_captures_required.index_select(0, idx),
            pending_captures_remaining=self.pending_captures_remaining.index_select(0, idx),
            forced_removals_done=self.forced_removals_done.index_select(0, idx),
            move_count=self.move_count.index_select(0, idx),
            moves_since_capture=self.moves_since_capture.index_select(0, idx),
        )

    @staticmethod
    def initial(device: torch.device | str, batch_size: int = 1) -> "GpuStateBatch":
        dev = torch.device(device)
        board = torch.zeros((batch_size, 6, 6), dtype=torch.int8, device=dev)
        marks_black = torch.zeros((batch_size, 6, 6), dtype=torch.bool, device=dev)
        marks_white = torch.zeros((batch_size, 6, 6), dtype=torch.bool, device=dev)
        phase = torch.ones((batch_size,), dtype=torch.int64, device=dev)  # Phase.PLACEMENT=1
        current_player = torch.ones((batch_size,), dtype=torch.int64, device=dev)  # Player.BLACK=1
        zeros = torch.zeros((batch_size,), dtype=torch.int64, device=dev)
        return GpuStateBatch(
            board=board,
            marks_black=marks_black,
            marks_white=marks_white,
            phase=phase,
            current_player=current_player,
            pending_marks_required=zeros.clone(),
            pending_marks_remaining=zeros.clone(),
            pending_captures_required=zeros.clone(),
            pending_captures_remaining=zeros.clone(),
            forced_removals_done=zeros.clone(),
            move_count=zeros.clone(),
            moves_since_capture=zeros.clone(),
        )


def states_to_model_input(batch: GpuStateBatch) -> torch.Tensor:
    return v0_core.states_to_model_input(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
    )


def encode_actions_fast(batch: GpuStateBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    return v0_core.encode_actions_fast(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
        batch.pending_marks_required,
        batch.pending_marks_remaining,
        batch.pending_captures_required,
        batch.pending_captures_remaining,
        batch.forced_removals_done,
        PLACEMENT_DIM,
        MOVEMENT_DIM,
        SELECTION_DIM,
        AUXILIARY_DIM,
    )


def _coerce_apply_output(output: tuple) -> GpuStateBatch:
    if len(output) == 12:
        return GpuStateBatch(
            board=output[0],
            marks_black=output[1].to(torch.bool),
            marks_white=output[2].to(torch.bool),
            phase=output[3],
            current_player=output[4],
            pending_marks_required=output[5],
            pending_marks_remaining=output[6],
            pending_captures_required=output[7],
            pending_captures_remaining=output[8],
            forced_removals_done=output[9],
            move_count=output[10],
            moves_since_capture=output[11],
        )
    raise RuntimeError(f"Unexpected batch_apply_moves output arity: {len(output)}")


def batch_apply_moves_compat(
    batch: GpuStateBatch,
    action_codes: torch.Tensor,
    parent_indices: torch.Tensor,
) -> GpuStateBatch:
    """Typed wrapper over v0_core.batch_apply_moves."""
    action_codes = action_codes.to(device=batch.device, dtype=torch.int32)
    parent_indices = parent_indices.to(device=batch.device, dtype=torch.int64)
    output = v0_core.batch_apply_moves(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
        batch.pending_marks_required,
        batch.pending_marks_remaining,
        batch.pending_captures_required,
        batch.pending_captures_remaining,
        batch.forced_removals_done,
        batch.move_count,
        batch.moves_since_capture,
        action_codes,
        parent_indices,
    )
    return _coerce_apply_output(output)


@dataclass
class V1RootMCTSConfig:
    num_simulations: int = 128
    exploration_weight: float = 1.0
    temperature: float = 1.0
    add_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    sample_moves: bool = True
    autocast_dtype: str = "float16"
    child_eval_mode: str = "value_only"  # "value_only" or "full"


@dataclass
class RootSearchOutput:
    model_input: torch.Tensor
    legal_mask: torch.Tensor
    policy_dense: torch.Tensor
    root_value: float
    terminal: bool
    chosen_action_index: Optional[int]
    chosen_action_code: Optional[torch.Tensor]


@dataclass
class RootSearchBatchOutput:
    model_input: torch.Tensor
    legal_mask: torch.Tensor
    policy_dense: torch.Tensor
    root_value: torch.Tensor
    terminal_mask: torch.Tensor
    chosen_action_indices: torch.Tensor
    chosen_action_codes: torch.Tensor
    chosen_valid_mask: torch.Tensor


class V1RootMCTS:
    """GPU-first root search (PUCT over root children only)."""

    def __init__(
        self,
        model,
        config: V1RootMCTSConfig,
        device: torch.device | str,
        inference_engine=None,
        collect_timing: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.inference_engine = inference_engine
        self._collect_timing = bool(collect_timing)
        self._timing_ms: Dict[str, float] = {
            "root_puct_ms": 0.0,
            "pack_writeback_ms": 0.0,
        }
        self._timing_calls: Dict[str, int] = {
            "root_puct_ms": 0,
            "pack_writeback_ms": 0,
        }
        mode = str(self.config.child_eval_mode).strip().lower()
        if mode not in ("value_only", "full"):
            raise ValueError(
                f"Unsupported child_eval_mode={self.config.child_eval_mode!r}; "
                "expected 'value_only' or 'full'."
            )
        self._child_eval_mode = mode

    @contextmanager
    def _nvtx_range(self, name: str):
        if self.device.type != "cuda":
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    @contextmanager
    def _timed(self, name: str):
        if not self._collect_timing:
            with nullcontext():
                yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._timing_ms[name] = float(self._timing_ms.get(name, 0.0) + elapsed_ms)
            self._timing_calls[name] = int(self._timing_calls.get(name, 0) + 1)

    def get_timing(self, reset: bool = False) -> Dict[str, Dict[str, float]]:
        payload = {
            "timing_ms": {k: float(v) for k, v in self._timing_ms.items()},
            "timing_calls": {k: int(v) for k, v in self._timing_calls.items()},
        }
        if reset:
            for key in list(self._timing_ms.keys()):
                self._timing_ms[key] = 0.0
            for key in list(self._timing_calls.keys()):
                self._timing_calls[key] = 0
        return payload

    def _autocast_context(self):
        if self.device.type != "cuda":
            return torch.autocast("cpu", enabled=False)
        dtype_key = str(self.config.autocast_dtype).strip().lower()
        if dtype_key in ("bf16", "bfloat16"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _to_scalar_value(self, raw_values: torch.Tensor) -> torch.Tensor:
        if raw_values.dim() == 2 and raw_values.size(1) == 3:
            probs = torch.softmax(raw_values, dim=1)
            return probs[:, 0] - probs[:, 2]
        if raw_values.dim() == 2 and raw_values.size(1) == 1:
            return raw_values[:, 0]
        return raw_values.view(-1)

    def _forward_model(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def _forward_eager(eager_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            self.model.eval()
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with self._autocast_context():
                        return self.model(eager_inputs)
                return self.model(eager_inputs)

        def _use_eager_for_small_batch(n_valid: int, engine_bs: int) -> bool:
            # Graph path pads to engine batch size; for very small batches this wastes
            # substantial compute and can be slower than eager on RTX 3060.
            fallback_threshold = min(128, max(32, engine_bs // 2))
            return n_valid < fallback_threshold

        if self.inference_engine is not None:
            n_valid = int(inputs.size(0))
            engine_bs = int(getattr(self.inference_engine, "batch_size", n_valid))
            if engine_bs <= 0:
                raise RuntimeError("Inference engine batch_size must be positive.")
            if _use_eager_for_small_batch(n_valid, engine_bs):
                return _forward_eager(inputs)
            if n_valid <= engine_bs:
                engine_inputs = inputs
                if engine_bs > n_valid:
                    pad_shape = (
                        engine_bs - n_valid,
                        int(inputs.size(1)),
                        int(inputs.size(2)),
                        int(inputs.size(3)),
                    )
                    pad = torch.zeros(pad_shape, dtype=inputs.dtype, device=inputs.device)
                    engine_inputs = torch.cat([inputs, pad], dim=0)
                log_p1, log_p2, log_pmc, raw_values = self.inference_engine.forward(
                    engine_inputs, n_valid
                )
                return log_p1[:n_valid], log_p2[:n_valid], log_pmc[:n_valid], raw_values[:n_valid]

            log_p1_parts = []
            log_p2_parts = []
            log_pmc_parts = []
            raw_parts = []
            for start in range(0, n_valid, engine_bs):
                end = min(start + engine_bs, n_valid)
                chunk = inputs[start:end]
                chunk_valid = int(end - start)
                if chunk_valid < engine_bs and _use_eager_for_small_batch(chunk_valid, engine_bs):
                    c1, c2, c3, cv = _forward_eager(chunk)
                    log_p1_parts.append(c1[:chunk_valid])
                    log_p2_parts.append(c2[:chunk_valid])
                    log_pmc_parts.append(c3[:chunk_valid])
                    raw_parts.append(cv[:chunk_valid])
                    continue
                if chunk_valid < engine_bs:
                    pad_shape = (
                        engine_bs - chunk_valid,
                        int(chunk.size(1)),
                        int(chunk.size(2)),
                        int(chunk.size(3)),
                    )
                    pad = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
                    chunk = torch.cat([chunk, pad], dim=0)
                c1, c2, c3, cv = self.inference_engine.forward(chunk, chunk_valid)
                log_p1_parts.append(c1[:chunk_valid])
                log_p2_parts.append(c2[:chunk_valid])
                log_pmc_parts.append(c3[:chunk_valid])
                raw_parts.append(cv[:chunk_valid])
            return (
                torch.cat(log_p1_parts, dim=0),
                torch.cat(log_p2_parts, dim=0),
                torch.cat(log_pmc_parts, dim=0),
                torch.cat(raw_parts, dim=0),
            )
        return _forward_eager(inputs)

    def _evaluate_batch(
        self, batch: GpuStateBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = states_to_model_input(batch)
        log_p1, log_p2, log_pmc, raw_values = self._forward_model(inputs)
        values = self._to_scalar_value(raw_values).float()
        legal_mask, metadata = encode_actions_fast(batch)
        probs, _ = v0_core.project_policy_logits_fast(
            log_p1,
            log_p2,
            log_pmc,
            legal_mask,
            PLACEMENT_DIM,
            MOVEMENT_DIM,
            SELECTION_DIM,
            AUXILIARY_DIM,
        )
        return inputs, legal_mask, metadata, probs.float(), values

    def _evaluate_values_only(self, batch: GpuStateBatch) -> torch.Tensor:
        inputs = states_to_model_input(batch)
        _log_p1, _log_p2, _log_pmc, raw_values = self._forward_model(inputs)
        values = self._to_scalar_value(raw_values).float()
        return values

    def apply_action(self, state: GpuStateBatch, action_code: torch.Tensor) -> GpuStateBatch:
        if action_code.dim() == 1:
            action_codes = action_code.view(1, 4)
        else:
            action_codes = action_code
        num_actions = int(action_codes.shape[0])
        if state.batch_size == 1:
            parent_indices = torch.zeros((num_actions,), dtype=torch.int64, device=state.device)
        elif num_actions == int(state.batch_size):
            parent_indices = torch.arange(num_actions, dtype=torch.int64, device=state.device)
        else:
            raise ValueError(
                "action_code batch does not match state batch size: "
                f"state_batch={state.batch_size}, action_batch={num_actions}"
            )
        return batch_apply_moves_compat(state, action_codes, parent_indices)

    @staticmethod
    def _normalize_temperatures(
        temperatures: Optional[float | torch.Tensor | list],
        batch_size: int,
        default_temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        if temperatures is None:
            return torch.full((batch_size,), float(default_temperature), dtype=torch.float32, device=device)
        if isinstance(temperatures, (float, int)):
            return torch.full((batch_size,), float(temperatures), dtype=torch.float32, device=device)
        if isinstance(temperatures, list):
            if len(temperatures) != batch_size:
                raise ValueError(
                    f"temperatures length mismatch: expected {batch_size}, got {len(temperatures)}"
                )
            return torch.tensor(temperatures, dtype=torch.float32, device=device)
        t = torch.as_tensor(temperatures, dtype=torch.float32, device=device).view(-1)
        if int(t.numel()) != batch_size:
            raise ValueError(
                f"temperatures size mismatch: expected {batch_size}, got {int(t.numel())}"
            )
        return t

    def search_batch(
        self,
        state: GpuStateBatch,
        *,
        temperatures: Optional[float | torch.Tensor | list] = None,
        add_dirichlet_noise: Optional[bool] = None,
    ) -> RootSearchBatchOutput:
        batch_size = int(state.batch_size)
        add_noise = self.config.add_dirichlet_noise if add_dirichlet_noise is None else bool(
            add_dirichlet_noise
        )
        temp_values = self._normalize_temperatures(
            temperatures=temperatures,
            batch_size=batch_size,
            default_temperature=self.config.temperature,
            device=state.device,
        )

        model_input, legal_mask, metadata, probs, values = self._evaluate_batch(state)
        root_values = values.clone().to(torch.float32)
        legal_mask_bool = legal_mask.to(torch.bool)
        policy_dense = torch.zeros(
            (batch_size, TOTAL_ACTION_DIM),
            dtype=torch.float32,
            device=state.device,
        )
        terminal_mask = torch.zeros((batch_size,), dtype=torch.bool, device=state.device)
        chosen_action_indices = torch.full((batch_size,), -1, dtype=torch.int64, device=state.device)
        chosen_action_codes = torch.full((batch_size, 4), -1, dtype=torch.int32, device=state.device)
        chosen_valid_mask = torch.zeros((batch_size,), dtype=torch.bool, device=state.device)

        with self._nvtx_range("v1.root_pack_sparse_actions"), self._timed("pack_writeback_ms"):
            (
                terminal_mask,
                valid_root_indices,
                counts,
                valid_mask,
                legal_index_mat,
                priors_mat,
                action_code_mat,
                flat_indices,
                action_codes_all,
                parent_indices_all,
            ) = v0_core.root_pack_sparse_actions(
                legal_mask_bool,
                probs,
                metadata,
            )

        if int(valid_root_indices.numel()) > 0:
            num_roots = int(valid_root_indices.numel())
            max_actions = int(valid_mask.shape[1])

            if add_noise:
                if max_actions > 1:
                    eps = float(self.config.dirichlet_epsilon)
                    alpha = float(self.config.dirichlet_alpha)
                    alpha_t = torch.full_like(priors_mat, alpha, dtype=torch.float32)
                    gamma_dist = torch.distributions.Gamma(alpha_t, torch.ones_like(priors_mat))
                    noise = gamma_dist.sample() * valid_mask.to(torch.float32)
                    noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    mixed = (1.0 - eps) * priors_mat + eps * noise
                    apply_mask = counts.gt(1).view(-1, 1)
                    priors_mat = torch.where(apply_mask, mixed, priors_mat)

            child_batch = batch_apply_moves_compat(state, action_codes_all, parent_indices_all)
            if self._child_eval_mode == "full":
                _, _, _, _, child_values = self._evaluate_batch(child_batch)
            else:
                child_values = self._evaluate_values_only(child_batch)
            child_leaf_values = (-child_values).to(torch.float32)

            leaf_mat = torch.zeros(
                (num_roots, max_actions), dtype=torch.float32, device=state.device
            )
            if int(flat_indices.numel()) > 0:
                leaf_mat.view(-1).index_copy_(0, flat_indices, child_leaf_values)

            sims = max(1, int(self.config.num_simulations))
            with self._nvtx_range("v1.root_puct_allocate_visits"), self._timed("root_puct_ms"):
                visits_cpp, value_sum_cpp, _ = v0_core.root_puct_allocate_visits(
                    priors_mat,
                    leaf_mat,
                    valid_mask,
                    sims,
                    float(self.config.exploration_weight),
                )
            visits = visits_cpp.to(dtype=torch.float32, device=state.device)
            value_sum = value_sum_cpp.to(dtype=torch.float32, device=state.device)

            root_temps = temp_values.index_select(0, valid_root_indices).clamp_min(1e-6).view(-1, 1)
            legal_policy = torch.pow(visits.clamp_min(1e-8), 1.0 / root_temps)
            legal_policy = legal_policy * valid_mask.to(torch.float32)
            legal_policy = legal_policy / legal_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)

            if self.config.sample_moves and max_actions > 1:
                local_picks = torch.multinomial(legal_policy, num_samples=1).view(-1)
            else:
                local_picks = torch.argmax(legal_policy, dim=1)

            root_value_vec = value_sum.sum(dim=1) / visits.sum(dim=1).clamp_min(1.0)
            with self._nvtx_range("v1.root_sparse_writeback"), self._timed("pack_writeback_ms"):
                (
                    policy_dense,
                    chosen_action_indices,
                    chosen_action_codes,
                    chosen_valid_mask,
                ) = v0_core.root_sparse_writeback(
                    legal_index_mat,
                    action_code_mat,
                    valid_mask,
                    legal_policy,
                    local_picks,
                    valid_root_indices,
                    batch_size,
                    TOTAL_ACTION_DIM,
                )
            root_values.index_copy_(0, valid_root_indices, root_value_vec)

        return RootSearchBatchOutput(
            model_input=model_input.detach(),
            legal_mask=legal_mask_bool.detach(),
            policy_dense=policy_dense.detach(),
            root_value=root_values.detach(),
            terminal_mask=terminal_mask.detach(),
            chosen_action_indices=chosen_action_indices.detach(),
            chosen_action_codes=chosen_action_codes.detach(),
            chosen_valid_mask=chosen_valid_mask.detach(),
        )

    def search(
        self,
        state: GpuStateBatch,
        *,
        temperature: Optional[float] = None,
        add_dirichlet_noise: Optional[bool] = None,
    ) -> RootSearchOutput:
        if state.batch_size != 1:
            raise ValueError("V1RootMCTS.search currently supports a single root state.")
        batch_out = self.search_batch(
            state,
            temperatures=self.config.temperature if temperature is None else float(temperature),
            add_dirichlet_noise=add_dirichlet_noise,
        )
        terminal = bool(batch_out.terminal_mask[0].item())
        chosen_valid = bool(batch_out.chosen_valid_mask[0].item())
        chosen_index: Optional[int] = (
            int(batch_out.chosen_action_indices[0].item()) if chosen_valid else None
        )
        chosen_code: Optional[torch.Tensor] = (
            batch_out.chosen_action_codes[0].detach() if chosen_valid else None
        )
        return RootSearchOutput(
            model_input=batch_out.model_input[0].detach(),
            legal_mask=batch_out.legal_mask[0].detach(),
            policy_dense=batch_out.policy_dense[0].detach(),
            root_value=float(batch_out.root_value[0].item()),
            terminal=terminal,
            chosen_action_index=chosen_index,
            chosen_action_code=chosen_code,
        )
