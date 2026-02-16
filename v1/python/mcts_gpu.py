"""v1 GPU-first root-search MCTS utilities.

This module keeps the hot path on GPU:
- legal-mask generation via ``v0_core.encode_actions_fast``
- policy projection via ``v0_core.project_policy_logits_fast``
- state transition via ``v0_core.batch_apply_moves``

The current implementation is a root-PUCT search. It is intentionally isolated
from v0 to allow incremental migration without changing the v0 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import v0_core
from v0.python.move_encoder import DEFAULT_ACTION_SPEC

PLACEMENT_DIM = int(DEFAULT_ACTION_SPEC.placement_dim)
MOVEMENT_DIM = int(DEFAULT_ACTION_SPEC.movement_dim)
SELECTION_DIM = int(DEFAULT_ACTION_SPEC.selection_dim)
AUXILIARY_DIM = int(DEFAULT_ACTION_SPEC.auxiliary_dim)
TOTAL_ACTION_DIM = int(DEFAULT_ACTION_SPEC.total_dim)

_BATCH_APPLY_SUPPORTS_MSC: Optional[bool] = None


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


def _coerce_apply_output(output: tuple, moves_since_capture: torch.Tensor) -> GpuStateBatch:
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
    if len(output) == 11:
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
            moves_since_capture=moves_since_capture,
        )
    raise RuntimeError(f"Unexpected batch_apply_moves output arity: {len(output)}")


def batch_apply_moves_compat(
    batch: GpuStateBatch,
    action_codes: torch.Tensor,
    parent_indices: torch.Tensor,
) -> GpuStateBatch:
    """Compatibility wrapper for different v0_core ``batch_apply_moves`` ABI."""

    global _BATCH_APPLY_SUPPORTS_MSC
    action_codes = action_codes.to(device=batch.device, dtype=torch.int32)
    parent_indices = parent_indices.to(device=batch.device, dtype=torch.int64)

    if _BATCH_APPLY_SUPPORTS_MSC is not False:
        try:
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
            _BATCH_APPLY_SUPPORTS_MSC = True
            return _coerce_apply_output(output, batch.moves_since_capture.index_select(0, parent_indices))
        except TypeError:
            _BATCH_APPLY_SUPPORTS_MSC = False
        except RuntimeError as exc:
            message = str(exc)
            if "incompatible function arguments" in message:
                _BATCH_APPLY_SUPPORTS_MSC = False
            else:
                raise

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
        action_codes,
        parent_indices,
    )
    parent_msc = batch.moves_since_capture.index_select(0, parent_indices)
    return _coerce_apply_output(output, parent_msc)


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

    def __init__(self, model, config: V1RootMCTSConfig, device: torch.device | str) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)

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

    def _evaluate_batch(
        self, batch: GpuStateBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = states_to_model_input(batch)
        self.model.eval()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with self._autocast_context():
                    log_p1, log_p2, log_pmc, raw_values = self.model(inputs)
            else:
                log_p1, log_p2, log_pmc, raw_values = self.model(inputs)
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
        return inputs, legal_mask, metadata, probs.float(), values, raw_values

    def _apply_dirichlet(self, priors: torch.Tensor) -> torch.Tensor:
        if priors.numel() <= 1:
            return priors
        alpha = max(1e-6, float(self.config.dirichlet_alpha))
        noise_dist = torch.distributions.Dirichlet(
            torch.full_like(priors, alpha, dtype=torch.float32)
        )
        noise = noise_dist.sample()
        mixed = (1.0 - float(self.config.dirichlet_epsilon)) * priors + float(
            self.config.dirichlet_epsilon
        ) * noise
        denom = mixed.sum().clamp_min(1e-8)
        return mixed / denom

    @staticmethod
    def _visits_to_policy(visits: torch.Tensor, temperature: float) -> torch.Tensor:
        if visits.numel() == 1:
            return torch.ones_like(visits)
        temp = max(float(temperature), 1e-6)
        if temp <= 1e-6:
            out = torch.zeros_like(visits)
            out[torch.argmax(visits)] = 1.0
            return out
        scaled = torch.pow(visits.clamp_min(1e-8), 1.0 / temp)
        denom = scaled.sum().clamp_min(1e-8)
        return scaled / denom

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

        model_input, legal_mask, metadata, probs, values, _raw_values = self._evaluate_batch(state)
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

        root_order: list[int] = []
        root_legal_indices: dict[int, torch.Tensor] = {}
        root_priors: dict[int, torch.Tensor] = {}
        action_chunks: list[torch.Tensor] = []
        parent_chunks: list[torch.Tensor] = []
        counts: list[int] = []

        for bi in range(batch_size):
            row_mask = legal_mask_bool[bi]
            legal_indices = torch.nonzero(row_mask, as_tuple=False).view(-1)
            if legal_indices.numel() == 0:
                terminal_mask[bi] = True
                continue
            priors = probs[bi, legal_indices].to(torch.float32)
            priors = priors / priors.sum().clamp_min(1e-8)
            if add_noise:
                priors = self._apply_dirichlet(priors)
            action_codes = metadata[bi, legal_indices].to(torch.int32)
            root_order.append(bi)
            root_legal_indices[bi] = legal_indices
            root_priors[bi] = priors
            action_chunks.append(action_codes)
            parent_chunks.append(
                torch.full((int(legal_indices.numel()),), bi, dtype=torch.int64, device=state.device)
            )
            counts.append(int(legal_indices.numel()))

        if action_chunks:
            action_codes_all = torch.cat(action_chunks, dim=0)
            parent_indices_all = torch.cat(parent_chunks, dim=0)
            child_batch = batch_apply_moves_compat(state, action_codes_all, parent_indices_all)
            _, _, _, _child_probs, child_values, _ = self._evaluate_batch(child_batch)
            child_leaf_values = (-child_values).to(torch.float32)

            num_roots = len(root_order)
            max_actions = max(counts)
            priors_mat = torch.zeros(
                (num_roots, max_actions), dtype=torch.float32, device=state.device
            )
            leaf_mat = torch.zeros(
                (num_roots, max_actions), dtype=torch.float32, device=state.device
            )
            valid_mask = torch.zeros((num_roots, max_actions), dtype=torch.bool, device=state.device)

            offset = 0
            for ridx, n in enumerate(counts):
                priors_mat[ridx, :n] = root_priors[root_order[ridx]]
                leaf_mat[ridx, :n] = child_leaf_values[offset : offset + n]
                valid_mask[ridx, :n] = True
                offset += n

            visits = torch.zeros_like(priors_mat)
            value_sum = torch.zeros_like(priors_mat)
            total_visit = torch.zeros((num_roots,), dtype=torch.float32, device=state.device)
            row_ids = torch.arange(num_roots, dtype=torch.int64, device=state.device)
            sims = max(1, int(self.config.num_simulations))

            for _ in range(sims):
                q = torch.where(
                    visits > 0,
                    value_sum / visits.clamp_min(1e-8),
                    torch.zeros_like(value_sum),
                )
                u = (
                    float(self.config.exploration_weight)
                    * priors_mat
                    * torch.sqrt(total_visit + 1.0).unsqueeze(1)
                    / (1.0 + visits)
                )
                scores = (q + u).masked_fill(~valid_mask, float("-inf"))
                selected = torch.argmax(scores, dim=1)
                visits[row_ids, selected] += 1.0
                value_sum[row_ids, selected] += leaf_mat[row_ids, selected]
                total_visit += 1.0

            root_ids = torch.tensor(root_order, dtype=torch.int64, device=state.device)
            root_temps = temp_values.index_select(0, root_ids).clamp_min(1e-6).view(-1, 1)
            legal_policy = torch.pow(visits.clamp_min(1e-8), 1.0 / root_temps)
            legal_policy = legal_policy * valid_mask.to(torch.float32)
            legal_policy = legal_policy / legal_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)

            if self.config.sample_moves and max_actions > 1:
                local_picks = torch.multinomial(legal_policy, num_samples=1).view(-1)
            else:
                local_picks = torch.argmax(legal_policy, dim=1)

            root_value_vec = value_sum.sum(dim=1) / visits.sum(dim=1).clamp_min(1.0)
            for ridx, bi in enumerate(root_order):
                legal_indices = root_legal_indices[bi]
                n = counts[ridx]
                policy_legal = legal_policy[ridx, :n]
                policy_dense[bi, legal_indices] = policy_legal
                local_pick_int = int(local_picks[ridx].item())
                chosen_action_indices[bi] = int(legal_indices[local_pick_int].item())
                chosen_action_codes[bi] = action_chunks[ridx][local_pick_int]
                chosen_valid_mask[bi] = True
                root_values[bi] = root_value_vec[ridx]

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
