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
        parent_indices = torch.zeros((action_codes.shape[0],), dtype=torch.int64, device=state.device)
        return batch_apply_moves_compat(state, action_codes, parent_indices)

    def search(
        self,
        state: GpuStateBatch,
        *,
        temperature: Optional[float] = None,
        add_dirichlet_noise: Optional[bool] = None,
    ) -> RootSearchOutput:
        if state.batch_size != 1:
            raise ValueError("V1RootMCTS.search currently supports a single root state.")

        temp = self.config.temperature if temperature is None else float(temperature)
        add_noise = self.config.add_dirichlet_noise if add_dirichlet_noise is None else bool(
            add_dirichlet_noise
        )

        model_input, legal_mask, metadata, probs, values, _raw_values = self._evaluate_batch(state)
        root_mask = legal_mask[0].to(torch.bool)
        legal_indices = torch.nonzero(root_mask, as_tuple=False).view(-1)
        root_value_eval = float(values[0].item())

        if legal_indices.numel() == 0:
            return RootSearchOutput(
                model_input=model_input[0].detach(),
                legal_mask=root_mask.detach(),
                policy_dense=torch.zeros((TOTAL_ACTION_DIM,), dtype=torch.float32, device=state.device),
                root_value=root_value_eval,
                terminal=True,
                chosen_action_index=None,
                chosen_action_code=None,
            )

        action_codes = metadata[0, legal_indices].to(torch.int32)
        parent_indices = torch.zeros((legal_indices.numel(),), dtype=torch.int64, device=state.device)
        child_batch = batch_apply_moves_compat(state, action_codes, parent_indices)

        _, _, _, _child_probs, child_values, _ = self._evaluate_batch(child_batch)
        leaf_values = (-child_values).float()

        priors = probs[0, legal_indices].float()
        priors = priors / priors.sum().clamp_min(1e-8)
        if add_noise:
            priors = self._apply_dirichlet(priors)

        visits = torch.zeros_like(priors, dtype=torch.float32)
        value_sum = torch.zeros_like(priors, dtype=torch.float32)
        total_visit = torch.tensor(0.0, dtype=torch.float32, device=state.device)
        sims = max(1, int(self.config.num_simulations))

        for _ in range(sims):
            q = torch.where(visits > 0, value_sum / visits.clamp_min(1e-8), torch.zeros_like(value_sum))
            u = (
                float(self.config.exploration_weight)
                * priors
                * torch.sqrt(total_visit + 1.0)
                / (1.0 + visits)
            )
            selected = torch.argmax(q + u)
            visits[selected] += 1.0
            value_sum[selected] += leaf_values[selected]
            total_visit += 1.0

        policy_legal = self._visits_to_policy(visits, temp)
        dense_policy = torch.zeros((TOTAL_ACTION_DIM,), dtype=torch.float32, device=state.device)
        dense_policy[legal_indices] = policy_legal

        if self.config.sample_moves and policy_legal.numel() > 1:
            local_pick = torch.multinomial(policy_legal, num_samples=1).view(())
        else:
            local_pick = torch.argmax(policy_legal).view(())

        local_pick_int = int(local_pick.item())
        chosen_action_index = int(legal_indices[local_pick_int].item())
        chosen_action_code = action_codes[local_pick_int].detach()
        mean_root = value_sum.sum() / visits.sum().clamp_min(1.0)

        return RootSearchOutput(
            model_input=model_input[0].detach(),
            legal_mask=root_mask.detach(),
            policy_dense=dense_policy.detach(),
            root_value=float(mean_root.item()),
            terminal=False,
            chosen_action_index=chosen_action_index,
            chosen_action_code=chosen_action_code,
        )

