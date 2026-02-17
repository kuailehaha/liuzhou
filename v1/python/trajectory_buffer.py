"""Tensor-native trajectory buffer for v1 GPU self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TensorSelfPlayBatch:
    """Training-ready tensor batch produced by v1 self-play."""

    state_tensors: torch.Tensor
    legal_masks: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    soft_value_targets: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.state_tensors.shape[0])

    def to(self, device: torch.device | str) -> "TensorSelfPlayBatch":
        dev = torch.device(device)
        return TensorSelfPlayBatch(
            state_tensors=self.state_tensors.to(dev),
            legal_masks=self.legal_masks.to(dev),
            policy_targets=self.policy_targets.to(dev),
            value_targets=self.value_targets.to(dev),
            soft_value_targets=self.soft_value_targets.to(dev),
        )


class TensorTrajectoryBuffer:
    """Preallocated tensor arena for v1 trajectories."""

    def __init__(
        self,
        device: torch.device | str,
        action_dim: int,
        *,
        max_steps_hint: int = 512,
        concurrent_games_hint: int = 8,
        initial_capacity: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        hint_capacity = max(1, int(max_steps_hint) * int(concurrent_games_hint))
        self._capacity = int(max(initial_capacity or 0, hint_capacity))
        self._size = 0
        self._state_shape: Optional[tuple[int, int, int]] = None

        self._state_tensors: Optional[torch.Tensor] = None
        self._legal_masks: Optional[torch.Tensor] = None
        self._policy_targets: Optional[torch.Tensor] = None
        self._value_targets: Optional[torch.Tensor] = None
        self._soft_value_targets: Optional[torch.Tensor] = None
        self._player_signs: Optional[torch.Tensor] = None

    def _allocate(self, capacity: int, state_shape: tuple[int, int, int]) -> None:
        self._state_shape = state_shape
        self._state_tensors = torch.empty(
            (capacity, state_shape[0], state_shape[1], state_shape[2]),
            dtype=torch.float32,
            device=self.device,
        )
        self._legal_masks = torch.empty((capacity, self.action_dim), dtype=torch.bool, device=self.device)
        self._policy_targets = torch.empty((capacity, self.action_dim), dtype=torch.float32, device=self.device)
        self._value_targets = torch.full((capacity,), float("nan"), dtype=torch.float32, device=self.device)
        self._soft_value_targets = torch.full((capacity,), float("nan"), dtype=torch.float32, device=self.device)
        self._player_signs = torch.empty((capacity,), dtype=torch.int8, device=self.device)
        self._capacity = int(capacity)

    def _ensure_storage(self, state_shape: tuple[int, int, int]) -> None:
        if self._state_tensors is None:
            self._allocate(self._capacity, state_shape)
            return
        if self._state_shape != state_shape:
            raise ValueError(
                f"Inconsistent state shape: expected {self._state_shape}, got {state_shape}"
            )

    def _grow(self, required: int) -> None:
        if self._state_tensors is None or self._state_shape is None:
            raise RuntimeError("Storage must be initialized before grow.")
        new_capacity = max(int(required), int(max(1, self._capacity) * 2))
        old_size = int(self._size)
        old_state = self._state_tensors
        old_legal = self._legal_masks
        old_policy = self._policy_targets
        old_value = self._value_targets
        old_soft = self._soft_value_targets
        old_sign = self._player_signs
        self._allocate(new_capacity, self._state_shape)
        self._state_tensors[:old_size].copy_(old_state[:old_size])
        self._legal_masks[:old_size].copy_(old_legal[:old_size])
        self._policy_targets[:old_size].copy_(old_policy[:old_size])
        self._value_targets[:old_size].copy_(old_value[:old_size])
        self._soft_value_targets[:old_size].copy_(old_soft[:old_size])
        self._player_signs[:old_size].copy_(old_sign[:old_size])

    def append_step(
        self,
        model_input: torch.Tensor,
        legal_mask: torch.Tensor,
        policy_dense: torch.Tensor,
        player_sign: int,
    ) -> int:
        idx = self.append_steps(
            model_input=model_input.unsqueeze(0),
            legal_mask=legal_mask.unsqueeze(0),
            policy_dense=policy_dense.unsqueeze(0),
            player_sign=torch.tensor([int(player_sign)], dtype=torch.int64, device=model_input.device),
        )
        return int(idx[0].item())

    def append_steps(
        self,
        model_input: torch.Tensor,
        legal_mask: torch.Tensor,
        policy_dense: torch.Tensor,
        player_sign: torch.Tensor,
    ) -> torch.Tensor:
        if model_input.dim() != 4:
            raise ValueError(f"model_input must be (N,C,H,W), got shape {tuple(model_input.shape)}")
        batch_n = int(model_input.shape[0])
        if legal_mask.dim() != 2 or int(legal_mask.shape[0]) != batch_n or int(legal_mask.shape[1]) != self.action_dim:
            raise ValueError(
                f"legal_mask must be (N,{self.action_dim}), got shape {tuple(legal_mask.shape)}"
            )
        if policy_dense.dim() != 2 or int(policy_dense.shape[0]) != batch_n or int(policy_dense.shape[1]) != self.action_dim:
            raise ValueError(
                f"policy_dense must be (N,{self.action_dim}), got shape {tuple(policy_dense.shape)}"
            )

        sign_t = torch.as_tensor(player_sign, device=model_input.device).view(-1)
        if int(sign_t.numel()) != batch_n:
            raise ValueError(f"player_sign must have {batch_n} elements, got {int(sign_t.numel())}")

        states = model_input.detach().to(self.device, non_blocking=True).to(torch.float32)
        legal = legal_mask.detach().to(self.device, non_blocking=True).to(torch.bool)
        policy = policy_dense.detach().to(self.device, non_blocking=True).to(torch.float32)
        signs = torch.where(sign_t >= 0, 1, -1).to(device=self.device, dtype=torch.int8)

        state_shape = (int(states.shape[1]), int(states.shape[2]), int(states.shape[3]))
        self._ensure_storage(state_shape)
        required = int(self._size + batch_n)
        if required > self._capacity:
            self._grow(required)

        start = int(self._size)
        end = int(required)
        self._state_tensors[start:end].copy_(states)
        self._legal_masks[start:end].copy_(legal)
        self._policy_targets[start:end].copy_(policy)
        self._value_targets[start:end].fill_(float("nan"))
        self._soft_value_targets[start:end].fill_(float("nan"))
        self._player_signs[start:end].copy_(signs)
        self._size = end
        return torch.arange(start, end, dtype=torch.int64, device=self.device)

    def finalize_games(
        self,
        *,
        step_index_matrix: torch.Tensor,
        step_counts: torch.Tensor,
        slots: torch.Tensor,
        result_from_black: torch.Tensor,
        soft_value_from_black: torch.Tensor,
    ) -> None:
        if self._size == 0:
            return
        slots_t = torch.as_tensor(slots, dtype=torch.int64, device=self.device).view(-1)
        if int(slots_t.numel()) == 0:
            return
        counts = step_counts.to(device=self.device, dtype=torch.int64).index_select(0, slots_t)
        has_steps = counts > 0
        if not bool(has_steps.any().item()):
            return

        slots_t = slots_t[has_steps]
        counts = counts[has_steps]
        result = torch.as_tensor(result_from_black, dtype=torch.float32, device=self.device).view(-1)[has_steps]
        soft = torch.as_tensor(soft_value_from_black, dtype=torch.float32, device=self.device).view(-1)[has_steps]
        if int(slots_t.numel()) == 0:
            return

        max_len = int(counts.max().item())
        if max_len <= 0:
            return

        step_rows = step_index_matrix.to(device=self.device, dtype=torch.int64).index_select(0, slots_t)[:, :max_len]
        pos = torch.arange(max_len, device=self.device, dtype=torch.int64).view(1, -1)
        valid = pos < counts.view(-1, 1)
        flat_idx = step_rows[valid]
        if int(flat_idx.numel()) == 0:
            return

        row_ids = torch.arange(int(slots_t.numel()), device=self.device, dtype=torch.int64).view(-1, 1)
        row_ids = row_ids.expand(-1, max_len)[valid]
        per_step_result = result.index_select(0, row_ids)
        per_step_soft = soft.index_select(0, row_ids)
        signs = self._player_signs.index_select(0, flat_idx).to(torch.float32)
        self._value_targets.index_copy_(0, flat_idx, signs * per_step_result)
        self._soft_value_targets.index_copy_(0, flat_idx, signs * per_step_soft)

    def finalize_games_inplace(
        self,
        *,
        step_index_matrix: torch.Tensor,
        step_counts: torch.Tensor,
        slots: torch.Tensor,
        result_from_black: torch.Tensor,
        soft_value_from_black: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finalize trajectories via v0_core op and return slots/lengths/outcome counts."""
        if self._size == 0:
            empty = torch.empty((0,), dtype=torch.int64, device=self.device)
            counts = torch.zeros((3,), dtype=torch.int64, device=self.device)
            return empty, empty, counts
        if self._value_targets is None or self._soft_value_targets is None or self._player_signs is None:
            raise RuntimeError("Trajectory storage is not initialized.")
        import v0_core

        return v0_core.finalize_trajectory_inplace(
            self._value_targets,
            self._soft_value_targets,
            self._player_signs,
            step_index_matrix,
            step_counts,
            slots,
            result_from_black,
            soft_value_from_black,
        )

    def finalize_game(self, step_indices: list[int], result_from_black: float, soft_value_from_black: float) -> None:
        if not step_indices:
            return
        idx = torch.tensor(step_indices, dtype=torch.int64, device=self.device)
        signs = self._player_signs.index_select(0, idx).to(torch.float32)
        value = signs * float(result_from_black)
        soft = signs * float(soft_value_from_black)
        self._value_targets.index_copy_(0, idx, value)
        self._soft_value_targets.index_copy_(0, idx, soft)

    def build(self) -> TensorSelfPlayBatch:
        if self._size == 0:
            state_shape = self._state_shape if self._state_shape is not None else (11, 6, 6)
            return TensorSelfPlayBatch(
                state_tensors=torch.empty((0, state_shape[0], state_shape[1], state_shape[2]), dtype=torch.float32, device=self.device),
                legal_masks=torch.empty((0, self.action_dim), dtype=torch.bool, device=self.device),
                policy_targets=torch.empty((0, self.action_dim), dtype=torch.float32, device=self.device),
                value_targets=torch.empty((0,), dtype=torch.float32, device=self.device),
                soft_value_targets=torch.empty((0,), dtype=torch.float32, device=self.device),
            )

        end = int(self._size)
        return TensorSelfPlayBatch(
            state_tensors=self._state_tensors[:end].clone(),
            legal_masks=self._legal_masks[:end].clone(),
            policy_targets=self._policy_targets[:end].clone(),
            value_targets=self._value_targets[:end].clone(),
            soft_value_targets=self._soft_value_targets[:end].clone(),
        )
