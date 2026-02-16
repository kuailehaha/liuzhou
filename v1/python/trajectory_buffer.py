"""Tensor-native trajectory buffer for v1 GPU self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    """Collect per-step trajectories and finalize value targets by game outcome."""

    def __init__(self, device: torch.device | str, action_dim: int) -> None:
        self.device = torch.device(device)
        self.action_dim = int(action_dim)

        self._state_tensors: List[torch.Tensor] = []
        self._legal_masks: List[torch.Tensor] = []
        self._policy_targets: List[torch.Tensor] = []
        self._value_targets: List[torch.Tensor] = []
        self._soft_value_targets: List[torch.Tensor] = []
        self._player_signs: List[int] = []

    def append_step(
        self,
        model_input: torch.Tensor,
        legal_mask: torch.Tensor,
        policy_dense: torch.Tensor,
        player_sign: int,
    ) -> int:
        if model_input.dim() != 3:
            raise ValueError(f"model_input must be (C,H,W), got shape {tuple(model_input.shape)}")
        if legal_mask.dim() != 1 or int(legal_mask.shape[0]) != self.action_dim:
            raise ValueError(
                f"legal_mask must be ({self.action_dim},), got shape {tuple(legal_mask.shape)}"
            )
        if policy_dense.dim() != 1 or int(policy_dense.shape[0]) != self.action_dim:
            raise ValueError(
                f"policy_dense must be ({self.action_dim},), got shape {tuple(policy_dense.shape)}"
            )

        self._state_tensors.append(model_input.detach().to(self.device, non_blocking=True))
        self._legal_masks.append(legal_mask.detach().to(self.device, non_blocking=True).to(torch.bool))
        self._policy_targets.append(policy_dense.detach().to(self.device, non_blocking=True).to(torch.float32))
        self._value_targets.append(torch.tensor(float("nan"), dtype=torch.float32, device=self.device))
        self._soft_value_targets.append(torch.tensor(float("nan"), dtype=torch.float32, device=self.device))
        self._player_signs.append(1 if int(player_sign) >= 0 else -1)
        return len(self._state_tensors) - 1

    def finalize_game(self, step_indices: List[int], result_from_black: float, soft_value_from_black: float) -> None:
        game_result = float(result_from_black)
        soft_result = float(soft_value_from_black)
        for idx in step_indices:
            sign = float(self._player_signs[idx])
            self._value_targets[idx] = torch.tensor(sign * game_result, dtype=torch.float32, device=self.device)
            self._soft_value_targets[idx] = torch.tensor(
                sign * soft_result, dtype=torch.float32, device=self.device
            )

    def build(self) -> TensorSelfPlayBatch:
        if not self._state_tensors:
            return TensorSelfPlayBatch(
                state_tensors=torch.empty((0, 11, 6, 6), dtype=torch.float32, device=self.device),
                legal_masks=torch.empty((0, self.action_dim), dtype=torch.bool, device=self.device),
                policy_targets=torch.empty((0, self.action_dim), dtype=torch.float32, device=self.device),
                value_targets=torch.empty((0,), dtype=torch.float32, device=self.device),
                soft_value_targets=torch.empty((0,), dtype=torch.float32, device=self.device),
            )

        states = torch.stack(self._state_tensors, dim=0).to(torch.float32)
        legal_masks = torch.stack(self._legal_masks, dim=0).to(torch.bool)
        policy_targets = torch.stack(self._policy_targets, dim=0).to(torch.float32)
        value_targets = torch.stack(self._value_targets, dim=0).to(torch.float32)
        soft_targets = torch.stack(self._soft_value_targets, dim=0).to(torch.float32)
        return TensorSelfPlayBatch(
            state_tensors=states,
            legal_masks=legal_masks,
            policy_targets=policy_targets,
            value_targets=value_targets,
            soft_value_targets=soft_targets,
        )

