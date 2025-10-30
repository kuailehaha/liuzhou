"""
Dataset utilities for tensorized rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..self_play.samples import RolloutTensorBatch


@dataclass
class TensorRolloutDataset(Dataset):
    """
    Thin Dataset wrapper around precomputed rollout tensors.
    """

    states: torch.Tensor
    policies: torch.Tensor
    values: torch.Tensor
    soft_values: torch.Tensor

    @classmethod
    def from_batch(cls, batch: RolloutTensorBatch) -> "TensorRolloutDataset":
        return cls(batch.states, batch.policies, batch.values, batch.soft_values)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.policies[idx],
            self.values[idx],
            self.soft_values[idx],
        )

