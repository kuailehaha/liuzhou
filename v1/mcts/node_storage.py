"""
Placeholder for tensorized tree node storage.

Holds arrays for visit counts, value sums, prior probabilities, and tree edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NodeStorage:
    """
    Structure for managing batched tree data.
    """

    visit_counts: torch.Tensor
    value_sums: torch.Tensor
    priors: torch.Tensor
    children_index: torch.Tensor
    children_mask: torch.BoolTensor

    @classmethod
    def allocate(cls, capacity: int, action_dim: int, device: Optional[torch.device] = None) -> "NodeStorage":
        """
        Reserve tensors for a given node capacity.
        """
        device = device or torch.device("cpu")
        return cls(
            visit_counts=torch.zeros(capacity, device=device),
            value_sums=torch.zeros(capacity, device=device),
            priors=torch.zeros(capacity, action_dim, device=device),
            children_index=torch.full((capacity, action_dim), -1, device=device, dtype=torch.long),
            children_mask=torch.zeros(capacity, action_dim, device=device, dtype=torch.bool),
        )

