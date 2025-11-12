"""
Utility helpers for tensor manipulation within the v0 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TensorGameConfig:
    """Basic configuration for tensorized state tensors."""

    board_size: int = 6
    device: torch.device = torch.device("cpu")


def ensure_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Move tensor to the requested device if necessary.
    """
    if device is None or tensor.device == device:
        return tensor
    return tensor.to(device)


def batched_index_select(batch_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Select per-batch entries along the first feature dimension (dim=1).
    """
    if batch_tensor.dim() < 2:
        raise ValueError("batched_index_select expects `batch_tensor` to have at least two dimensions.")
    if indices.dim() == 0:
        raise ValueError("indices tensor must have at least one dimension.")
    if batch_tensor.size(0) != indices.size(0):
        raise ValueError(
            f"Batch dimension mismatch: tensor batch={batch_tensor.size(0)} indices batch={indices.size(0)}"
        )

    dim = 1
    original_index_dim = indices.dim()

    work_indices = indices
    while work_indices.dim() < batch_tensor.dim():
        work_indices = work_indices.unsqueeze(-1)

    expand_shape = list(work_indices.shape)
    expand_shape[dim + 1 :] = list(batch_tensor.shape[dim + 1 :])
    expanded_indices = work_indices.expand(*expand_shape)

    gathered = torch.take_along_dim(batch_tensor, expanded_indices, dim=dim)

    if original_index_dim == 1:
        gathered = gathered.squeeze(dim)
    return gathered
