"""
Utility helpers for tensor manipulation within the v1 pipeline.

The functions defined here are placeholders that outline the expected API for
batched state handling. Replace `NotImplementedError` stubs as the underlying
implementations are developed.
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
    Keeping a central helper avoids littering the codebase with repetitive `to(...)`.
    """
    if device is None or tensor.device == device:
        return tensor
    return tensor.to(device)


def batched_index_select(batch_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for a batched gather helper.
    Implement using torch.take_along_dim or advanced indexing once the action
    encoding is finalized.
    """
    raise NotImplementedError("batched_index_select will be implemented with the action encoding.")

