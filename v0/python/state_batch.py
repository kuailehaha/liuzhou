from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

import v0_core as core
from src.game_state import GameState


def _device_to_string(device: torch.device | str | None) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return str(device)
    return str(torch.device(device))


def from_game_states(
    states: Sequence[GameState],
    device: torch.device | str | None = None,
) -> core.TensorStateBatch:
    """
    Create a C++ TensorStateBatch from Python GameState objects.
    """
    if not isinstance(states, Iterable):
        raise TypeError("states must be an iterable of GameState objects.")
    device_str = _device_to_string(device)
    return core.tensor_batch_from_game_states(list(states), device_str)


def to_game_states(batch: core.TensorStateBatch) -> List[GameState]:
    """
    Convert a C++ TensorStateBatch back into Python GameState objects.
    """
    if not isinstance(batch, core.TensorStateBatch):
        raise TypeError("batch must be a TensorStateBatch created by v0_core.")
    return core.tensor_batch_to_game_states(batch)
