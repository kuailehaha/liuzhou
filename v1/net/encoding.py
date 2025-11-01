"""
Neural input/output helpers for tensorized self-play.

This module will host `states_to_tensor` (batched) and utilities to project the
model logits back onto legal move masks according to the action encoding spec.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..game.move_encoder import ActionEncodingSpec, DEFAULT_ACTION_SPEC
from ..game.state_batch import TensorStateBatch

_PHASE_ORDER = (
    1,  # PLACEMENT
    2,  # MARK_SELECTION
    3,  # REMOVAL
    4,  # MOVEMENT
    5,  # CAPTURE_SELECTION
    6,  # FORCED_REMOVAL
    7,  # COUNTER_REMOVAL
)
_NUM_PHASE_CHANNELS = len(_PHASE_ORDER)
_NUM_BOARD_CHANNELS = 2
_NUM_MARK_CHANNELS = 2
_TOTAL_CHANNELS = _NUM_BOARD_CHANNELS + _NUM_MARK_CHANNELS + _NUM_PHASE_CHANNELS


def states_to_model_input(batch: TensorStateBatch) -> torch.Tensor:
    """
    Convert a `TensorStateBatch` into model-ready tensors (B, C, H, W).
    """
    if not isinstance(batch, TensorStateBatch):
        raise TypeError(f"states_to_model_input expects TensorStateBatch, got {type(batch)!r}")

    board = batch.board
    if board.dim() != 3:
        raise ValueError(f"Expected board tensor of shape (B, H, W), got {tuple(board.shape)}")

    batch_size, height, width = board.shape
    device = board.device
    dtype = torch.float32

    board_planes = _board_planes(batch, dtype)
    mark_planes = _mark_planes(batch, dtype)
    phase_planes = _phase_planes(batch, dtype, height, width)

    stacked = torch.cat((board_planes, mark_planes, phase_planes), dim=1)
    if stacked.shape[1] != _TOTAL_CHANNELS:
        raise RuntimeError(
            f"states_to_model_input produced {stacked.shape[1]} channels; expected {_TOTAL_CHANNELS}"
        )
    return stacked.contiguous()


def _board_planes(batch: TensorStateBatch, dtype: torch.dtype) -> torch.Tensor:
    board = batch.board
    current = batch.current_player.view(-1, 1, 1).to(board.dtype)

    self_mask = (board == current).to(dtype)
    opp_mask = (board == -current).to(dtype)

    return torch.stack((self_mask, opp_mask), dim=1)


def _mark_planes(batch: TensorStateBatch, dtype: torch.dtype) -> torch.Tensor:
    current_is_black = (batch.current_player == 1).view(-1, 1, 1)

    self_marks = torch.where(current_is_black, batch.marks_black, batch.marks_white).to(dtype)
    opp_marks = torch.where(current_is_black, batch.marks_white, batch.marks_black).to(dtype)

    return torch.stack((self_marks, opp_marks), dim=1)


def _phase_planes(
    batch: TensorStateBatch,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    device = batch.phase.device
    phase_ids = torch.tensor(_PHASE_ORDER, dtype=batch.phase.dtype, device=device)
    matches = batch.phase.view(-1, 1) == phase_ids.view(1, -1)
    phase_one_hot = matches.to(dtype).view(-1, _NUM_PHASE_CHANNELS, 1, 1)
    return phase_one_hot.expand(-1, -1, height, width)


def project_policy_logits(
    logits: torch.Tensor,
    legal_mask: torch.BoolTensor,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine raw logits with legal-action masks.

    Returns tuple `(probs, masked_logits)` for downstream sampling/training.
    """
    raise NotImplementedError("project_policy_logits requires action encoding support.")
