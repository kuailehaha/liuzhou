"""
Neural input/output helpers for tensorized self-play.

This module will host `states_to_tensor` (batched) and utilities to project the
model logits back onto legal move masks according to the action encoding spec.
"""

from __future__ import annotations

from typing import Tuple

try:
    import v0_core  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    v0_core = None

import torch

from ..game.move_encoder import ActionEncodingSpec, DEFAULT_ACTION_SPEC, DIRECTIONS
from ..game.state_batch import TensorStateBatch
from .project_policy_logits_fast import project_policy_logits_fast

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

    if v0_core is not None and hasattr(v0_core, "states_to_model_input"):
        try:
            return v0_core.states_to_model_input(
                batch.board,
                batch.marks_black,
                batch.marks_white,
                batch.phase,
                batch.current_player,
            )
        except RuntimeError:
            pass

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


def _project_policy_logits_python(
    logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    legal_mask: torch.BoolTensor,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine raw logits with legal-action masks.

    Returns tuple `(probs, masked_logits)` for downstream sampling/training.
    """
    if len(logits) != 3:
        raise ValueError("project_policy_logits expects a tuple of (log_p1, log_p2, log_pmc).")

    log_p1, log_p2, log_pmc = logits
    if log_p1.shape != log_p2.shape or log_p1.shape != log_pmc.shape:
        raise ValueError("All policy heads must share the same shape.")

    batch_size, head_dim = log_p1.shape
    device = log_p1.device
    dtype = log_p1.dtype

    placement_dim = spec.placement_dim
    movement_dim = spec.movement_dim
    selection_dim = spec.selection_dim
    auxiliary_dim = spec.auxiliary_dim

    board_size = int(round(placement_dim ** 0.5))
    if board_size * board_size != placement_dim:
        raise ValueError("placement_dim must be a perfect square representing the board area.")

    if head_dim != placement_dim:
        raise ValueError(
            f"Policy head dimension mismatch: expected {placement_dim}, got {head_dim}."
        )
    total_dim = spec.total_dim
    if legal_mask.shape != (batch_size, total_dim):
        raise ValueError(
            f"legal_mask expected shape ({batch_size}, {total_dim}), got {tuple(legal_mask.shape)}"
        )

    neginf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    placement_slice = slice(0, placement_dim)
    movement_slice = slice(placement_slice.stop, placement_slice.stop + movement_dim)
    selection_slice = slice(movement_slice.stop, movement_slice.stop + selection_dim)
    auxiliary_slice = slice(selection_slice.stop, selection_slice.stop + auxiliary_dim)

    combined = torch.empty((batch_size, total_dim), dtype=dtype, device=device)
    combined[:, placement_slice] = log_p1
    combined[:, selection_slice] = log_pmc

    dirs = len(DIRECTIONS)
    if movement_dim != placement_dim * dirs:
        raise ValueError(
            f"movement_dim mismatch: expected {placement_dim * dirs}, got {movement_dim}."
        )

    indices = torch.arange(placement_dim, device=device)
    rows = indices // board_size
    cols = indices % board_size

    movement_logits = []
    for dr, dc in DIRECTIONS:
        dest_rows = rows + dr
        dest_cols = cols + dc
        valid = (
            (dest_rows >= 0)
            & (dest_rows < board_size)
            & (dest_cols >= 0)
            & (dest_cols < board_size)
        )
        dest_indices = dest_rows * board_size + dest_cols
        dest_indices = dest_indices.to(torch.long)

        dest_scores = torch.full((batch_size, placement_dim), neginf, dtype=dtype, device=device)
        if valid.any():
            gather_idx = dest_indices[valid].view(1, -1).expand(batch_size, -1)
            gathered = log_p1.gather(1, gather_idx)
            dest_scores[:, valid] = gathered

        movement_logits.append(log_p2 + dest_scores)

    movement_concat = torch.stack(movement_logits, dim=2).reshape(batch_size, movement_dim)
    combined[:, movement_slice] = movement_concat

    if auxiliary_dim > 0:
        aux = torch.zeros((batch_size, auxiliary_dim), dtype=dtype, device=device)
        combined[:, auxiliary_slice] = aux

    masked_logits = torch.where(legal_mask, combined, neginf)
    probs = torch.zeros_like(combined)

    for idx in range(batch_size):
        legal_indices = legal_mask[idx].nonzero(as_tuple=False).view(-1)
        if legal_indices.numel() == 0:
            continue
        row_logits = masked_logits[idx, legal_indices]
        if not torch.isfinite(row_logits).any():
            row_logits = torch.zeros_like(row_logits)
            masked_logits[idx, legal_indices] = row_logits
        if legal_indices.numel() == 1:
            probs[idx, legal_indices[0]] = 1.0
        else:
            row_probs = torch.softmax(row_logits, dim=0)
            probs[idx, legal_indices] = row_probs

    return probs, masked_logits


def project_policy_logits(
    logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    legal_mask: torch.BoolTensor,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine raw logits with legal-action masks, preferring the fast C++ path when available.

    Returns tuple `(probs, masked_logits)` for downstream sampling/training.
    """
    if len(logits) != 3:
        raise ValueError("project_policy_logits expects a tuple of (log_p1, log_p2, log_pmc).")

    log_p1, log_p2, log_pmc = logits
    try:
        fast_result = project_policy_logits_fast(log_p1, log_p2, log_pmc, legal_mask, spec)
    except RuntimeError:
        fast_result = None

    if fast_result is not None:
        return fast_result

    return _project_policy_logits_python(logits, legal_mask, spec)
