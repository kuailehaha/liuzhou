"""
Optional fast C++ implementation of applying encoded actions to tensorized states.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

from ..common.tensor_utils import TensorGameConfig
from .state_batch import TensorStateBatch


@lru_cache()
def _load_extension():
    src_path = os.path.join(os.path.dirname(__file__), "fast_apply_moves.cpp")
    try:
        return load(
            name="fast_apply_moves_ext",
            sources=[src_path],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            verbose=False,
        )
    except Exception:
        return None


def batch_apply_moves_fast(
    batch: TensorStateBatch,
    parent_indices: torch.LongTensor,
    action_codes: torch.IntTensor,
) -> Optional[TensorStateBatch]:
    """
    Apply the encoded actions (metadata) to the provided batch of states.

    Parameters
    ----------
    batch : TensorStateBatch
        Source states (must reside on CPU for the fast path).
    parent_indices : torch.LongTensor
        Shape (N,) parent row for each action.
    action_codes : torch.IntTensor
        Shape (N, 4) encoded metadata from the legal mask extension.

    Returns
    -------
    TensorStateBatch | None
        Batched next states (one per action) on CPU, or ``None`` when the
        extension is unavailable or the inputs are on unsupported devices.
    """
    ext = _load_extension()
    if ext is None:
        return None

    if batch.board.device.type != "cpu":
        return None

    parent_indices = parent_indices.to("cpu", dtype=torch.long, copy=False)
    action_codes = action_codes.to("cpu", dtype=torch.int32, copy=False)

    result = ext.batch_apply_moves(
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

    (
        board,
        marks_black,
        marks_white,
        phase,
        current_player,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
        move_count,
    ) = result

    mask_alive = torch.ones(board.size(0), dtype=torch.bool, device=board.device)
    config = TensorGameConfig(board_size=batch.config.board_size, device=board.device)
    return TensorStateBatch(
        board=board,
        marks_black=marks_black,
        marks_white=marks_white,
        phase=phase,
        current_player=current_player,
        pending_marks_required=pending_marks_required,
        pending_marks_remaining=pending_marks_remaining,
        pending_captures_required=pending_captures_required,
        pending_captures_remaining=pending_captures_remaining,
        forced_removals_done=forced_removals_done,
        move_count=move_count,
        mask_alive=mask_alive,
        config=config,
    )
