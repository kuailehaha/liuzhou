"""Wrapper around v0_core.encode_actions_fast to match the legacy API."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

import v0_core


def encode_actions_fast(
    batch,
    spec,
    *,
    return_metadata: bool = False,
) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    if batch.board.device.type != "cpu":
        board = batch.board.to("cpu")
        marks_black = batch.marks_black.to("cpu")
        marks_white = batch.marks_white.to("cpu")
        phase = batch.phase.to("cpu")
        current_player = batch.current_player.to("cpu")
        pending_marks_required = batch.pending_marks_required.to("cpu")
        pending_marks_remaining = batch.pending_marks_remaining.to("cpu")
        pending_captures_required = batch.pending_captures_required.to("cpu")
        pending_captures_remaining = batch.pending_captures_remaining.to("cpu")
        forced_removals_done = batch.forced_removals_done.to("cpu")
    else:
        board = batch.board
        marks_black = batch.marks_black
        marks_white = batch.marks_white
        phase = batch.phase
        current_player = batch.current_player
        pending_marks_required = batch.pending_marks_required
        pending_marks_remaining = batch.pending_marks_remaining
        pending_captures_required = batch.pending_captures_required
        pending_captures_remaining = batch.pending_captures_remaining
        forced_removals_done = batch.forced_removals_done

    result = v0_core.encode_actions_fast(
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
        spec.placement_dim,
        spec.movement_dim,
        spec.selection_dim,
        spec.auxiliary_dim,
    )
    if result is None:
        return None
    if return_metadata:
        return result
    return result[0]
