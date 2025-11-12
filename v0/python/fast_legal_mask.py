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
    tensors = (
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
    )

    def _invoke(args):
        return v0_core.encode_actions_fast(
            *args,
            spec.placement_dim,
            spec.movement_dim,
            spec.selection_dim,
            spec.auxiliary_dim,
        )

    result = None
    tried_cuda = False
    if tensors[0].device.type == "cuda":
        tried_cuda = True
        try:
            result = _invoke(tensors)
        except RuntimeError as exc:  # CUDA kernels missing -> fall back to CPU
            if "CUDA kernels were not built" not in str(exc):
                raise

    if result is None:
        cpu_tensors = tuple(t.to("cpu") for t in tensors) if tried_cuda else tensors
        result = _invoke(cpu_tensors)

    if result is None:
        return None
    if return_metadata:
        return result
    return result[0]
