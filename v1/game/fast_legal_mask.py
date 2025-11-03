"""
Optional fast legal-action mask encoder backed by a Torch C++ extension.

Falls back to returning ``None`` when the extension is unavailable or the
current device is unsupported, allowing callers to keep using the Python
implementation transparently.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from torch.utils.cpp_extension import load


@lru_cache()
def _load_extension():
    src_path = os.path.join(os.path.dirname(__file__), "fast_legal_mask.cpp")
    try:
        return load(
            name="fast_legal_mask_ext",
            sources=[src_path],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            verbose=False,
        )
    except Exception:
        return None


def encode_actions_fast(batch, spec) -> Optional[torch.Tensor]:
    """
    Attempt to build legal masks using the C++ extension.

    Returns
    -------
    torch.Tensor | None
        When successful, a boolean tensor of shape ``(B, spec.total_dim)`` on CPU.
        Returns ``None`` if the extension is unavailable or the batch resides on
        an unsupported device.
    """
    ext = _load_extension()
    if ext is None:
        return None

    if batch.board.device.type != "cpu":
        return None

    return ext.encode_actions_fast(
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
        spec.placement_dim,
        spec.movement_dim,
        spec.selection_dim,
        spec.auxiliary_dim,
    )

