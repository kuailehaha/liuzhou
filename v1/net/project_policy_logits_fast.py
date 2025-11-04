"""
Optional fast implementation of `project_policy_logits` backed by a Torch C++ extension.

Falls back to ``None`` when the extension cannot be loaded or the inputs are on
unsupported devices, allowing the Python implementation to continue serving as a
reference path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

from ..game.move_encoder import ActionEncodingSpec


@lru_cache()
def _load_extension():
    src_path = os.path.join(os.path.dirname(__file__), "project_policy_logits_fast.cpp")
    try:
        return load(
            name="project_policy_logits_ext",
            sources=[src_path],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            verbose=False,
        )
    except Exception:
        return None


def project_policy_logits_fast(
    log_p1: torch.Tensor,
    log_p2: torch.Tensor,
    log_pmc: torch.Tensor,
    legal_mask: torch.BoolTensor,
    spec: ActionEncodingSpec,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Attempt to fuse policy projection and masked softmax in C++.

    Returns ``None`` when the extension is unavailable or when tensors are on
    unsupported devices, signalling the caller to fall back to the Python path.
    """
    ext = _load_extension()
    if ext is None:
        return None

    device = log_p1.device

    if any(t.device != device for t in (log_p2, log_pmc, legal_mask)):
        return None

    if not legal_mask.dtype == torch.bool:
        return None

    return ext.project_policy_logits_fast(
        log_p1,
        log_p2,
        log_pmc,
        legal_mask,
        spec.placement_dim,
        spec.movement_dim,
        spec.selection_dim,
        spec.auxiliary_dim,
    )

