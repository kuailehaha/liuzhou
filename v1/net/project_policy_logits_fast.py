"""
Optional fast implementation of `project_policy_logits` backed by a Torch C++ extension.

Falls back to ``None`` when the extension cannot be loaded or the inputs are on
unsupported devices, allowing the Python implementation to continue serving as a
reference path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.utils.cpp_extension as cpp
from torch.utils.cpp_extension import load

from ..game.move_encoder import ActionEncodingSpec


@lru_cache()
def _load_extension():
    src_path = os.path.join(os.path.dirname(__file__), "project_policy_logits_fast.cpp")
    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]

    def _compile() -> Optional[object]:
        return load(
            name="project_policy_logits_ext",
            sources=[src_path],
            extra_cflags=extra_cflags,
            verbose=False,
        )

    try:
        return _compile()
    except Exception as exc:
        message = str(exc)
        if "Ninja is required" in message:
            try:
                cpp.USE_NINJA = False
            except AttributeError:
                pass
            try:
                return _compile()
            except Exception:
                return None
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


def ensure_project_policy_logits_fast(
    spec: ActionEncodingSpec,
    device: Union[str, torch.device] = "cpu",
) -> bool:
    """
    Attempt to build the fast extension and verify that it runs for the given device.

    Returns True when the C++ path is available, False otherwise.
    """
    try:
        ext = _load_extension()
    except Exception:
        return False

    if ext is None:
        return False

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        return False
    dtypes = [torch.float32]
    if torch_device.type == "cuda":
        dtypes.append(torch.float16)

    for dtype in dtypes:
        log_p1 = torch.zeros((1, spec.placement_dim), dtype=dtype, device=torch_device)
        log_p2 = torch.zeros_like(log_p1)
        log_pmc = torch.zeros_like(log_p1)
        legal_mask = torch.zeros((1, spec.total_dim), dtype=torch.bool, device=torch_device)
        if spec.total_dim > 0:
            legal_mask[..., 0] = True
        try:
            result = project_policy_logits_fast(log_p1, log_p2, log_pmc, legal_mask, spec)
        except RuntimeError:
            return False
        if result is None:
            return False
    return True
