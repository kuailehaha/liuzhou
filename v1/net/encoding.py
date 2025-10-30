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


def states_to_model_input(batch: TensorStateBatch) -> torch.Tensor:
    """
    Convert a `TensorStateBatch` into model-ready tensors (B, C, H, W).
    """
    raise NotImplementedError("states_to_model_input must align with the final channel semantics.")


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

