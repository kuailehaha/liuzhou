"""
Policy decoding helpers for tensorized MCTS and self-play.
"""

from __future__ import annotations

import torch

from ..game.move_encoder import ActionEncodingSpec, decode_action_indices


def sample_actions(
    probs: torch.Tensor,
    legal_mask: torch.BoolTensor,
    temperature: float,
    spec: ActionEncodingSpec,
) -> torch.Tensor:
    """
    Sample a batched set of actions respecting legality masks.
    """
    raise NotImplementedError("sample_actions will implement tempered sampling with masking.")


def argmax_actions(
    logits: torch.Tensor,
    legal_mask: torch.BoolTensor,
    spec: ActionEncodingSpec,
) -> torch.Tensor:
    """
    Select the highest-probability legal action per sample.
    """
    raise NotImplementedError("argmax_actions will be used for evaluation agents.")

