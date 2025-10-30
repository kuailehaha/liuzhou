"""
Policy decoding helpers for tensorized MCTS and self-play.
"""

from __future__ import annotations

import torch

from ..game.move_encoder import ActionEncodingSpec


def _prepare_masked_probs(probs: torch.Tensor, legal_mask: torch.BoolTensor) -> torch.Tensor:
    return torch.where(legal_mask, probs, torch.zeros_like(probs))


def sample_actions(
    probs: torch.Tensor,
    legal_mask: torch.BoolTensor,
    temperature: float,
    spec: ActionEncodingSpec,
    active_mask: torch.BoolTensor | None = None,
) -> torch.Tensor:
    """
    Sample a batched set of actions respecting legality masks.

    Returns a tensor of shape (B,) with selected action indices. Inactive rows
    or rows without legal moves yield -1.
    """
    device = probs.device
    batch_size = probs.size(0)
    if active_mask is None:
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    chosen = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    legal_counts = legal_mask.sum(dim=1)
    active_rows = active_mask & (legal_counts > 0)
    if not active_rows.any():
        return chosen

    masked_probs = _prepare_masked_probs(probs, legal_mask)

    if temperature <= 1e-6:
        logits = torch.where(
            legal_mask,
            masked_probs,
            torch.full_like(masked_probs, float("-inf")),
        )
        best = torch.argmax(logits, dim=1)
        chosen[active_rows] = best[active_rows]
        return chosen

    adjusted = masked_probs.clone()
    exponent = 1.0 / max(temperature, 1e-6)
    adjusted = torch.where(adjusted > 0, adjusted.pow(exponent), adjusted)
    adjusted = torch.where(legal_mask, adjusted, torch.zeros_like(adjusted))
    sums = adjusted.sum(dim=1, keepdim=True)
    fallback_rows = active_rows & (sums.squeeze(1) <= 0)
    sampling_rows = active_rows & ~fallback_rows

    if sampling_rows.any():
        dist = adjusted[sampling_rows]
        dist = dist / dist.sum(dim=1, keepdim=True)
        sampled = torch.multinomial(dist, 1).squeeze(1)
        chosen[sampling_rows] = sampled

    if fallback_rows.any():
        uniform = legal_mask[fallback_rows].float()
        uniform = uniform / uniform.sum(dim=1, keepdim=True)
        sampled = torch.multinomial(uniform, 1).squeeze(1)
        chosen[fallback_rows] = sampled

    return chosen


def argmax_actions(
    logits: torch.Tensor,
    legal_mask: torch.BoolTensor,
    spec: ActionEncodingSpec,
) -> torch.Tensor:
    """
    Select the highest-probability legal action per sample. Rows without legal
    moves return -1.
    """
    masked_logits = torch.where(
        legal_mask,
        logits,
        torch.full_like(logits, float("-inf")),
    )
    chosen = torch.argmax(masked_logits, dim=1)
    empty_rows = legal_mask.sum(dim=1) == 0
    chosen = chosen.to(torch.long)
    chosen[empty_rows] = -1
    return chosen
