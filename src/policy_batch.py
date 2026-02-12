"""
Batched policy loss utilities for training.
Uses the same flat action index convention as v0 (ActionEncodingSpec:
placement_dim=36, movement_dim=144, selection_dim=36, auxiliary_dim=4, total_dim=220)
so that training and v0 self-play/推理 share one encoding.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.game_state import Phase


# Match v0/python/move_encoder.py ActionEncodingSpec for 6x6 board
PLACEMENT_DIM = 36
MOVEMENT_DIM = 144  # 36 * 4 directions
SELECTION_DIM = 36
AUXILIARY_DIM = 4
TOTAL_DIM = PLACEMENT_DIM + MOVEMENT_DIM + SELECTION_DIM + AUXILIARY_DIM

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(DIRECTIONS)}


def action_to_index(
    move: dict,
    board_size: int,
) -> Optional[int]:
    """
    Map a legacy move dict to flat index in [0, TOTAL_DIM).
    Same convention as v0 move_encoder.action_to_index (placement, movement, selection, auxiliary).
    """
    placement_end = PLACEMENT_DIM
    movement_end = placement_end + MOVEMENT_DIM
    selection_end = movement_end + SELECTION_DIM

    phase = move.get("phase")
    action_type = move.get("action_type")

    if phase == Phase.PLACEMENT and action_type == "place":
        r, c = move["position"]
        return r * board_size + c

    if phase == Phase.MOVEMENT and action_type == "move":
        (r_from, c_from) = move["from_position"]
        (r_to, c_to) = move["to_position"]
        dr = r_to - r_from
        dc = c_to - c_from
        dir_idx = DIR_TO_INDEX.get((dr, dc))
        if dir_idx is None:
            return None
        cell_idx = r_from * board_size + c_from
        return placement_end + cell_idx * len(DIRECTIONS) + dir_idx

    selection_offset = placement_end + MOVEMENT_DIM
    if action_type in {"mark", "capture", "remove", "counter_remove", "no_moves_remove"}:
        r, c = move["position"]
        return selection_offset + r * board_size + c

    if phase == Phase.REMOVAL and action_type == "process_removal":
        return selection_end

    return None


def legal_mask_and_target_dense(
    legal_moves: list,
    target_policy_tensor: torch.Tensor,
    board_size: int,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (1, TOTAL_DIM) legal_mask (bool) and target_dense (float).
    Only legal positions are True / non-zero; target_dense sums to 1 over legal moves.
    """
    if device is None:
        device = torch.device("cpu")
    legal_mask = torch.zeros(1, TOTAL_DIM, dtype=torch.bool, device=device)
    target_dense = torch.zeros(1, TOTAL_DIM, dtype=torch.float32, device=device)
    if not legal_moves or target_policy_tensor.numel() == 0:
        return legal_mask, target_dense
    policy = target_policy_tensor.to(device).view(-1)
    n = min(len(legal_moves), policy.numel())
    for j in range(n):
        idx = action_to_index(legal_moves[j], board_size)
        if idx is not None and 0 <= idx < TOTAL_DIM:
            legal_mask[0, idx] = True
            target_dense[0, idx] = policy[j].item()
    return legal_mask, target_dense


def build_combined_logits(
    log_p1: torch.Tensor,
    log_p2: torch.Tensor,
    log_pmc: torch.Tensor,
    board_size: int,
) -> torch.Tensor:
    """
    Build (B, TOTAL_DIM) combined logits from policy heads.
    Logic mirrors v0 C++ project_policy_logits_fast: placement=log_p1, movement=log_p2[from]+log_p1[to], selection=log_pmc, auxiliary=0.
    """
    B = log_p1.size(0)
    device = log_p1.device
    dtype = log_p1.dtype
    placement_dim = board_size * board_size
    movement_dim = placement_dim * 4
    selection_dim = placement_dim
    auxiliary_dim = AUXILIARY_DIM
    total_dim = placement_dim + movement_dim + selection_dim + auxiliary_dim

    combined = torch.empty(B, total_dim, device=device, dtype=dtype)
    combined[:, :placement_dim] = log_p1

    indices = torch.arange(placement_dim, device=device, dtype=torch.long)
    rows = indices // board_size
    cols = indices % board_size
    neginf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    movement_chunks = []
    for dr, dc in DIRECTIONS:
        dest_rows = rows + dr
        dest_cols = cols + dc
        valid = (dest_rows >= 0) & (dest_rows < board_size) & (dest_cols >= 0) & (dest_cols < board_size)
        dest_indices = (dest_rows * board_size + dest_cols).clamp(0, placement_dim - 1)
        gathered = log_p1[:, dest_indices]
        movement_dir = log_p2 + gathered
        movement_dir = torch.where(valid.unsqueeze(0).expand_as(movement_dir), movement_dir, neginf)
        movement_chunks.append(movement_dir)
    movement_concat = torch.stack(movement_chunks, dim=2).reshape(B, movement_dim)
    combined[:, placement_dim : placement_dim + movement_dim] = movement_concat
    combined[:, placement_dim + movement_dim : placement_dim + movement_dim + selection_dim] = log_pmc
    combined[:, placement_dim + movement_dim + selection_dim :] = 0
    return combined


def masked_log_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Log-softmax only over positions where mask is True on dim.
    Invalid positions get 0 so they do not contribute to loss.
    """
    neginf = logits.new_full((), float("-inf"))
    masked = torch.where(mask, logits, neginf)
    log_sum_exp = torch.logsumexp(masked, dim=dim, keepdim=True)
    has_legal = mask.any(dim=dim, keepdim=True)
    log_sum_exp_finite = torch.where(
        torch.isfinite(log_sum_exp), log_sum_exp, torch.zeros_like(log_sum_exp)
    )
    log_sum_exp_safe = torch.where(has_legal, log_sum_exp_finite, torch.zeros_like(log_sum_exp))
    log_probs = masked - log_sum_exp_safe
    out = torch.where(mask, log_probs, torch.zeros_like(logits))
    return torch.where(torch.isfinite(out), out, torch.full_like(out, -50.0))


def batched_policy_loss(
    log_probs: torch.Tensor,
    target_dense: torch.Tensor,
    legal_mask: torch.Tensor,
    value_batch: torch.Tensor,
    policy_draw_weight: float,
    policy_soft_only: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-sample KL(target || pred) = CE - H(target), then weighted average.
    Base sample weight follows draw weighting (policy_draw_weight), unless
    policy_soft_only=True (ignore hard WDL labels for policy weighting).

    Matches legacy KLDivLoss scale (0~1 nats). Gradient equals CE gradient since H(target) is constant.
    log_probs: (B, total_dim), target_dense: (B, total_dim), value_batch: (B, 1).
    """
    log_probs_safe = log_probs.clamp(min=-50.0)
    ce_per_sample = -(target_dense * log_probs_safe).sum(dim=1)
    target_entropy = -(target_dense * (target_dense.clamp(min=1e-8).log())).sum(dim=1)
    kl_per_sample = ce_per_sample - target_entropy
    if policy_soft_only:
        weight = log_probs.new_ones(log_probs.size(0))
    else:
        draw_mask = value_batch.abs().squeeze(1) < 1e-8
        weight = torch.where(draw_mask, log_probs.new_full((), policy_draw_weight), log_probs.new_ones(()))
    weighted = kl_per_sample * weight
    return weighted.sum() / (weight.sum() + eps)
