"""Tensor-native training bridge for v1 self-play output."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.neural_network import scalar_to_wdl, wdl_to_scalar
from src.policy_batch import (
    batched_policy_loss,
    build_combined_logits,
    masked_log_softmax,
)

from .trajectory_buffer import TensorSelfPlayBatch


def _train_permutation(num_samples: int, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.randperm(num_samples, device=device)
    return torch.randperm(num_samples)


def train_network_from_tensors(
    model,
    samples: TensorSelfPlayBatch,
    *,
    batch_size: int = 512,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    soft_label_alpha: float = 0.0,
    policy_draw_weight: float = 1.0,
    device: str = "cpu",
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Train directly from tensor-native self-play output."""

    if samples.num_samples <= 0:
        return model, {"epoch_stats": [], "num_samples": 0}

    device_obj = torch.device(device)
    model.to(device_obj)
    model.train()

    states = samples.state_tensors.to(device_obj, non_blocking=True).to(torch.float32)
    legal_masks = samples.legal_masks.to(device_obj, non_blocking=True).to(torch.bool)
    policy_targets = samples.policy_targets.to(device_obj, non_blocking=True).to(torch.float32)
    value_targets = samples.value_targets.to(device_obj, non_blocking=True).to(torch.float32).view(-1, 1)
    soft_targets = samples.soft_value_targets.to(device_obj, non_blocking=True).to(torch.float32).view(-1, 1)

    num_samples = int(states.shape[0])
    if num_samples == 0:
        return model, {"epoch_stats": [], "num_samples": 0}

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp_enabled = bool(use_amp and device_obj.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_enabled)

    alpha = float(max(0.0, min(1.0, soft_label_alpha)))
    draw_weight = float(max(0.0, policy_draw_weight))
    bsz = max(1, int(batch_size))
    epoch_stats: List[Dict[str, Any]] = []

    for epoch in range(max(1, int(epochs))):
        perm = _train_permutation(num_samples, device_obj)
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        total_seen = 0
        total_policy_weight = 0.0
        total_valid_policy = 0
        soft_abs_sum = 0.0
        mix_abs_sum = 0.0
        mix_batches = 0

        for start in range(0, num_samples, bsz):
            end = min(start + bsz, num_samples)
            idx = perm[start:end]

            batch_states = states.index_select(0, idx)
            batch_masks = legal_masks.index_select(0, idx)
            batch_policy = policy_targets.index_select(0, idx)
            batch_values = value_targets.index_select(0, idx)
            batch_soft = soft_targets.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp_enabled):
                log_p1, log_p2, log_pmc, wdl_logits = model(batch_states)

                wdl_hard = scalar_to_wdl(batch_values)
                wdl_soft = scalar_to_wdl(batch_soft)
                wdl_target = (1.0 - alpha) * wdl_hard + alpha * wdl_soft
                wdl_log_probs = torch.log_softmax(wdl_logits, dim=-1)
                value_loss = -(wdl_target * wdl_log_probs).sum(dim=-1).mean()

                combined_logits = build_combined_logits(
                    log_p1.view(log_p1.size(0), -1),
                    log_p2.view(log_p2.size(0), -1),
                    log_pmc.view(log_pmc.size(0), -1),
                    board_size=6,
                )
                log_probs = masked_log_softmax(combined_logits, batch_masks, dim=1)
                policy_loss = batched_policy_loss(
                    log_probs,
                    batch_policy,
                    batch_masks,
                    batch_values,
                    draw_weight,
                )
                loss = policy_loss + value_loss

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()

            draw_mask = batch_values.abs().squeeze(1) < 1e-8
            policy_weights = torch.where(
                draw_mask,
                batch_values.new_full((), draw_weight),
                batch_values.new_ones(()),
            )
            weight_sum = float(policy_weights.sum().item())
            valid_policy = int((batch_policy.sum(dim=1) > 1e-8).sum().item())

            batch_count = int(batch_states.size(0))
            total_seen += batch_count
            total_loss_sum += float(loss.item()) * batch_count
            policy_loss_sum += float(policy_loss.item()) * weight_sum
            value_loss_sum += float(value_loss.item()) * batch_count
            total_policy_weight += weight_sum
            total_valid_policy += valid_policy

            soft_abs_sum += float(batch_soft.abs().mean().item())
            mix_abs_sum += float(wdl_to_scalar(wdl_target).abs().mean().item())
            mix_batches += 1

        avg_loss = total_loss_sum / max(1, total_seen)
        avg_policy_loss = policy_loss_sum / max(1e-8, total_policy_weight)
        avg_value_loss = value_loss_sum / max(1, total_seen)
        avg_soft_abs = soft_abs_sum / max(1, mix_batches)
        avg_mix_abs = mix_abs_sum / max(1, mix_batches)

        epoch_stats.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_policy_loss": avg_policy_loss,
                "avg_value_loss": avg_value_loss,
                "samples": total_seen,
                "valid_policy_samples": total_valid_policy,
                "policy_weight_sum": total_policy_weight,
                "soft_alpha": alpha,
                "avg_soft_abs": avg_soft_abs,
                "avg_mix_abs": avg_mix_abs,
            }
        )

    return model, {"epoch_stats": epoch_stats, "num_samples": num_samples}

