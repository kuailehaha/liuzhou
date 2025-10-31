"""
Tensorized training pipeline built on top of the vectorized self-play runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from ..game.move_encoder import (
    ActionEncodingSpec,
    DIRECTIONS,
    DEFAULT_ACTION_SPEC,
)
from ..self_play.runner import SelfPlayBatchResult, SelfPlayConfig, run_self_play
from ..self_play.samples import RolloutTensorBatch
from ..train.dataset import TensorRolloutDataset


@dataclass
class TrainingLoopConfig:
    self_play_config: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    soft_label_alpha: float = 0.0
    dataloader_workers: int = 0
    shuffle_buffer: bool = True


def _flatten_self_play_result(
    result: SelfPlayBatchResult,
    action_spec: ActionEncodingSpec,
) -> RolloutTensorBatch:
    """
    Convert a batched self-play result into flat tensors suitable for training.
    """
    batch_size, _, action_dim = result.policies.shape
    board_size = result.states.shape[-1] if result.states.numel() else GameState.BOARD_SIZE

    states_list: List[torch.Tensor] = []
    policy_list: List[torch.Tensor] = []
    legal_list: List[torch.BoolTensor] = []
    values_list: List[torch.Tensor] = []
    soft_list: List[torch.Tensor] = []

    for game_idx in range(batch_size):
        length = int(result.lengths[game_idx].item())
        if length <= 0:
            continue
        outcome = float(result.results[game_idx].item())
        soft_value = float(result.soft_values[game_idx].item())
        for step in range(length):
            mask = result.mask[game_idx, step]
            if not bool(mask.item()):
                continue
            legal_mask = result.legal_masks[game_idx, step]
            if legal_mask.sum().item() == 0:
                continue
            policy = result.policies[game_idx, step].clone()
            policy[~legal_mask] = 0.0
            total = policy.sum().item()
            if total <= 0.0:
                continue
            policy /= total
            sign = float(result.player_signs[game_idx, step].item())
            states_list.append(result.states[game_idx, step].clone())
            policy_list.append(policy)
            legal_list.append(legal_mask.clone())
            values_list.append(torch.tensor(sign * outcome, dtype=torch.float32))
            soft_list.append(torch.tensor(sign * soft_value, dtype=torch.float32))

    if not states_list:
        empty_states = torch.empty(0, NUM_INPUT_CHANNELS, board_size, board_size, dtype=torch.float32)
        empty_policy = torch.empty(0, action_spec.total_dim, dtype=torch.float32)
        empty_legal = torch.zeros(0, action_spec.total_dim, dtype=torch.bool)
        empty_vals = torch.empty(0, dtype=torch.float32)
        return RolloutTensorBatch(
            states=empty_states,
            policies=empty_policy,
            values=empty_vals,
            soft_values=empty_vals.clone(),
            legal_masks=empty_legal,
        )

    states_tensor = torch.stack(states_list, dim=0)
    policies_tensor = torch.stack(policy_list, dim=0)
    legal_tensor = torch.stack(legal_list, dim=0)
    values_tensor = torch.stack(values_list, dim=0)
    soft_tensor = torch.stack(soft_list, dim=0)

    return RolloutTensorBatch(
        states=states_tensor,
        policies=policies_tensor,
        values=values_tensor,
        soft_values=soft_tensor,
        legal_masks=legal_tensor,
    )


def generate_training_data(
    model: ChessNet,
    config: TrainingLoopConfig,
    device: str = "cpu",
) -> RolloutTensorBatch:
    """
    Run self-play and package the tensor outputs for training.
    """
    result = run_self_play(
        model=model,
        batch_size=config.batch_size,
        device=device,
        config=config.self_play_config,
    )
    return _flatten_self_play_result(result, config.self_play_config.action_spec)


def _gather_action_log_scores(
    log_p1_flat: torch.Tensor,
    log_p2_flat: torch.Tensor,
    log_pmc_flat: torch.Tensor,
    indices: torch.Tensor,
    spec: ActionEncodingSpec,
    board_size: int,
) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    placement_dim = spec.placement_dim
    movement_dim = spec.movement_dim
    selection_dim = spec.selection_dim
    total_dim = spec.total_dim
    device = log_p1_flat.device

    for idx in indices.tolist():
        if idx < placement_dim:
            scores.append(log_p1_flat[idx])
        elif idx < placement_dim + movement_dim:
            rel = idx - placement_dim
            cell_idx, dir_idx = divmod(rel, len(DIRECTIONS))
            src_r = cell_idx // board_size
            src_c = cell_idx % board_size
            dr, dc = DIRECTIONS[dir_idx]
            dst_r = src_r + dr
            dst_c = src_c + dc
            if 0 <= dst_r < board_size and 0 <= dst_c < board_size:
                dst_idx = dst_r * board_size + dst_c
                scores.append(log_p2_flat[cell_idx] + log_p1_flat[dst_idx])
            else:
                scores.append(torch.full((), float("-inf"), device=device))
        elif idx < placement_dim + movement_dim + selection_dim:
            rel = idx - (placement_dim + movement_dim)
            scores.append(log_pmc_flat[rel])
        else:
            aux_rel = idx - (placement_dim + movement_dim + selection_dim)
            if aux_rel == 0:
                scores.append(torch.zeros((), device=device))
            else:
                scores.append(torch.full((), float("-inf"), device=device))
    if not scores:
        return torch.empty(0, device=device)
    return torch.stack(scores, dim=0)


def _compute_policy_loss(
    log_p1_batch: torch.Tensor,
    log_p2_batch: torch.Tensor,
    log_pmc_batch: torch.Tensor,
    target_policies: torch.Tensor,
    legal_masks: torch.BoolTensor,
    action_spec: ActionEncodingSpec,
    board_size: int,
) -> Tuple[torch.Tensor, int]:
    losses: List[torch.Tensor] = []
    batch_size = log_p1_batch.size(0)
    for i in range(batch_size):
        indices = legal_masks[i].nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        targets = target_policies[i, indices]
        total = targets.sum()
        if total <= 0:
            continue
        targets = targets / total
        log_scores = _gather_action_log_scores(
            log_p1_batch[i].view(-1),
            log_p2_batch[i].view(-1),
            log_pmc_batch[i].view(-1),
            indices,
            action_spec,
            board_size,
        )
        if log_scores.numel() == 0:
            continue
        log_probs = torch.log_softmax(log_scores, dim=0)
        loss = -(targets * log_probs).sum()
        losses.append(loss)
    if not losses:
        zero = log_p1_batch.new_zeros(())
        return zero, 0
    stacked = torch.stack(losses, dim=0)
    return stacked.mean(), len(losses)


def train_one_iteration(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutTensorBatch,
    config: TrainingLoopConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Consume tensorized rollout data and perform optimization steps.
    """
    device_obj = torch.device(device)
    dataset = TensorRolloutDataset.from_batch(rollout)
    if len(dataset) == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "samples": 0,
            "policy_samples": 0,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_buffer,
        num_workers=config.dataloader_workers,
        pin_memory=False,
    )

    model.to(device_obj)
    model.train()

    value_loss_fn = nn.MSELoss()

    total_samples = 0
    total_policy_samples = 0
    sum_loss = 0.0
    sum_value_loss = 0.0
    sum_policy_loss = 0.0

    board_size = model.board_size if hasattr(model, "board_size") else GameState.BOARD_SIZE
    action_spec = config.self_play_config.action_spec

    for _ in range(config.epochs):
        for states, policies, values, soft_values, legal_masks in dataloader:
            states = states.to(device_obj)
            policies = policies.to(device_obj)
            values = values.to(device_obj)
            soft_values = soft_values.to(device_obj)
            legal_masks = legal_masks.to(device_obj)

            optimizer.zero_grad()

            log_p1, log_p2, log_pmc, value_pred = model(states)
            value_pred = value_pred.squeeze(-1)

            mix_targets = (1.0 - config.soft_label_alpha) * values + config.soft_label_alpha * soft_values
            value_loss = value_loss_fn(value_pred, mix_targets)

            policy_loss, valid_policy = _compute_policy_loss(
                log_p1,
                log_p2,
                log_pmc,
                policies,
                legal_masks,
                action_spec,
                board_size,
            )

            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            batch_size_actual = states.size(0)
            total_samples += batch_size_actual
            total_policy_samples += valid_policy
            sum_loss += loss.item() * batch_size_actual
            sum_value_loss += value_loss.item() * batch_size_actual
            if valid_policy > 0:
                sum_policy_loss += policy_loss.item() * batch_size_actual

    avg_loss = sum_loss / max(total_samples, 1)
    avg_value_loss = sum_value_loss / max(total_samples, 1)
    avg_policy_loss = (
        sum_policy_loss / max(total_samples, 1) if total_policy_samples > 0 else 0.0
    )

    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "samples": float(total_samples),
        "policy_samples": float(total_policy_samples),
    }


def training_loop(
    model: ChessNet,
    optimizer: Optional[torch.optim.Optimizer],
    iterations: int,
    config: Optional[TrainingLoopConfig] = None,
    device: str = "cpu",
) -> Iterable[Dict[str, float]]:
    """
    High-level training loop entry point for the tensorized pipeline.
    """
    cfg = config or TrainingLoopConfig()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    history: List[Dict[str, float]] = []
    for iteration in range(iterations):
        rollout = generate_training_data(model, cfg, device=device)
        metrics = train_one_iteration(model, optimizer, rollout, cfg, device=device)
        metrics["iteration"] = iteration
        metrics["examples"] = float(rollout.states.shape[0])
        history.append(metrics)
    return history
