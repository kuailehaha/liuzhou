"""
Tensorized training pipeline scaffold.

This module will integrate the vectorized self-play runner with the existing
optimization routines once the batched data flow is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterable

import torch

from ..self_play.runner import run_self_play, SelfPlayConfig
from ..self_play.samples import RolloutTensorBatch


@dataclass
class TrainingLoopConfig:
    self_play_config: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def generate_training_data(model, config: TrainingLoopConfig, device: str = "cpu") -> RolloutTensorBatch:
    """
    Run self-play and package the tensor outputs for training.
    """
    raise NotImplementedError("generate_training_data will call run_self_play once implemented.")


def train_one_iteration(
    model,
    optimizer,
    rollout: RolloutTensorBatch,
    config: TrainingLoopConfig,
) -> dict:
    """
    Consume tensorized rollout data and perform optimization steps.
    """
    raise NotImplementedError("train_one_iteration will plug in the loss computation.")


def training_loop(
    model,
    optimizer,
    iterations: int,
    config: Optional[TrainingLoopConfig] = None,
    device: str = "cpu",
) -> Iterable[dict]:
    """
    High-level training loop entry point for the tensorized pipeline.
    """
    raise NotImplementedError("training_loop needs self-play + optimization integrated.")
