"""
Self-play runner scaffold for the tensorized pipeline.

The runner orchestrates batched MCTS calls, action sampling, and rollout data
collection. Implementation will follow once vectorized MCTS and move encodings
are ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ..mcts.vectorized_mcts import VectorizedMCTS, VectorizedMCTSConfig
from ..game.state_batch import TensorStateBatch
from ..net.policy_decoder import sample_actions
from ..game.move_encoder import ActionEncodingSpec, DEFAULT_ACTION_SPEC


@dataclass
class SelfPlayConfig:
    mcts: VectorizedMCTSConfig = field(default_factory=VectorizedMCTSConfig)
    temperature_init: float = 1.0
    temperature_final: float = 0.2
    temperature_threshold: int = 30
    max_moves: int = 200
    action_spec: ActionEncodingSpec = field(default_factory=lambda: DEFAULT_ACTION_SPEC)


@dataclass
class SelfPlayBatchResult:
    states: torch.Tensor  # (B, T, C, H, W) placeholder
    policies: torch.Tensor  # (B, T, A)
    results: torch.Tensor  # (B,)
    soft_values: torch.Tensor  # (B,)


def run_self_play(
    model,
    batch_size: int,
    device: str = "cpu",
    config: Optional[SelfPlayConfig] = None,
) -> SelfPlayBatchResult:
    """
    Execute a batch of self-play games using the vectorized components.
    """
    raise NotImplementedError("run_self_play will orchestrate batched rollouts once dependencies are ready.")
