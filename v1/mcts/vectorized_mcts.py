"""
Vectorized MCTS skeleton.

The implementation mirrors the existing `src.mcts.MCTS` API but operates on
batched root states, sharing network evaluations across multiple games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from src.mcts import MCTS

from ..game.state_batch import TensorStateBatch, to_game_states
from ..game.move_encoder import (
    ActionEncodingSpec,
    DEFAULT_ACTION_SPEC,
    action_to_index,
    decode_action_indices,
    encode_actions,
)


@dataclass
class VectorizedMCTSConfig:
    num_simulations: int = 128
    exploration_weight: float = 1.0
    temperature: float = 1.0
    action_spec: ActionEncodingSpec = field(default_factory=lambda: DEFAULT_ACTION_SPEC)
    virtual_loss_weight: float = 0.0
    batch_leaves: int = 16


class VectorizedMCTS:
    """
    WIP batched MCTS driver.

    Methods intentionally mirror the legacy class to simplify eventual drop-in.
    """

    def __init__(self, model, config: Optional[VectorizedMCTSConfig] = None, device: Optional[str] = None):
        self.model = model
        self.config = config or VectorizedMCTSConfig()
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._trees: Dict[int, MCTS] = {}

    def _make_mcts(self) -> MCTS:
        return MCTS(
            model=self.model,
            num_simulations=self.config.num_simulations,
            exploration_weight=self.config.exploration_weight,
            temperature=self.config.temperature,
            device=str(self.device),
            add_dirichlet_noise=False,
            virtual_loss_weight=self.config.virtual_loss_weight,
            batch_K=self.config.batch_leaves,
        )

    def search(self, batch: TensorStateBatch):
        """
        Run batched simulations for the provided root states.

        Returns
        -------
        policy : torch.Tensor
            Shape (B, action_dim) probability over encoded actions.
        legal_mask : torch.BoolTensor
            Same shape mask indicating legal actions for each sample.
        """
        spec = self.config.action_spec
        legal_mask = encode_actions(batch, spec).to(self.device)
        policies = torch.zeros(
            (batch.batch_size, spec.total_dim), dtype=torch.float32, device=self.device
        )

        states = to_game_states(batch)
        board_size = batch.config.board_size

        # Remove stale trees if batch size shrunk
        self._trees = {idx: tree for idx, tree in self._trees.items() if idx < batch.batch_size}

        for idx, state in enumerate(states):
            if batch.mask_alive is not None and not batch.mask_alive[idx].item():
                self._trees.pop(idx, None)
                continue
            if not legal_mask[idx].any().item():
                self._trees.pop(idx, None)
                continue

            mcts = self._trees.get(idx)
            if mcts is None:
                mcts = self._make_mcts()
                self._trees[idx] = mcts
            else:
                # Update config knobs in case they changed between calls
                mcts.num_simulations = self.config.num_simulations
                mcts.exploration_weight = self.config.exploration_weight
                mcts.temperature = self.config.temperature
                mcts.virtual_loss_weight = self.config.virtual_loss_weight
                mcts.batch_K = self.config.batch_leaves

            moves, policy = mcts.search(state)

            if not moves:
                continue

            row = policies[idx]
            mask_row = legal_mask[idx]
            for move, prob in zip(moves, policy):
                action_idx = action_to_index(move, board_size, spec)
                if action_idx is None:
                    continue
                if not mask_row[action_idx].item():
                    continue
                row[action_idx] = float(prob)

            total = row[mask_row].sum()
            if total.item() > 0:
                row[mask_row] /= total
            else:
                legal_indices = mask_row.nonzero(as_tuple=False).flatten()
                if legal_indices.numel() > 0:
                    row[legal_indices] = 1.0 / legal_indices.numel()

        return policies, legal_mask

    def advance_roots(self, batch: TensorStateBatch, action_indices: torch.Tensor):
        """
        Update roots after external callers apply sampled actions.
        `batch` should correspond to the states used to choose `action_indices`.
        """
        if not self._trees:
            return None

        if action_indices.device != torch.device("cpu"):
            indices_cpu = action_indices.to("cpu")
        else:
            indices_cpu = action_indices

        decoded_moves = decode_action_indices(indices_cpu, batch, self.config.action_spec)

        for idx, move in enumerate(decoded_moves):
            tree = self._trees.get(idx)
            if tree is None:
                continue
            if move is None:
                # No valid move, drop cached tree
                self._trees.pop(idx, None)
                continue
            try:
                tree.advance_root(move)
            except Exception:
                # If the move cannot be matched in the tree, discard the cache to rebuild next time.
                self._trees.pop(idx, None)

        return None
