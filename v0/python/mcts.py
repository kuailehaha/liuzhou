"""
C++-backed MCTS wrapper that mirrors the legacy `src.mcts.MCTS` API.

Python is only responsible for providing the neural-network forward callback;
all tree selection/expansion/backprop runs inside `v0_core.MCTSCore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch

import v0_core
from src.game_state import GameState
from src.move_generator import apply_move
from .move_encoder import (
    ActionEncodingSpec,
    DEFAULT_ACTION_SPEC,
    action_to_index,
    decode_action_indices,
)
from .state_batch import from_game_states

__all__ = ["MCTS"]


def _state_signature(state: GameState) -> Tuple:
    return (
        state.phase,
        state.current_player,
        tuple(tuple(row) for row in state.board),
        frozenset(state.marked_black),
        frozenset(state.marked_white),
        state.pending_marks_required,
        state.pending_marks_remaining,
        state.pending_captures_required,
        state.pending_captures_remaining,
        state.forced_removals_done,
        state.move_count,
    )


@dataclass
class MCTSParams:
    num_simulations: int = 800
    exploration_weight: float = 1.0
    temperature: float = 1.0
    batch_leaves: int = 16
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    device: str = "cpu"
    seed: int = 12345


class MCTS:
    """
    Thin wrapper over `v0_core.MCTSCore` with an API compatible with `src.mcts.MCTS`.
    """

    def __init__(
        self,
        model,
        num_simulations: int = 800,
        exploration_weight: float = 1.0,
        temperature: float = 1.0,
        device: str = "cpu",
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_K: int = 16,
        seed: int = 12345,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC
        self.device = torch.device(device)
        self.verbose = bool(verbose)
        self.params = MCTSParams(
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
            temperature=temperature,
            batch_leaves=batch_K,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            device=str(self.device),
            seed=seed,
        )

        cfg = v0_core.MCTSConfig()
        cfg.num_simulations = self.params.num_simulations
        cfg.exploration_weight = self.params.exploration_weight
        cfg.temperature = self.params.temperature
        cfg.batch_size = self.params.batch_leaves
        cfg.add_dirichlet_noise = self.params.add_dirichlet_noise
        cfg.dirichlet_alpha = self.params.dirichlet_alpha
        cfg.dirichlet_epsilon = self.params.dirichlet_epsilon
        cfg.virtual_loss = 1.0
        cfg.device = str(self.device)
        cfg.seed = self.params.seed

        self._core = v0_core.MCTSCore(cfg)
        self._core.set_forward_callback(self._forward_callback)

        self._current_state: Optional[GameState] = None
        self._root_signature: Optional[Tuple] = None

    # ------------------------------------------------------------------ Helpers

    def _forward_callback(self, inputs: torch.Tensor):
        inputs = inputs.to(self.device, non_blocking=True)
        self.model.eval()
        if self.device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                log_p1, log_p2, log_pmc, value = self.model(inputs)
        else:
            with torch.inference_mode():
                log_p1, log_p2, log_pmc, value = self.model(inputs)
        return log_p1, log_p2, log_pmc, value

    def _ensure_root(self, state: GameState):
        signature = _state_signature(state)
        if self._root_signature != signature:
            self._core.set_root_state(state)
            self._current_state = state.copy()
            self._root_signature = signature

    def _log_search(
        self,
        state: GameState,
        moves: List[dict],
        probs: np.ndarray,
        action_to_move: Dict[int, dict],
        action_to_policy: Dict[int, float],
    ) -> None:
        print(
            f"\n[v0.MCTS] root_player={state.current_player.name} "
            f"| temperature={self.params.temperature:.2f} | legal_moves={len(moves)}"
        )
        print(state)
        if not moves:
            print("  (no legal moves from this state)")
            return

        child_stats = self._core.get_root_children_stats()
        parent_visit = max(1.0, float(getattr(self._core, "root_visit_count", 0.0) or 0.0))
        rows = []
        for stats in child_stats:
            action_idx = int(stats.get("action_index", -1))
            move = action_to_move.get(action_idx)
            if move is None:
                continue
            visit = float(stats.get("visit_count", 0.0))
            value_sum = float(stats.get("value_sum", 0.0))
            prior = float(stats.get("prior", 0.0))
            q_value = value_sum / visit if visit > 0 else 0.0
            u_value = self.params.exploration_weight * prior * math.sqrt(parent_visit) / (1.0 + visit)
            policy_prob = action_to_policy.get(action_idx, 0.0)
            # 注意：实际选择分数是 -Q + U（子节点Q从对手视角，需要取负）
            rows.append(
                {
                    "visit": visit,
                    "q": q_value,
                    "p": prior,
                    "u": u_value,
                    "pu": -q_value + u_value,  # 修正：实际使用 -Q + U
                    "pi": policy_prob,
                    "move": move,
                }
            )

        rows.sort(key=lambda item: item["visit"], reverse=True)
        if not rows:
            print("  (no child stats available)")
        else:
            print("  -- Root children (sorted by visit count) --")
            for idx, row in enumerate(rows, start=1):
                print(
                    f"  [{idx:02d}] N={row['visit']:5.0f} | Q={row['q']:+.3f} | "
                    f"P={row['p']:.3f} | U={row['u']:.3f} | Q+U={row['pu']:+.3f} | "
                    f"pi={row['pi']:.3f} | move={row['move']}"
                )

        if len(moves) > 1:
            sorted_idx = np.argsort(probs)[::-1]
            topk = min(10, len(sorted_idx))
            print("  -- Policy ranking --")
            for rank in range(topk):
                idx = sorted_idx[rank]
                print(f"    #{rank + 1:02d} pi={probs[idx]:.3f} move={moves[idx]}")
        print("-" * 48)

    # ------------------------------------------------------------------ Public API

    def search(self, state: GameState) -> Tuple[List[dict], np.ndarray]:
        """
        Run MCTS simulations from the provided state and return (moves, probs).
        """
        self._ensure_root(state)
        try:
            self._core.run_simulations(self.params.num_simulations)
        except Exception as exc:
            debug_info = (
                f"phase={state.phase} move_count={state.move_count} "
                f"pending_marks={state.pending_marks_remaining}/{state.pending_marks_required} "
                f"pending_captures={state.pending_captures_remaining}/{state.pending_captures_required} "
                f"forced_removals_done={state.forced_removals_done}"
            )
            raise RuntimeError(f"v0 MCTS run_simulations failed ({debug_info}): {exc}") from exc

        policy_pairs = self._core.get_policy(self.params.temperature)
        if not policy_pairs:
            return [], np.array([], dtype=float)

        action_indices = torch.tensor([idx for idx, _ in policy_pairs], dtype=torch.long)
        replicated_states = [state.copy() for _ in policy_pairs]
        batch = from_game_states(replicated_states, device=torch.device("cpu"))
        decoded = decode_action_indices(action_indices, batch, self.spec)

        moves: List[dict] = []
        probs: List[float] = []
        action_ids: List[int] = []
        action_to_move: Dict[int, dict] = {}
        for (idx, prob), move in zip(policy_pairs, decoded):
            if move is None:
                continue
            idx_int = int(idx)
            moves.append(move)
            probs.append(float(prob))
            action_ids.append(idx_int)
            action_to_move[idx_int] = move

        if not moves:
            return [], np.array([], dtype=float)

        probs_array = np.array(probs, dtype=float)
        total = probs_array.sum()
        if total <= 0:
            probs_array[:] = 1.0 / len(probs_array)
        else:
            probs_array /= total

        if action_ids:
            action_to_policy = {action_ids[i]: float(probs_array[i]) for i in range(len(action_ids))}
        else:
            action_to_policy = {}

        if self.verbose:
            self._log_search(state, moves, probs_array, action_to_move, action_to_policy)

        return moves, probs_array

    def advance_root(self, move: dict) -> None:
        """
        Advance the root to the child reached by `move`, mirroring `src.mcts.MCTS`.
        """
        if self._current_state is None:
            return
        action_idx = action_to_index(move, self._current_state.BOARD_SIZE, self.spec)
        if action_idx is None:
            self.reset()
            return
        self._core.advance_root(int(action_idx))
        try:
            self._current_state = apply_move(self._current_state, move, quiet=True)
            self._root_signature = _state_signature(self._current_state)
        except Exception:
            self.reset()

    def reset(self) -> None:
        self._core.reset()
        self._current_state = None
        self._root_signature = None

    def set_root_state(self, state: GameState) -> None:
        self._core.set_root_state(state)
        self._current_state = state.copy()
        self._root_signature = _state_signature(state)

    def set_root_states(self, states: Sequence[GameState]) -> None:
        if not states:
            self.reset()
            return
        if len(states) != 1:
            raise ValueError("v0 MCTS currently supports a single root state.")
        self.set_root_state(states[0])

    def set_temperature(self, temperature: float) -> None:
        self.params.temperature = float(temperature)

    def get_root_value(self) -> float:
        return self._core.root_value
