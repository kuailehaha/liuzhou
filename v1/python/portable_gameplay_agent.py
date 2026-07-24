"""Persistent single-game agent backed by the production portable MCTS."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch

from src.game_state import GameState
from src.move_generator import MoveType, apply_move, generate_all_legal_moves
from src.policy_batch import action_to_index

from .portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree


def state_fingerprint(state: GameState) -> tuple[Any, ...]:
    return (
        tuple(tuple(int(cell) for cell in row) for row in state.board),
        int(state.phase.value),
        int(state.current_player.value),
        tuple(sorted(state.marked_black)),
        tuple(sorted(state.marked_white)),
        int(state.forced_removals_done),
        int(state.move_count),
        int(state.moves_since_capture),
        int(state.pending_marks_required),
        int(state.pending_marks_remaining),
        int(state.pending_captures_required),
        int(state.pending_captures_remaining),
    )


class PortableGameplayAgent:
    """Use one persistent portable tree and never silently change backend/device."""

    def __init__(
        self,
        model,
        *,
        mcts_simulations: int,
        temperature: float = 0.0,
        device: str = "cpu",
        portable_mcts_backend: str = "cpp",
        portable_cpp_threads: int = 1,
    ) -> None:
        backend = str(portable_mcts_backend).strip().lower()
        if backend not in {"python", "cpp"}:
            raise ValueError("portable_mcts_backend must be python or cpp")
        if int(portable_cpp_threads) <= 0:
            raise ValueError("portable_cpp_threads must be positive")
        self.model = model
        self.backend = backend
        self.device = str(device)
        self.mcts_simulations = max(1, int(mcts_simulations))
        self.temperature = float(temperature)
        self.portable_cpp_threads = int(portable_cpp_threads)
        self.config = PortableMCTSConfig(
            num_simulations=self.mcts_simulations,
            exploration_weight=1.0,
            temperature=self.temperature,
            policy_target_temperature=None,
            policy_target_prior_pseudocount=0.0,
            add_dirichlet_noise=False,
            sample_moves=False,
        )
        self._expected_fingerprint: Optional[tuple[Any, ...]] = None
        self._python_search: Optional[PortableMCTS] = None
        self._python_tree: Optional[PortableTree] = None
        self._cpp_search: Any = None
        self.last_search: Dict[str, Any] = {}

    def sync_state(self, state: GameState) -> None:
        """Reset only after an external move or explicit state replacement."""

        self._expected_fingerprint = state_fingerprint(state)
        if self.backend == "python":
            if self._python_search is None:
                self._python_search = PortableMCTS(
                    model=self.model,
                    config=self.config,
                    device=self.device,
                )
            self._python_tree = PortableTree(state)
            self._cpp_search = None
        else:
            from .portable_cpp_mcts import PortableCppMCTS

            self._cpp_search = PortableCppMCTS(
                model=self.model,
                config=self.config,
                device=self.device,
                num_threads=self.portable_cpp_threads,
                initial_states=[state],
            )
            self._python_tree = None

    def reset_search(self, state: Optional[GameState] = None) -> None:
        self._expected_fingerprint = None
        self._python_tree = None
        self._cpp_search = None
        if state is not None:
            self.sync_state(state)

    def select_move(self, state: GameState) -> MoveType:
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            raise ValueError("PortableGameplayAgent has no legal move.")
        reused_tree = self._expected_fingerprint is not None
        if self._expected_fingerprint is None:
            self.sync_state(state)
            reused_tree = False
        elif self._expected_fingerprint != state_fingerprint(state):
            raise ValueError(
                "PortableGameplayAgent state mismatch; call sync_state after an "
                "external move instead of silently rebuilding the search."
            )

        started = time.perf_counter()
        if self.backend == "python":
            assert self._python_search is not None
            assert self._python_tree is not None
            output = self._python_search.search_batch(
                [self._python_tree],
                temperatures=self.temperature,
                add_dirichlet_noise=False,
            )[0]
        else:
            if self._cpp_search is None:
                raise RuntimeError("Portable C++ gameplay search is not initialized.")
            output = self._cpp_search.search_batch(
                temperatures=self.temperature,
                add_dirichlet_noise=False,
            )[0]
            if int(self._cpp_search.illegal_action_count) != 0:
                raise RuntimeError("Portable C++ gameplay observed an illegal action.")
            if int(self._cpp_search.non_finite_count) != 0:
                raise RuntimeError("Portable C++ gameplay observed a non-finite value.")

        action_index = output.chosen_action_index
        if action_index is None:
            raise ValueError("PortableGameplayAgent search returned no action.")
        move_by_index = {
            int(encoded): move
            for move in legal_moves
            for encoded in [action_to_index(move, GameState.BOARD_SIZE)]
            if encoded is not None
        }
        chosen_move = move_by_index.get(int(action_index))
        if chosen_move is None:
            raise RuntimeError(
                f"PortableGameplayAgent could not decode legal action {action_index}."
            )

        next_state = apply_move(state, chosen_move, quiet=True)
        if self.backend == "python":
            assert self._python_tree is not None
            if not self._python_tree.advance_root(int(action_index)):
                raise RuntimeError(
                    f"Portable Python subtree reuse failed for action {action_index}."
                )
        else:
            self._cpp_search.advance_roots([int(action_index)])
        self._expected_fingerprint = state_fingerprint(next_state)

        legal_indices = torch.where(output.legal_mask)[0].tolist()
        top_rows = sorted(
            legal_indices,
            key=lambda index: (
                int(output.visit_counts.get(int(index), 0)),
                float(output.root_priors[int(index)].item()),
                -int(index),
            ),
            reverse=True,
        )[:10]
        top = [
            {
                "actionIndex": int(index),
                "visits": int(output.visit_counts.get(int(index), 0)),
                "prior": float(output.root_priors[int(index)].item()),
                "q": float(output.root_action_values[int(index)].item()),
                "selectionProbability": float(
                    output.selection_policy_dense[int(index)].item()
                ),
            }
            for index in top_rows
        ]
        elapsed = max(0.0, time.perf_counter() - started)
        if not math.isfinite(elapsed):
            raise RuntimeError("Portable gameplay timing became non-finite.")
        self.last_search = {
            "backend": f"portable_{self.backend}",
            "device": self.device,
            "simulations": self.mcts_simulations,
            "temperature": self.temperature,
            "elapsedSec": elapsed,
            "treeReused": bool(reused_tree),
            "chosenActionIndex": int(action_index),
            "rootValue": float(output.root_value),
            "top": top,
            "fallbackCount": 0,
            "illegalActionCount": 0,
            "nonFiniteCount": 0,
        }
        return chosen_move

    def audit_metadata(self) -> Dict[str, Any]:
        return {
            "searchBackend": f"portable_{self.backend}",
            "device": self.device,
            "simulations": self.mcts_simulations,
            "temperature": self.temperature,
            "portableCppThreads": self.portable_cpp_threads,
            "lastSearch": dict(self.last_search),
        }
