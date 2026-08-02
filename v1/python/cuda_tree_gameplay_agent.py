"""Persistent single-game agent backed by the GPU-resident full MCTS tree."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Optional

import torch

from src.game_state import GameState
from src.move_generator import MoveType, apply_move, generate_all_legal_moves
from src.policy_batch import action_to_index

from .mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
from .mcts_gpu import GpuStateBatch
from .portable_gameplay_agent import state_fingerprint


def game_state_to_gpu_batch(
    state: GameState,
    device: torch.device | str,
) -> GpuStateBatch:
    """Encode one authoritative Python state without changing its semantics."""

    dev = torch.device(device)
    marks_black = torch.zeros((1, 6, 6), dtype=torch.bool, device=dev)
    marks_white = torch.zeros((1, 6, 6), dtype=torch.bool, device=dev)
    for row, col in state.marked_black:
        marks_black[0, row, col] = True
    for row, col in state.marked_white:
        marks_white[0, row, col] = True

    def scalar(value: int) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.int64, device=dev)

    return GpuStateBatch(
        board=torch.tensor(state.board, dtype=torch.int8, device=dev).unsqueeze(0),
        marks_black=marks_black,
        marks_white=marks_white,
        phase=scalar(state.phase.value),
        current_player=scalar(state.current_player.value),
        pending_marks_required=scalar(state.pending_marks_required),
        pending_marks_remaining=scalar(state.pending_marks_remaining),
        pending_captures_required=scalar(state.pending_captures_required),
        pending_captures_remaining=scalar(state.pending_captures_remaining),
        forced_removals_done=scalar(state.forced_removals_done),
        move_count=scalar(state.move_count),
        moves_since_capture=scalar(state.moves_since_capture),
    )


class CudaTreeGameplayAgent:
    """Run deterministic full-tree MCTS for one interactive CUDA game."""

    def __init__(
        self,
        model,
        *,
        mcts_simulations: int,
        temperature: float = 0.0,
        device: str = "cuda:0",
        precision: str = "amp_fp16",
        ponder_enabled: bool = True,
        ponder_chunk_simulations: int = 64,
        ponder_max_simulations: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("cuda_tree gameplay requires an available CUDA device")
        self.precision = str(precision).strip().lower()
        if self.precision not in {"fp32", "amp_fp16"}:
            raise ValueError("cuda_tree precision must be fp32 or amp_fp16")
        self.model = model.to(self.device).eval()
        self.mcts_simulations = max(1, int(mcts_simulations))
        self.temperature = float(temperature)
        self.ponder_enabled = bool(ponder_enabled)
        self.ponder_chunk_simulations = min(
            self.mcts_simulations,
            max(1, int(ponder_chunk_simulations)),
        )
        self.ponder_max_simulations = max(
            self.ponder_chunk_simulations,
            int(
                4 * self.mcts_simulations
                if ponder_max_simulations is None
                else ponder_max_simulations
            ),
        )
        self.mcts = CudaTreeMCTS(
            model=self.model,
            config=CudaTreeMCTSConfig(
                num_simulations=self.mcts_simulations,
                exploration_weight=1.0,
                temperature=self.temperature,
                add_dirichlet_noise=False,
                sample_moves=False,
                precision=self.precision,
                arena_node_multiplier=12,
            ),
            device=self.device,
        )
        self.expected_fingerprint: Optional[tuple[Any, ...]] = None
        self._has_searched = False
        self._mcts_lock = threading.RLock()
        self._metadata_lock = threading.Lock()
        self._ponder_stop = threading.Event()
        self._ponder_thread: Optional[threading.Thread] = None
        self._ponder_rounds = 0
        self._ponder_simulations = 0
        self._ponder_elapsed_sec = 0.0
        self._ponder_error: Optional[str] = None
        self._last_metrics = dict(self.mcts.metrics())
        self.last_search: Dict[str, Any] = {}

    def _reset_active_tree(self) -> None:
        if self._has_searched:
            with self._mcts_lock:
                self.mcts.reset_tree_rows(
                    torch.zeros((1,), dtype=torch.int64, device=self.device)
                )

    def _cache_metrics(self) -> Dict[str, Any]:
        metrics = dict(self.mcts.metrics())
        with self._metadata_lock:
            self._last_metrics = metrics
        return metrics

    @property
    def ponder_active(self) -> bool:
        with self._metadata_lock:
            thread = self._ponder_thread
            return bool(thread is not None and thread.is_alive())

    def _ponder_loop(self, state: GameState) -> None:
        try:
            while not self._ponder_stop.is_set():
                with self._metadata_lock:
                    remaining = self.ponder_max_simulations - self._ponder_simulations
                if remaining <= 0:
                    break
                chunk = min(self.ponder_chunk_simulations, remaining)
                started = time.perf_counter()
                with self._mcts_lock:
                    if self._ponder_stop.is_set():
                        break
                    self.mcts.search_batch(
                        game_state_to_gpu_batch(state, self.device),
                        temperatures=self.temperature,
                        add_dirichlet_noise=False,
                        num_simulations=chunk,
                    )
                    self._has_searched = True
                    self._cache_metrics()
                elapsed = max(0.0, time.perf_counter() - started)
                with self._metadata_lock:
                    self._ponder_rounds += 1
                    self._ponder_simulations += chunk
                    self._ponder_elapsed_sec += elapsed
        except Exception as exc:
            with self._metadata_lock:
                self._ponder_error = f"{type(exc).__name__}: {exc}"
            self._ponder_stop.set()

    def start_pondering(self, state: GameState) -> None:
        """Continue full-tree search while the external player is thinking."""

        if not self.ponder_enabled or state.is_game_over():
            return
        self.stop_pondering()
        fingerprint = state_fingerprint(state)
        if self.expected_fingerprint is None:
            self.expected_fingerprint = fingerprint
        elif self.expected_fingerprint != fingerprint:
            raise ValueError("Cannot ponder a state that is not the active root.")
        self._ponder_stop.clear()
        with self._metadata_lock:
            self._ponder_rounds = 0
            self._ponder_simulations = 0
            self._ponder_elapsed_sec = 0.0
            self._ponder_error = None
            thread = threading.Thread(
                target=self._ponder_loop,
                args=(state.copy(),),
                name="liuzhou-cuda-tree-ponder",
                daemon=True,
            )
            self._ponder_thread = thread
        thread.start()

    def stop_pondering(self) -> None:
        """Stop after the current small simulation chunk and surface failures."""

        self._ponder_stop.set()
        with self._metadata_lock:
            thread = self._ponder_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._metadata_lock:
            if self._ponder_thread is thread:
                self._ponder_thread = None
            error = self._ponder_error
        if error is not None:
            raise RuntimeError(f"CUDA-tree pondering failed: {error}")

    def sync_state(self, state: GameState) -> None:
        """Synchronize an initial/replaced state and invalidate stale roots."""

        self.stop_pondering()
        fingerprint = state_fingerprint(state)
        if self.expected_fingerprint is not None and self.expected_fingerprint != fingerprint:
            self._reset_active_tree()
        self.expected_fingerprint = fingerprint

    def observe_external_move(
        self,
        state_before: GameState,
        move: MoveType,
        state_after: GameState,
    ) -> None:
        """Advance to a human-selected child, retaining it when it was visited."""

        self.stop_pondering()
        before = state_fingerprint(state_before)
        after = state_fingerprint(state_after)
        if self.expected_fingerprint is not None and self.expected_fingerprint != before:
            self._reset_active_tree()
            self.expected_fingerprint = after
            return
        action_index = action_to_index(move, GameState.BOARD_SIZE)
        if action_index is None:
            self._reset_active_tree()
            self.expected_fingerprint = after
            return
        if self._has_searched:
            with self._mcts_lock:
                self.mcts.advance_roots(
                    torch.tensor([action_index], dtype=torch.int64, device=self.device)
                )
                self._cache_metrics()
        self.expected_fingerprint = after

    def select_move(self, state: GameState) -> MoveType:
        self.stop_pondering()
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            raise ValueError("CudaTreeGameplayAgent has no legal move.")
        fingerprint = state_fingerprint(state)
        if self.expected_fingerprint is None:
            self.expected_fingerprint = fingerprint
        elif self.expected_fingerprint != fingerprint:
            raise ValueError(
                "CudaTreeGameplayAgent state mismatch; synchronize external moves "
                "instead of silently rebuilding the search."
            )

        with self._mcts_lock:
            root_visits_before = (
                int(self.mcts.active_root_visit_counts()[0].item())
                if self._has_searched
                else 0
            )
        remaining = max(1, self.mcts_simulations - root_visits_before)
        search_rounds = max(
            1,
            math.ceil(remaining / self.ponder_chunk_simulations),
        )
        simulations_to_run = search_rounds * self.ponder_chunk_simulations
        tree_reused = root_visits_before > 0
        started = time.perf_counter()
        with self._mcts_lock:
            gpu_state = game_state_to_gpu_batch(state, self.device)
            output = None
            for _ in range(search_rounds):
                output = self.mcts.search_batch(
                    gpu_state,
                    temperatures=self.temperature,
                    add_dirichlet_noise=False,
                    num_simulations=self.ponder_chunk_simulations,
                )
            assert output is not None
        if not bool(output.chosen_valid_mask[0].item()):
            raise ValueError("CudaTreeGameplayAgent search returned no action.")
        action_index = int(output.chosen_action_indices[0].item())
        move_by_index = {
            int(encoded): move
            for move in legal_moves
            for encoded in [action_to_index(move, GameState.BOARD_SIZE)]
            if encoded is not None
        }
        chosen_move = move_by_index.get(action_index)
        if chosen_move is None:
            raise RuntimeError(
                f"CudaTreeGameplayAgent could not decode legal action {action_index}."
            )

        with self._mcts_lock:
            self.mcts.advance_roots(output.chosen_action_indices)
            self._has_searched = True
            metrics = self._cache_metrics()
        next_state = apply_move(state, chosen_move, quiet=True)
        self.expected_fingerprint = state_fingerprint(next_state)

        visits = self.mcts.last_root_visits
        priors = self.mcts.last_root_priors
        values = self.mcts.last_root_action_values
        if visits is None or priors is None or values is None:
            raise RuntimeError("CudaTreeGameplayAgent search audit is unavailable.")
        legal_indices = torch.where(output.legal_mask[0])[0].tolist()
        top_indices = sorted(
            legal_indices,
            key=lambda index: (
                int(visits[0, index].item()),
                float(priors[0, index].item()),
                -int(index),
            ),
            reverse=True,
        )[:10]
        top = [
            {
                "actionIndex": int(index),
                "visits": int(visits[0, index].item()),
                "prior": float(priors[0, index].item()),
                "q": float(values[0, index].item()),
                "selectionProbability": float(output.policy_dense[0, index].item()),
            }
            for index in top_indices
        ]
        elapsed = max(0.0, time.perf_counter() - started)
        if not math.isfinite(elapsed):
            raise RuntimeError("CUDA-tree gameplay timing became non-finite.")
        self.last_search = {
            "backend": "cuda_tree",
            "device": str(self.device),
            "precision": self.precision,
            "simulations": self.mcts_simulations,
            "simulationsRun": simulations_to_run,
            "rootVisitsBeforeSearch": root_visits_before,
            "temperature": self.temperature,
            "elapsedSec": elapsed,
            "treeReused": bool(tree_reused),
            "chosenActionIndex": action_index,
            "rootValue": float(output.root_value[0].item()),
            "top": top,
            "fallbackCount": int(metrics["fallback"]),
            "illegalActionCount": int(metrics["illegal_actions"]),
            "nonFiniteCount": int(metrics["non_finite"]),
            "arenaOverflowCount": int(metrics["arena_overflow"]),
            "subtreeReuseHits": int(metrics["subtree_reuse_hits"]),
            "subtreeReuseMisses": int(metrics["subtree_reuse_misses"]),
            "subtreeStateMismatches": int(metrics["subtree_state_mismatches"]),
        }
        return chosen_move

    def audit_metadata(self) -> Dict[str, Any]:
        with self._metadata_lock:
            metrics = dict(self._last_metrics)
            thread = self._ponder_thread
            ponder_active = bool(thread is not None and thread.is_alive())
            ponder_rounds = int(self._ponder_rounds)
            ponder_simulations = int(self._ponder_simulations)
            ponder_elapsed_sec = float(self._ponder_elapsed_sec)
            ponder_error = self._ponder_error
        return {
            "searchBackend": "cuda_tree",
            "device": str(self.device),
            "precision": self.precision,
            "simulations": self.mcts_simulations,
            "temperature": self.temperature,
            "ponderEnabled": self.ponder_enabled,
            "ponderActive": ponder_active,
            "ponderChunkSimulations": self.ponder_chunk_simulations,
            "ponderMaxSimulations": self.ponder_max_simulations,
            "ponderRounds": ponder_rounds,
            "ponderSimulations": ponder_simulations,
            "ponderElapsedSec": ponder_elapsed_sec,
            "ponderError": ponder_error,
            "arenaPeakBytes": int(metrics["arena_peak_bytes"]),
            "subtreeReuseHits": int(metrics["subtree_reuse_hits"]),
            "subtreeReuseMisses": int(metrics["subtree_reuse_misses"]),
            "subtreeStateMismatches": int(metrics["subtree_state_mismatches"]),
            "subtreeCapacityResets": int(metrics["subtree_capacity_resets"]),
            "arenaOverflowCount": int(metrics["arena_overflow"]),
            "fallbackCount": int(metrics["fallback"]),
            "illegalActionCount": int(metrics["illegal_actions"]),
            "nonFiniteCount": int(metrics["non_finite"]),
            "lastSearch": dict(self.last_search),
        }
