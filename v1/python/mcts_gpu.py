"""v1 GPU-first root-search MCTS utilities.

This module keeps the hot path on GPU:
- legal-mask generation via ``v0_core.encode_actions_fast``
- policy projection via ``v0_core.project_policy_logits_fast``
- state transition via ``v0_core.batch_apply_moves``

The current implementation is a root-PUCT search. It is intentionally isolated
from v0 to allow incremental migration without changing the v0 path.
"""

from __future__ import annotations

import math
import time
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

import v0_core
from src.game_state import GameState
from src.neural_network import bucket_logits_to_scalar
from v0.python.move_encoder import DEFAULT_ACTION_SPEC

PLACEMENT_DIM = int(DEFAULT_ACTION_SPEC.placement_dim)
MOVEMENT_DIM = int(DEFAULT_ACTION_SPEC.movement_dim)
SELECTION_DIM = int(DEFAULT_ACTION_SPEC.selection_dim)
AUXILIARY_DIM = int(DEFAULT_ACTION_SPEC.auxiliary_dim)
TOTAL_ACTION_DIM = int(DEFAULT_ACTION_SPEC.total_dim)
MAX_MOVE_COUNT = int(GameState.MAX_MOVE_COUNT)
NO_CAPTURE_DRAW_LIMIT = int(GameState.NO_CAPTURE_DRAW_LIMIT)
LOSE_PIECE_THRESHOLD = int(GameState.LOSE_PIECE_THRESHOLD)
PHASE_MOVEMENT = int(v0_core.Phase.MOVEMENT)
PHASE_CAPTURE_SELECTION = int(v0_core.Phase.CAPTURE_SELECTION)
PHASE_COUNTER_REMOVAL = int(v0_core.Phase.COUNTER_REMOVAL)
SOFT_TAN_INPUT_CLAMP = float((math.pi * 0.5) - 1e-3)

@dataclass
class GpuStateBatch:
    """Tensor-native game state batch for v1 GPU self-play."""

    board: torch.Tensor
    marks_black: torch.Tensor
    marks_white: torch.Tensor
    phase: torch.Tensor
    current_player: torch.Tensor
    pending_marks_required: torch.Tensor
    pending_marks_remaining: torch.Tensor
    pending_captures_required: torch.Tensor
    pending_captures_remaining: torch.Tensor
    forced_removals_done: torch.Tensor
    move_count: torch.Tensor
    moves_since_capture: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.board.device

    @property
    def batch_size(self) -> int:
        return int(self.board.shape[0])

    def to(self, device: torch.device | str) -> "GpuStateBatch":
        dev = torch.device(device)
        return GpuStateBatch(
            board=self.board.to(dev),
            marks_black=self.marks_black.to(dev),
            marks_white=self.marks_white.to(dev),
            phase=self.phase.to(dev),
            current_player=self.current_player.to(dev),
            pending_marks_required=self.pending_marks_required.to(dev),
            pending_marks_remaining=self.pending_marks_remaining.to(dev),
            pending_captures_required=self.pending_captures_required.to(dev),
            pending_captures_remaining=self.pending_captures_remaining.to(dev),
            forced_removals_done=self.forced_removals_done.to(dev),
            move_count=self.move_count.to(dev),
            moves_since_capture=self.moves_since_capture.to(dev),
        )

    def slice(self, index: int) -> "GpuStateBatch":
        sl = slice(index, index + 1)
        return GpuStateBatch(
            board=self.board[sl],
            marks_black=self.marks_black[sl],
            marks_white=self.marks_white[sl],
            phase=self.phase[sl],
            current_player=self.current_player[sl],
            pending_marks_required=self.pending_marks_required[sl],
            pending_marks_remaining=self.pending_marks_remaining[sl],
            pending_captures_required=self.pending_captures_required[sl],
            pending_captures_remaining=self.pending_captures_remaining[sl],
            forced_removals_done=self.forced_removals_done[sl],
            move_count=self.move_count[sl],
            moves_since_capture=self.moves_since_capture[sl],
        )

    def select(self, indices: torch.Tensor | list[int]) -> "GpuStateBatch":
        if isinstance(indices, list):
            if not indices:
                raise ValueError("indices must not be empty.")
            idx = torch.tensor(indices, dtype=torch.int64, device=self.device)
        else:
            idx = indices.to(device=self.device, dtype=torch.int64).view(-1)
            if int(idx.numel()) == 0:
                raise ValueError("indices must not be empty.")
        return GpuStateBatch(
            board=self.board.index_select(0, idx),
            marks_black=self.marks_black.index_select(0, idx),
            marks_white=self.marks_white.index_select(0, idx),
            phase=self.phase.index_select(0, idx),
            current_player=self.current_player.index_select(0, idx),
            pending_marks_required=self.pending_marks_required.index_select(0, idx),
            pending_marks_remaining=self.pending_marks_remaining.index_select(0, idx),
            pending_captures_required=self.pending_captures_required.index_select(0, idx),
            pending_captures_remaining=self.pending_captures_remaining.index_select(0, idx),
            forced_removals_done=self.forced_removals_done.index_select(0, idx),
            move_count=self.move_count.index_select(0, idx),
            moves_since_capture=self.moves_since_capture.index_select(0, idx),
        )

    @staticmethod
    def initial(device: torch.device | str, batch_size: int = 1) -> "GpuStateBatch":
        dev = torch.device(device)
        board = torch.zeros((batch_size, 6, 6), dtype=torch.int8, device=dev)
        marks_black = torch.zeros((batch_size, 6, 6), dtype=torch.bool, device=dev)
        marks_white = torch.zeros((batch_size, 6, 6), dtype=torch.bool, device=dev)
        phase = torch.ones((batch_size,), dtype=torch.int64, device=dev)  # Phase.PLACEMENT=1
        current_player = torch.ones((batch_size,), dtype=torch.int64, device=dev)  # Player.BLACK=1
        zeros = torch.zeros((batch_size,), dtype=torch.int64, device=dev)
        return GpuStateBatch(
            board=board,
            marks_black=marks_black,
            marks_white=marks_white,
            phase=phase,
            current_player=current_player,
            pending_marks_required=zeros.clone(),
            pending_marks_remaining=zeros.clone(),
            pending_captures_required=zeros.clone(),
            pending_captures_remaining=zeros.clone(),
            forced_removals_done=zeros.clone(),
            move_count=zeros.clone(),
            moves_since_capture=zeros.clone(),
        )


def states_to_model_input(batch: GpuStateBatch) -> torch.Tensor:
    return v0_core.states_to_model_input(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
    )


def encode_actions_fast(batch: GpuStateBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    return v0_core.encode_actions_fast(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
        batch.pending_marks_required,
        batch.pending_marks_remaining,
        batch.pending_captures_required,
        batch.pending_captures_remaining,
        batch.forced_removals_done,
        PLACEMENT_DIM,
        MOVEMENT_DIM,
        SELECTION_DIM,
        AUXILIARY_DIM,
    )


def _coerce_apply_output(output: tuple) -> GpuStateBatch:
    if len(output) == 12:
        return GpuStateBatch(
            board=output[0],
            marks_black=output[1].to(torch.bool),
            marks_white=output[2].to(torch.bool),
            phase=output[3],
            current_player=output[4],
            pending_marks_required=output[5],
            pending_marks_remaining=output[6],
            pending_captures_required=output[7],
            pending_captures_remaining=output[8],
            forced_removals_done=output[9],
            move_count=output[10],
            moves_since_capture=output[11],
        )
    raise RuntimeError(f"Unexpected batch_apply_moves output arity: {len(output)}")


def batch_apply_moves_compat(
    batch: GpuStateBatch,
    action_codes: torch.Tensor,
    parent_indices: torch.Tensor,
) -> GpuStateBatch:
    """Typed wrapper over v0_core.batch_apply_moves."""
    action_codes = action_codes.to(device=batch.device, dtype=torch.int32)
    parent_indices = parent_indices.to(device=batch.device, dtype=torch.int64)
    output = v0_core.batch_apply_moves(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player,
        batch.pending_marks_required,
        batch.pending_marks_remaining,
        batch.pending_captures_required,
        batch.pending_captures_remaining,
        batch.forced_removals_done,
        batch.move_count,
        batch.moves_since_capture,
        action_codes,
        parent_indices,
    )
    return _coerce_apply_output(output)


@dataclass
class V1RootMCTSConfig:
    num_simulations: int = 128
    exploration_weight: float = 1.0
    temperature: float = 1.0
    add_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    sample_moves: bool = True
    autocast_dtype: str = "float16"
    child_eval_mode: str = "value_only"  # "value_only" or "full"
    soft_value_k: float = 2.0


@dataclass
class RootSearchOutput:
    model_input: torch.Tensor
    legal_mask: torch.Tensor
    policy_dense: torch.Tensor
    root_value: float
    terminal: bool
    chosen_action_index: Optional[int]
    chosen_action_code: Optional[torch.Tensor]


@dataclass
class RootSearchBatchOutput:
    model_input: torch.Tensor
    legal_mask: torch.Tensor
    policy_dense: torch.Tensor
    root_value: torch.Tensor
    terminal_mask: torch.Tensor
    chosen_action_indices: torch.Tensor
    chosen_action_codes: torch.Tensor
    chosen_valid_mask: torch.Tensor


@dataclass
class _FinalizeGraphEntry:
    graph: torch.cuda.CUDAGraph
    legal_index_mat: torch.Tensor
    action_code_mat: torch.Tensor
    valid_mask: torch.Tensor
    visits: torch.Tensor
    value_sum: torch.Tensor
    valid_root_indices: torch.Tensor
    root_temps: torch.Tensor
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class V1RootMCTS:
    """GPU-first root search (PUCT over root children only)."""

    _SHARED_FINALIZE_GRAPH_CACHE: Dict[Tuple[int, int, int, int, bool], _FinalizeGraphEntry] = {}
    _SHARED_FINALIZE_GRAPH_LRU: List[Tuple[int, int, int, int, bool]] = []
    _SHARED_FINALIZE_GRAPH_BLOCKED: set[Tuple[int, int, int, int, bool]] = set()
    _WARNED_LAUNCH_BLOCKING_GRAPH_DISABLE: bool = False

    def __init__(
        self,
        model,
        config: V1RootMCTSConfig,
        device: torch.device | str,
        inference_engine=None,
        collect_timing: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.inference_engine = inference_engine
        self._collect_timing = bool(collect_timing)
        self._timing_ms: Dict[str, float] = {
            "root_puct_ms": 0.0,
            "pack_writeback_ms": 0.0,
        }
        self._timing_calls: Dict[str, int] = {
            "root_puct_ms": 0,
            "pack_writeback_ms": 0,
        }
        self._timing_event_queue: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._timing_drain_period = 256
        self._timed_calls_since_drain = 0
        self._finalize_graph_capture_count = 0
        self._finalize_graph_replay_count = 0
        self._finalize_graph_fallback_count = 0
        self._finalize_graph_cache_hit_count = 0
        self._finalize_graph_event_bridge_count = 0
        self._finalize_graph_event_wait_count = 0
        self._finalize_graph_inline_replay_count = 0
        self._terminal_soft_override_count = 0
        self._forced_uniform_pick_count = 0
        self._finalize_graph_cache = V1RootMCTS._SHARED_FINALIZE_GRAPH_CACHE
        self._finalize_graph_lru = V1RootMCTS._SHARED_FINALIZE_GRAPH_LRU
        self._finalize_graph_blocked = V1RootMCTS._SHARED_FINALIZE_GRAPH_BLOCKED
        try:
            self._finalize_graph_max_entries = max(
                6,
                int(os.environ.get("V1_FINALIZE_GRAPH_MAX_ENTRIES", "64")),
            )
        except ValueError:
            self._finalize_graph_max_entries = 64
        self._finalize_graph_min_roots = 32
        self._finalize_graph_min_actions = 8
        self._finalize_graph_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        mode = str(os.environ.get("V1_FINALIZE_GRAPH", "auto")).strip().lower()
        if mode in ("off", "0", "false", "no"):
            self._finalize_graph_enabled = False
        elif mode in ("on", "1", "true", "yes"):
            self._finalize_graph_enabled = self.device.type == "cuda"
        else:
            self._finalize_graph_enabled = self.device.type == "cuda" and not any(
                str(k).upper().startswith("NSYS_") for k in os.environ.keys()
            )
        launch_blocking = str(os.environ.get("CUDA_LAUNCH_BLOCKING", "")).strip().lower()
        if self._finalize_graph_enabled and launch_blocking in ("1", "true", "yes", "on"):
            self._finalize_graph_enabled = False
            if not V1RootMCTS._WARNED_LAUNCH_BLOCKING_GRAPH_DISABLE:
                print(
                    "[v1.mcts] CUDA_LAUNCH_BLOCKING is enabled; disable finalize graph capture "
                    "for compatibility."
                )
                V1RootMCTS._WARNED_LAUNCH_BLOCKING_GRAPH_DISABLE = True
        mode = str(self.config.child_eval_mode).strip().lower()
        if mode not in ("value_only", "full"):
            raise ValueError(
                f"Unsupported child_eval_mode={self.config.child_eval_mode!r}; "
                "expected 'value_only' or 'full'."
            )
        self._child_eval_mode = mode

    def _touch_finalize_graph_key(self, key: Tuple[int, int, int, int, bool]) -> None:
        if key in self._finalize_graph_lru:
            self._finalize_graph_lru.remove(key)
        self._finalize_graph_lru.append(key)
        while len(self._finalize_graph_lru) > self._finalize_graph_max_entries:
            old = self._finalize_graph_lru.pop(0)
            self._finalize_graph_cache.pop(old, None)

    def _drain_timing_events(self, force: bool) -> None:
        if self.device.type != "cuda" or not self._timing_event_queue:
            return
        if force:
            torch.cuda.synchronize(self.device)
        pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for name, start_evt, end_evt in self._timing_event_queue:
            if force or end_evt.query():
                elapsed_ms = float(start_evt.elapsed_time(end_evt))
                self._timing_ms[name] = float(self._timing_ms.get(name, 0.0) + elapsed_ms)
                self._timing_calls[name] = int(self._timing_calls.get(name, 0) + 1)
            else:
                pending.append((name, start_evt, end_evt))
        self._timing_event_queue = pending

    def _capture_finalize_graph(
        self,
        *,
        key: Tuple[int, int, int, int, bool],
        legal_index_mat: torch.Tensor,
        action_code_mat: torch.Tensor,
        valid_mask: torch.Tensor,
        visits: torch.Tensor,
        value_sum: torch.Tensor,
        valid_root_indices: torch.Tensor,
        batch_size: int,
        root_temps: torch.Tensor,
        sample_moves: bool,
    ) -> Optional[_FinalizeGraphEntry]:
        if self.device.type != "cuda":
            return None
        if key in self._finalize_graph_blocked:
            return None
        try:
            s_legal_index_mat = legal_index_mat.detach().clone()
            s_action_code_mat = action_code_mat.detach().clone()
            s_valid_mask = valid_mask.detach().clone()
            s_visits = visits.detach().clone()
            s_value_sum = value_sum.detach().clone()
            s_valid_root_indices = valid_root_indices.detach().clone()
            s_root_temps = root_temps.detach().clone()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = v0_core.root_finalize_from_visits(
                    s_legal_index_mat,
                    s_action_code_mat,
                    s_valid_mask,
                    s_visits,
                    s_value_sum,
                    s_valid_root_indices,
                    batch_size,
                    TOTAL_ACTION_DIM,
                    s_root_temps,
                    sample_moves,
                )
            return _FinalizeGraphEntry(
                graph=graph,
                legal_index_mat=s_legal_index_mat,
                action_code_mat=s_action_code_mat,
                valid_mask=s_valid_mask,
                visits=s_visits,
                value_sum=s_value_sum,
                valid_root_indices=s_valid_root_indices,
                root_temps=s_root_temps,
                outputs=outputs,
            )
        except Exception:
            self._finalize_graph_blocked.add(key)
            return None

    def _root_finalize_from_visits(
        self,
        *,
        legal_index_mat: torch.Tensor,
        action_code_mat: torch.Tensor,
        valid_mask: torch.Tensor,
        visits: torch.Tensor,
        value_sum: torch.Tensor,
        valid_root_indices: torch.Tensor,
        batch_size: int,
        root_temps: torch.Tensor,
        sample_moves: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_roots = int(legal_index_mat.size(0))
        max_actions = int(legal_index_mat.size(1))
        can_graph = (
            self._finalize_graph_enabled
            and num_roots >= self._finalize_graph_min_roots
            and max_actions >= self._finalize_graph_min_actions
            and num_roots == int(batch_size)
        )
        if not can_graph:
            self._finalize_graph_fallback_count += 1
            return v0_core.root_finalize_from_visits(
                legal_index_mat,
                action_code_mat,
                valid_mask,
                visits,
                value_sum,
                valid_root_indices,
                batch_size,
                TOTAL_ACTION_DIM,
                root_temps,
                sample_moves,
            )

        device_idx = int(self.device.index or 0)
        key = (device_idx, num_roots, max_actions, int(batch_size), bool(sample_moves))
        entry = self._finalize_graph_cache.get(key)
        if entry is None:
            if len(self._finalize_graph_cache) >= self._finalize_graph_max_entries:
                self._finalize_graph_fallback_count += 1
                return v0_core.root_finalize_from_visits(
                    legal_index_mat,
                    action_code_mat,
                    valid_mask,
                    visits,
                    value_sum,
                    valid_root_indices,
                    batch_size,
                    TOTAL_ACTION_DIM,
                    root_temps,
                    sample_moves,
                )
            entry = self._capture_finalize_graph(
                key=key,
                legal_index_mat=legal_index_mat,
                action_code_mat=action_code_mat,
                valid_mask=valid_mask,
                visits=visits,
                value_sum=value_sum,
                valid_root_indices=valid_root_indices,
                batch_size=batch_size,
                root_temps=root_temps,
                sample_moves=sample_moves,
            )
            if entry is None:
                self._finalize_graph_fallback_count += 1
                return v0_core.root_finalize_from_visits(
                    legal_index_mat,
                    action_code_mat,
                    valid_mask,
                    visits,
                    value_sum,
                    valid_root_indices,
                    batch_size,
                    TOTAL_ACTION_DIM,
                    root_temps,
                    sample_moves,
                )
            self._finalize_graph_cache[key] = entry
            self._finalize_graph_capture_count += 1
        else:
            self._finalize_graph_cache_hit_count += 1
        self._touch_finalize_graph_key(key)

        caller_stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        replay_stream = self._finalize_graph_stream if self.device.type == "cuda" else None
        use_event_bridge = (
            self.device.type == "cuda"
            and caller_stream is not None
            and replay_stream is not None
            and caller_stream.cuda_stream != replay_stream.cuda_stream
        )

        if use_event_bridge:
            ready_evt = torch.cuda.Event(enable_timing=False, blocking=False)
            done_evt = torch.cuda.Event(enable_timing=False, blocking=False)
            ready_evt.record(caller_stream)
            replay_stream.wait_event(ready_evt)
            self._finalize_graph_event_wait_count += 1
            with torch.cuda.stream(replay_stream):
                entry.legal_index_mat.copy_(legal_index_mat, non_blocking=True)
                entry.action_code_mat.copy_(action_code_mat, non_blocking=True)
                entry.valid_mask.copy_(valid_mask, non_blocking=True)
                entry.visits.copy_(visits, non_blocking=True)
                entry.value_sum.copy_(value_sum, non_blocking=True)
                entry.valid_root_indices.copy_(valid_root_indices, non_blocking=True)
                entry.root_temps.copy_(root_temps, non_blocking=True)
                entry.graph.replay()
                done_evt.record(replay_stream)
            caller_stream.wait_event(done_evt)
            self._finalize_graph_event_wait_count += 1
            self._finalize_graph_event_bridge_count += 1
        else:
            entry.legal_index_mat.copy_(legal_index_mat, non_blocking=True)
            entry.action_code_mat.copy_(action_code_mat, non_blocking=True)
            entry.valid_mask.copy_(valid_mask, non_blocking=True)
            entry.visits.copy_(visits, non_blocking=True)
            entry.value_sum.copy_(value_sum, non_blocking=True)
            entry.valid_root_indices.copy_(valid_root_indices, non_blocking=True)
            entry.root_temps.copy_(root_temps, non_blocking=True)
            entry.graph.replay()
            self._finalize_graph_inline_replay_count += 1
        self._finalize_graph_replay_count += 1
        return entry.outputs

    @contextmanager
    def _nvtx_range(self, name: str):
        if self.device.type != "cuda":
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    @contextmanager
    def _timed(self, name: str):
        if not self._collect_timing:
            with nullcontext():
                yield
            return
        if self.device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            try:
                yield
            finally:
                end_evt.record()
                self._timing_event_queue.append((name, start_evt, end_evt))
                self._timed_calls_since_drain += 1
                if self._timed_calls_since_drain >= self._timing_drain_period:
                    self._timed_calls_since_drain = 0
                    self._drain_timing_events(force=False)
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._timing_ms[name] = float(self._timing_ms.get(name, 0.0) + elapsed_ms)
            self._timing_calls[name] = int(self._timing_calls.get(name, 0) + 1)

    def get_timing(self, reset: bool = False) -> Dict[str, Dict[str, float]]:
        if self._collect_timing and self.device.type == "cuda":
            self._drain_timing_events(force=True)
        payload = {
            "timing_ms": {k: float(v) for k, v in self._timing_ms.items()},
            "timing_calls": {k: int(v) for k, v in self._timing_calls.items()},
            "counters": {
                "finalize_graph_capture_count": int(self._finalize_graph_capture_count),
                "finalize_graph_replay_count": int(self._finalize_graph_replay_count),
                "finalize_graph_fallback_count": int(self._finalize_graph_fallback_count),
                "finalize_graph_cache_hit_count": int(self._finalize_graph_cache_hit_count),
                "finalize_graph_event_bridge_count": int(self._finalize_graph_event_bridge_count),
                "finalize_graph_event_wait_count": int(self._finalize_graph_event_wait_count),
                "finalize_graph_inline_replay_count": int(self._finalize_graph_inline_replay_count),
                "terminal_soft_override_count": int(self._terminal_soft_override_count),
                "forced_uniform_pick_count": int(self._forced_uniform_pick_count),
            },
        }
        if reset:
            self._timing_event_queue.clear()
            self._timed_calls_since_drain = 0
            for key in list(self._timing_ms.keys()):
                self._timing_ms[key] = 0.0
            for key in list(self._timing_calls.keys()):
                self._timing_calls[key] = 0
            self._finalize_graph_capture_count = 0
            self._finalize_graph_replay_count = 0
            self._finalize_graph_fallback_count = 0
            self._finalize_graph_cache_hit_count = 0
            self._finalize_graph_event_bridge_count = 0
            self._finalize_graph_event_wait_count = 0
            self._finalize_graph_inline_replay_count = 0
            self._terminal_soft_override_count = 0
            self._forced_uniform_pick_count = 0
        return payload

    def _autocast_context(self):
        if self.device.type != "cuda":
            return torch.autocast("cpu", enabled=False)
        dtype_key = str(self.config.autocast_dtype).strip().lower()
        if dtype_key in ("bf16", "bfloat16"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _to_scalar_value(self, raw_values: torch.Tensor) -> torch.Tensor:
        if raw_values.dim() == 2 and raw_values.size(1) == 3:
            probs = torch.softmax(raw_values, dim=1)
            return probs[:, 0] - probs[:, 2]
        if raw_values.dim() == 2 and raw_values.size(1) == 1:
            return raw_values[:, 0]
        if raw_values.dim() == 2 and raw_values.size(1) >= 2:
            return bucket_logits_to_scalar(raw_values, num_bins=int(raw_values.size(1)))
        return raw_values.view(-1)

    @staticmethod
    def _terminal_mask_from_next_state(batch: GpuStateBatch) -> torch.Tensor:
        post_movement_phase = (
            batch.phase.eq(PHASE_MOVEMENT)
            .logical_or(batch.phase.eq(PHASE_CAPTURE_SELECTION))
            .logical_or(batch.phase.eq(PHASE_COUNTER_REMOVAL))
        )
        black_count = batch.board.eq(1).sum(dim=(1, 2))
        white_count = batch.board.eq(-1).sum(dim=(1, 2))
        winner_mask = post_movement_phase.logical_and(
            black_count.lt(LOSE_PIECE_THRESHOLD).logical_or(
                white_count.lt(LOSE_PIECE_THRESHOLD)
            )
        )
        draw_mask = batch.move_count.ge(MAX_MOVE_COUNT).logical_or(
            batch.moves_since_capture.ge(NO_CAPTURE_DRAW_LIMIT)
        )
        return winner_mask.logical_or(draw_mask)

    @staticmethod
    def _soft_tan_from_board_black(
        board: torch.Tensor,
        soft_value_k: float,
    ) -> torch.Tensor:
        black = board.eq(1).sum(dim=(1, 2)).to(torch.float32)
        white = board.eq(-1).sum(dim=(1, 2)).to(torch.float32)
        # In Liuzhou, |black-white| <= 18, so normalize by 18 to use full signal range.
        material_delta = (black - white) / 18.0
        scaled = torch.clamp(
            material_delta * float(soft_value_k),
            min=-SOFT_TAN_INPUT_CLAMP,
            max=SOFT_TAN_INPUT_CLAMP,
        )
        shaped = torch.tan(scaled)
        return torch.clamp(shaped, min=-1.0, max=1.0)

    @staticmethod
    def _child_values_to_parent_perspective(
        child_values: torch.Tensor,
        parent_players: torch.Tensor,
        child_players: torch.Tensor,
    ) -> torch.Tensor:
        """Convert value(head) outputs on child states to parent-state perspective.

        The network is trained with current-player perspective targets.
        So sign should only flip when action application switches the side to move.
        """
        vals = child_values.to(torch.float32).view(-1)
        parents = parent_players.to(torch.int64).view(-1)
        children = child_players.to(torch.int64).view(-1)
        if int(vals.numel()) != int(parents.numel()) or int(vals.numel()) != int(children.numel()):
            raise ValueError(
                "child/parent perspective tensors must align: "
                f"values={int(vals.numel())}, parents={int(parents.numel())}, children={int(children.numel())}"
            )
        same_to_move = children.eq(parents)
        return torch.where(same_to_move, vals, -vals)

    def _forward_model(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def _forward_eager(eager_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            self.model.eval()
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with self._autocast_context():
                        return self.model(eager_inputs)
                return self.model(eager_inputs)

        def _use_eager_for_small_batch(n_valid: int, engine_bs: int) -> bool:
            # Graph path pads to engine batch size; for very small batches this wastes
            # substantial compute and can be slower than eager on RTX 3060.
            fallback_threshold = min(128, max(32, engine_bs // 2))
            return n_valid < fallback_threshold

        if self.inference_engine is not None:
            n_valid = int(inputs.size(0))
            engine_bs = int(getattr(self.inference_engine, "batch_size", n_valid))
            if engine_bs <= 0:
                raise RuntimeError("Inference engine batch_size must be positive.")
            if _use_eager_for_small_batch(n_valid, engine_bs):
                return _forward_eager(inputs)
            if n_valid <= engine_bs:
                engine_inputs = inputs
                if engine_bs > n_valid:
                    pad_shape = (
                        engine_bs - n_valid,
                        int(inputs.size(1)),
                        int(inputs.size(2)),
                        int(inputs.size(3)),
                    )
                    pad = torch.zeros(pad_shape, dtype=inputs.dtype, device=inputs.device)
                    engine_inputs = torch.cat([inputs, pad], dim=0)
                log_p1, log_p2, log_pmc, raw_values = self.inference_engine.forward(
                    engine_inputs, n_valid
                )
                return log_p1[:n_valid], log_p2[:n_valid], log_pmc[:n_valid], raw_values[:n_valid]

            log_p1_parts = []
            log_p2_parts = []
            log_pmc_parts = []
            raw_parts = []
            for start in range(0, n_valid, engine_bs):
                end = min(start + engine_bs, n_valid)
                chunk = inputs[start:end]
                chunk_valid = int(end - start)
                if chunk_valid < engine_bs and _use_eager_for_small_batch(chunk_valid, engine_bs):
                    c1, c2, c3, cv = _forward_eager(chunk)
                    log_p1_parts.append(c1[:chunk_valid])
                    log_p2_parts.append(c2[:chunk_valid])
                    log_pmc_parts.append(c3[:chunk_valid])
                    raw_parts.append(cv[:chunk_valid])
                    continue
                if chunk_valid < engine_bs:
                    pad_shape = (
                        engine_bs - chunk_valid,
                        int(chunk.size(1)),
                        int(chunk.size(2)),
                        int(chunk.size(3)),
                    )
                    pad = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
                    chunk = torch.cat([chunk, pad], dim=0)
                c1, c2, c3, cv = self.inference_engine.forward(chunk, chunk_valid)
                log_p1_parts.append(c1[:chunk_valid])
                log_p2_parts.append(c2[:chunk_valid])
                log_pmc_parts.append(c3[:chunk_valid])
                raw_parts.append(cv[:chunk_valid])
            return (
                torch.cat(log_p1_parts, dim=0),
                torch.cat(log_p2_parts, dim=0),
                torch.cat(log_pmc_parts, dim=0),
                torch.cat(raw_parts, dim=0),
            )
        return _forward_eager(inputs)

    def _evaluate_batch(
        self, batch: GpuStateBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = states_to_model_input(batch)
        log_p1, log_p2, log_pmc, raw_values = self._forward_model(inputs)
        values = self._to_scalar_value(raw_values).float()
        legal_mask, metadata = encode_actions_fast(batch)
        probs, _ = v0_core.project_policy_logits_fast(
            log_p1,
            log_p2,
            log_pmc,
            legal_mask,
            PLACEMENT_DIM,
            MOVEMENT_DIM,
            SELECTION_DIM,
            AUXILIARY_DIM,
        )
        return inputs, legal_mask, metadata, probs.float(), values

    def _evaluate_values_only(self, batch: GpuStateBatch) -> torch.Tensor:
        inputs = states_to_model_input(batch)
        _log_p1, _log_p2, _log_pmc, raw_values = self._forward_model(inputs)
        values = self._to_scalar_value(raw_values).float()
        return values

    def apply_action(self, state: GpuStateBatch, action_code: torch.Tensor) -> GpuStateBatch:
        if action_code.dim() == 1:
            action_codes = action_code.view(1, 4)
        else:
            action_codes = action_code
        num_actions = int(action_codes.shape[0])
        if state.batch_size == 1:
            parent_indices = torch.zeros((num_actions,), dtype=torch.int64, device=state.device)
        elif num_actions == int(state.batch_size):
            parent_indices = torch.arange(num_actions, dtype=torch.int64, device=state.device)
        else:
            raise ValueError(
                "action_code batch does not match state batch size: "
                f"state_batch={state.batch_size}, action_batch={num_actions}"
            )
        return batch_apply_moves_compat(state, action_codes, parent_indices)

    @staticmethod
    def _normalize_temperatures(
        temperatures: Optional[float | torch.Tensor | list],
        batch_size: int,
        default_temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        if temperatures is None:
            return torch.full((batch_size,), float(default_temperature), dtype=torch.float32, device=device)
        if isinstance(temperatures, (float, int)):
            return torch.full((batch_size,), float(temperatures), dtype=torch.float32, device=device)
        if isinstance(temperatures, list):
            if len(temperatures) != batch_size:
                raise ValueError(
                    f"temperatures length mismatch: expected {batch_size}, got {len(temperatures)}"
                )
            return torch.tensor(temperatures, dtype=torch.float32, device=device)
        t = torch.as_tensor(temperatures, dtype=torch.float32, device=device).view(-1)
        if int(t.numel()) != batch_size:
            raise ValueError(
                f"temperatures size mismatch: expected {batch_size}, got {int(t.numel())}"
            )
        return t

    @staticmethod
    def _stable_legal_policy_from_visits(
        *,
        visits: torch.Tensor,
        valid_mask: torch.Tensor,
        root_temps: torch.Tensor,
    ) -> torch.Tensor:
        """Build a numerically stable policy from visits and temperature.

        Avoid ``visits ** (1 / T)`` overflow at low T by working in log-space:
        ``softmax(log(visits) / T)`` on legal actions only.
        """
        mask_f = valid_mask.to(torch.float32)
        safe_visits = torch.nan_to_num(
            visits.to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(1e-8)
        safe_temps = torch.nan_to_num(
            root_temps.to(torch.float32),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        ).clamp_min(1e-6).view(-1, 1)

        logits = torch.log(safe_visits) / safe_temps
        logits = logits.masked_fill(~valid_mask, float("-inf"))

        row_max = logits.max(dim=1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

        exp_logits = torch.exp(logits - row_max) * mask_f
        exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = exp_logits.sum(dim=1, keepdim=True)

        fallback = mask_f / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        no_valid = mask_f.sum(dim=1).eq(0.0)
        fallback[no_valid, 0] = 1.0

        probs = exp_logits / row_sum.clamp_min(1e-8)
        bad_rows = torch.logical_or(~torch.isfinite(row_sum.view(-1)), row_sum.view(-1).le(0.0))
        probs = torch.where(bad_rows.view(-1, 1), fallback, probs)
        probs = probs * mask_f
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    def search_batch(
        self,
        state: GpuStateBatch,
        *,
        temperatures: Optional[float | torch.Tensor | list] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random_mask: Optional[torch.Tensor] = None,
    ) -> RootSearchBatchOutput:
        batch_size = int(state.batch_size)
        add_noise = self.config.add_dirichlet_noise if add_dirichlet_noise is None else bool(
            add_dirichlet_noise
        )
        force_uniform_mask: Optional[torch.Tensor] = None
        if force_uniform_random_mask is not None:
            force_uniform_mask = (
                torch.as_tensor(force_uniform_random_mask, device=state.device)
                .to(torch.bool)
                .view(-1)
            )
            if int(force_uniform_mask.numel()) != batch_size:
                raise ValueError(
                    "force_uniform_random_mask size mismatch: "
                    f"expected {batch_size}, got {int(force_uniform_mask.numel())}"
                )
        temp_values = self._normalize_temperatures(
            temperatures=temperatures,
            batch_size=batch_size,
            default_temperature=self.config.temperature,
            device=state.device,
        )

        model_input, legal_mask, metadata, probs, values = self._evaluate_batch(state)
        root_values = values.clone().to(torch.float32)
        legal_mask_bool = legal_mask.to(torch.bool)
        policy_dense = torch.zeros(
            (batch_size, TOTAL_ACTION_DIM),
            dtype=torch.float32,
            device=state.device,
        )
        terminal_mask = torch.zeros((batch_size,), dtype=torch.bool, device=state.device)
        chosen_action_indices = torch.full((batch_size,), -1, dtype=torch.int64, device=state.device)
        chosen_action_codes = torch.full((batch_size, 4), -1, dtype=torch.int32, device=state.device)
        chosen_valid_mask = torch.zeros((batch_size,), dtype=torch.bool, device=state.device)

        with self._nvtx_range("v1.root_pack_sparse_actions"), self._timed("pack_writeback_ms"):
            (
                terminal_mask,
                valid_root_indices,
                counts,
                valid_mask,
                legal_index_mat,
                priors_mat,
                action_code_mat,
                flat_indices,
                action_codes_all,
                parent_indices_all,
            ) = v0_core.root_pack_sparse_actions(
                legal_mask_bool,
                probs,
                metadata,
            )

        if int(valid_root_indices.numel()) > 0:
            num_roots = int(valid_root_indices.numel())
            max_actions = int(valid_mask.shape[1])

            if add_noise:
                if max_actions > 1:
                    eps = float(self.config.dirichlet_epsilon)
                    alpha = float(self.config.dirichlet_alpha)
                    alpha_t = torch.full_like(priors_mat, alpha, dtype=torch.float32)
                    gamma_dist = torch.distributions.Gamma(alpha_t, torch.ones_like(priors_mat))
                    noise = gamma_dist.sample() * valid_mask.to(torch.float32)
                    noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    mixed = (1.0 - eps) * priors_mat + eps * noise
                    apply_mask = counts.gt(1).view(-1, 1)
                    priors_mat = torch.where(apply_mask, mixed, priors_mat)

            child_batch = batch_apply_moves_compat(state, action_codes_all, parent_indices_all)
            if self._child_eval_mode == "full":
                _, _, _, _, child_values = self._evaluate_batch(child_batch)
            else:
                child_values = self._evaluate_values_only(child_batch)
            parent_player = state.current_player.index_select(0, parent_indices_all)
            child_leaf_values = self._child_values_to_parent_perspective(
                child_values=child_values,
                parent_players=parent_player,
                child_players=child_batch.current_player,
            )
            terminal_child = self._terminal_mask_from_next_state(child_batch)
            if bool(terminal_child.any().item()):
                soft_from_black = self._soft_tan_from_board_black(
                    child_batch.board,
                    soft_value_k=float(self.config.soft_value_k),
                ).to(torch.float32)
                parent_player = parent_player.to(torch.float32)
                parent_sign = torch.where(
                    parent_player.ge(0.0),
                    torch.ones_like(parent_player),
                    -torch.ones_like(parent_player),
                )
                terminal_leaf_values = soft_from_black * parent_sign
                child_leaf_values = torch.where(
                    terminal_child,
                    terminal_leaf_values,
                    child_leaf_values,
                )
                self._terminal_soft_override_count += int(terminal_child.sum().item())

            leaf_mat = torch.zeros(
                (num_roots, max_actions), dtype=torch.float32, device=state.device
            )
            if int(flat_indices.numel()) > 0:
                leaf_mat.view(-1).index_copy_(0, flat_indices, child_leaf_values)

            sims = max(1, int(self.config.num_simulations))
            with self._nvtx_range("v1.root_puct_allocate_visits"), self._timed("root_puct_ms"):
                visits_cpp, value_sum_cpp, _ = v0_core.root_puct_allocate_visits(
                    priors_mat,
                    leaf_mat,
                    valid_mask,
                    sims,
                    float(self.config.exploration_weight),
                )
            visits = visits_cpp.to(dtype=torch.float32, device=state.device)
            value_sum = value_sum_cpp.to(dtype=torch.float32, device=state.device)

            root_temps = temp_values.index_select(0, valid_root_indices)
            sample_moves = bool(self.config.sample_moves and max_actions > 1)
            with self._nvtx_range("v1.root_finalize_from_visits"), self._timed("pack_writeback_ms"):
                (
                    policy_dense,
                    chosen_action_indices,
                    chosen_action_codes,
                    chosen_valid_mask,
                    root_value_vec,
                ) = self._root_finalize_from_visits(
                    legal_index_mat=legal_index_mat,
                    action_code_mat=action_code_mat,
                    valid_mask=valid_mask,
                    visits=visits,
                    value_sum=value_sum,
                    valid_root_indices=valid_root_indices,
                    batch_size=batch_size,
                    root_temps=root_temps,
                    sample_moves=False,
                )
            if sample_moves:
                with self._nvtx_range("v1.root_sample_outside_graph"), self._timed("pack_writeback_ms"):
                    legal_policy = self._stable_legal_policy_from_visits(
                        visits=visits,
                        valid_mask=valid_mask,
                        root_temps=root_temps,
                    )
                    local_picks = torch.multinomial(legal_policy, num_samples=1).view(-1)
                    sampled_indices_local = legal_index_mat.gather(1, local_picks.view(-1, 1)).view(-1)
                    sampled_codes_local = action_code_mat.gather(
                        1,
                        local_picks.view(-1, 1, 1).expand(-1, 1, 4),
                    ).view(-1, 4)
                    chosen_action_indices.index_copy_(0, valid_root_indices, sampled_indices_local)
                    chosen_action_codes.index_copy_(0, valid_root_indices, sampled_codes_local)
            if force_uniform_mask is not None and bool(force_uniform_mask.any().item()):
                force_local_mask = force_uniform_mask.index_select(0, valid_root_indices)
                force_local_idx = torch.where(force_local_mask)[0]
                if int(force_local_idx.numel()) > 0:
                    force_valid_mask = valid_mask.index_select(0, force_local_idx).to(torch.float32)
                    force_policy = force_valid_mask / force_valid_mask.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                    force_picks = torch.multinomial(force_policy, num_samples=1).view(-1)
                    force_indices = legal_index_mat.index_select(0, force_local_idx).gather(
                        1, force_picks.view(-1, 1)
                    ).view(-1)
                    force_codes = action_code_mat.index_select(0, force_local_idx).gather(
                        1,
                        force_picks.view(-1, 1, 1).expand(-1, 1, 4),
                    ).view(-1, 4)
                    force_global_roots = valid_root_indices.index_select(0, force_local_idx)
                    chosen_action_indices.index_copy_(0, force_global_roots, force_indices)
                    chosen_action_codes.index_copy_(0, force_global_roots, force_codes)
                    chosen_valid_mask.index_fill_(0, force_global_roots, True)
                    self._forced_uniform_pick_count += int(force_local_idx.numel())
            root_values.index_copy_(0, valid_root_indices, root_value_vec)

        return RootSearchBatchOutput(
            model_input=model_input.detach(),
            legal_mask=legal_mask_bool.detach(),
            policy_dense=policy_dense.detach(),
            root_value=root_values.detach(),
            terminal_mask=terminal_mask.detach(),
            chosen_action_indices=chosen_action_indices.detach(),
            chosen_action_codes=chosen_action_codes.detach(),
            chosen_valid_mask=chosen_valid_mask.detach(),
        )

    def search(
        self,
        state: GpuStateBatch,
        *,
        temperature: Optional[float] = None,
        add_dirichlet_noise: Optional[bool] = None,
    ) -> RootSearchOutput:
        if state.batch_size != 1:
            raise ValueError("V1RootMCTS.search currently supports a single root state.")
        batch_out = self.search_batch(
            state,
            temperatures=self.config.temperature if temperature is None else float(temperature),
            add_dirichlet_noise=add_dirichlet_noise,
        )
        terminal = bool(batch_out.terminal_mask[0].item())
        chosen_valid = bool(batch_out.chosen_valid_mask[0].item())
        chosen_index: Optional[int] = (
            int(batch_out.chosen_action_indices[0].item()) if chosen_valid else None
        )
        chosen_code: Optional[torch.Tensor] = (
            batch_out.chosen_action_codes[0].detach() if chosen_valid else None
        )
        return RootSearchOutput(
            model_input=batch_out.model_input[0].detach(),
            legal_mask=batch_out.legal_mask[0].detach(),
            policy_dense=batch_out.policy_dense[0].detach(),
            root_value=float(batch_out.root_value[0].item()),
            terminal=terminal,
            chosen_action_index=chosen_index,
            chosen_action_code=chosen_code,
        )
