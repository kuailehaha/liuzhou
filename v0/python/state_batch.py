"""
Batched game-state representation for the tensorized pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch

from .tensor_utils import TensorGameConfig


@dataclass
class TensorStateBatch:
    board: torch.Tensor
    marks_black: torch.BoolTensor
    marks_white: torch.BoolTensor
    phase: torch.LongTensor
    current_player: torch.LongTensor
    pending_marks_required: torch.LongTensor
    pending_marks_remaining: torch.LongTensor
    pending_captures_required: torch.LongTensor
    pending_captures_remaining: torch.LongTensor
    forced_removals_done: torch.LongTensor
    move_count: torch.LongTensor
    mask_alive: Optional[torch.BoolTensor] = None
    config: TensorGameConfig = field(default_factory=TensorGameConfig)

    def to(self, device: torch.device) -> "TensorStateBatch":
        kwargs = {
            "board": self.board.to(device),
            "marks_black": self.marks_black.to(device),
            "marks_white": self.marks_white.to(device),
            "phase": self.phase.to(device),
            "current_player": self.current_player.to(device),
            "pending_marks_required": self.pending_marks_required.to(device),
            "pending_marks_remaining": self.pending_marks_remaining.to(device),
            "pending_captures_required": self.pending_captures_required.to(device),
            "pending_captures_remaining": self.pending_captures_remaining.to(device),
            "forced_removals_done": self.forced_removals_done.to(device),
            "move_count": self.move_count.to(device),
            "mask_alive": None if self.mask_alive is None else self.mask_alive.to(device),
            "config": TensorGameConfig(board_size=self.config.board_size, device=device),
        }
        return TensorStateBatch(**kwargs)

    @property
    def batch_size(self) -> int:
        return int(self.board.shape[0])

    def clone(self) -> "TensorStateBatch":
        return TensorStateBatch(
            board=self.board.clone(),
            marks_black=self.marks_black.clone(),
            marks_white=self.marks_white.clone(),
            phase=self.phase.clone(),
            current_player=self.current_player.clone(),
            pending_marks_required=self.pending_marks_required.clone(),
            pending_marks_remaining=self.pending_marks_remaining.clone(),
            pending_captures_required=self.pending_captures_required.clone(),
            pending_captures_remaining=self.pending_captures_remaining.clone(),
            forced_removals_done=self.forced_removals_done.clone(),
            move_count=self.move_count.clone(),
            mask_alive=None if self.mask_alive is None else self.mask_alive.clone(),
            config=TensorGameConfig(board_size=self.config.board_size, device=self.config.device),
        )


def _ensure_sequence(states) -> List:
    if isinstance(states, (list, tuple)):
        return list(states)
    if isinstance(states, Iterable):
        return list(states)
    raise TypeError(f"Unsupported states container type: {type(states)!r}")


def from_game_states(states, device: Optional[torch.device] = None) -> TensorStateBatch:
    from src.game_state import GameState

    states = _ensure_sequence(states)
    if not states:
        raise ValueError("from_game_states requires at least one GameState instance.")

    device = device or torch.device("cpu")
    board_size = states[0].BOARD_SIZE
    batch_size = len(states)

    board = torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device)
    marks_black = torch.zeros_like(board, dtype=torch.bool)
    marks_white = torch.zeros_like(board, dtype=torch.bool)
    phase = torch.zeros(batch_size, dtype=torch.long, device=device)
    current_player = torch.zeros(batch_size, dtype=torch.long, device=device)
    pending_marks_required = torch.zeros(batch_size, dtype=torch.long, device=device)
    pending_marks_remaining = torch.zeros(batch_size, dtype=torch.long, device=device)
    pending_captures_required = torch.zeros(batch_size, dtype=torch.long, device=device)
    pending_captures_remaining = torch.zeros(batch_size, dtype=torch.long, device=device)
    forced_removals_done = torch.zeros(batch_size, dtype=torch.long, device=device)
    move_count = torch.zeros(batch_size, dtype=torch.long, device=device)

    for idx, state in enumerate(states):
        if not isinstance(state, GameState):
            raise TypeError(f"Expected GameState at position {idx}, got {type(state)!r}")
        board[idx] = torch.tensor(state.board, dtype=torch.int8, device=device)

        if state.marked_black:
            rb, cb = zip(*state.marked_black)
            marks_black[idx, list(rb), list(cb)] = True
        if state.marked_white:
            rw, cw = zip(*state.marked_white)
            marks_white[idx, list(rw), list(cw)] = True

        phase[idx] = int(state.phase.value)
        current_player[idx] = int(state.current_player.value)
        pending_marks_required[idx] = int(state.pending_marks_required)
        pending_marks_remaining[idx] = int(state.pending_marks_remaining)
        pending_captures_required[idx] = int(state.pending_captures_required)
        pending_captures_remaining[idx] = int(state.pending_captures_remaining)
        forced_removals_done[idx] = int(state.forced_removals_done)
        move_count[idx] = int(state.move_count)

    mask_alive = torch.ones(batch_size, dtype=torch.bool, device=device)

    return TensorStateBatch(
        board=board,
        marks_black=marks_black,
        marks_white=marks_white,
        phase=phase,
        current_player=current_player,
        pending_marks_required=pending_marks_required,
        pending_marks_remaining=pending_marks_remaining,
        pending_captures_required=pending_captures_required,
        pending_captures_remaining=pending_captures_remaining,
        forced_removals_done=forced_removals_done,
        move_count=move_count,
        mask_alive=mask_alive,
        config=TensorGameConfig(board_size=board_size, device=device),
    )


def to_game_states(batch: TensorStateBatch):
    from src.game_state import GameState, Phase, Player

    device_cpu = torch.device("cpu")

    board_cpu = batch.board.to(device_cpu, copy=True)
    marks_black_cpu = batch.marks_black.to(device_cpu, copy=True)
    marks_white_cpu = batch.marks_white.to(device_cpu, copy=True)
    phase_cpu = batch.phase.to(device_cpu, copy=True)
    current_cpu = batch.current_player.to(device_cpu, copy=True)
    p_marks_req = batch.pending_marks_required.to(device_cpu, copy=True)
    p_marks_rem = batch.pending_marks_remaining.to(device_cpu, copy=True)
    p_caps_req = batch.pending_captures_required.to(device_cpu, copy=True)
    p_caps_rem = batch.pending_captures_remaining.to(device_cpu, copy=True)
    forced_cpu = batch.forced_removals_done.to(device_cpu, copy=True)
    move_count_cpu = batch.move_count.to(device_cpu, copy=True)

    states: List[GameState] = []

    for idx in range(batch.batch_size):
        board_list = board_cpu[idx].to(dtype=torch.int64).tolist()
        black_positions = marks_black_cpu[idx].nonzero(as_tuple=True)
        marked_black = {(int(r), int(c)) for r, c in zip(*black_positions)}
        white_positions = marks_white_cpu[idx].nonzero(as_tuple=True)
        marked_white = {(int(r), int(c)) for r, c in zip(*white_positions)}

        phase_val = Phase(int(phase_cpu[idx].item()))
        player_val = Player(int(current_cpu[idx].item()))

        state = GameState(
            board=board_list,
            phase=phase_val,
            current_player=player_val,
            marked_black=marked_black,
            marked_white=marked_white,
            forced_removals_done=int(forced_cpu[idx].item()),
            move_count=int(move_count_cpu[idx].item()),
            pending_marks_required=int(p_marks_req[idx].item()),
            pending_marks_remaining=int(p_marks_rem[idx].item()),
            pending_captures_required=int(p_caps_req[idx].item()),
            pending_captures_remaining=int(p_caps_rem[idx].item()),
        )
        states.append(state)

    return states
