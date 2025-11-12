"""
Tensor-friendly rule utilities for v0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

from src.game_state import GameState, Phase, Player
from src.rule_engine import (
    apply_capture_selection,
    apply_counter_removal_phase3,
    apply_forced_removal,
    apply_mark_selection,
    apply_movement_move,
    apply_placement_move,
    generate_capture_targets,
    generate_mark_targets,
    generate_movement_moves as legacy_generate_movement_moves,
    generate_placement_positions,
    handle_no_moves_phase3,
    has_legal_moves_phase3,
    is_piece_in_shape,
    process_phase2_removals,
)

from .state_batch import TensorStateBatch, from_game_states, to_game_states

_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(_DIRECTIONS)}


def _states_to_batch(states: Sequence[GameState], template: TensorStateBatch) -> TensorStateBatch:
    new_batch = from_game_states(states, device=template.board.device)
    if template.mask_alive is not None:
        new_batch.mask_alive = template.mask_alive.clone()
    return new_batch


def _prefer_normal_pieces(
    board: List[List[int]],
    positions: Iterable[Tuple[int, int]],
    player_value: int,
) -> List[Tuple[int, int]]:
    normal: List[Tuple[int, int]] = []
    all_positions: List[Tuple[int, int]] = []
    for r, c in positions:
        all_positions.append((r, c))
        if not is_piece_in_shape(board, r, c, player_value, set()):
            normal.append((r, c))
    return normal or all_positions


def _forced_removal_targets(state: GameState) -> List[Tuple[int, int]]:
    if state.phase != Phase.FORCED_REMOVAL:
        return []
    if state.forced_removals_done == 0:
        target_player = Player.BLACK
    elif state.forced_removals_done == 1:
        target_player = Player.WHITE
    else:
        return []

    value = target_player.value
    board = state.board
    size = state.BOARD_SIZE

    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board[r][c] == value
    ]
    return _prefer_normal_pieces(board, candidates, value)


def _counter_removal_targets(state: GameState) -> List[Tuple[int, int]]:
    if state.phase != Phase.COUNTER_REMOVAL:
        return []
    remover = state.current_player
    stuck_player = remover.opponent()
    value = stuck_player.value
    board = state.board
    size = state.BOARD_SIZE
    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board[r][c] == value
    ]
    return _prefer_normal_pieces(board, candidates, value)


def _no_moves_removal_targets(state: GameState) -> List[Tuple[int, int]]:
    current_player = state.current_player
    opponent = current_player.opponent()
    value = opponent.value
    board = state.board
    size = state.BOARD_SIZE
    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board[r][c] == value
    ]
    return _prefer_normal_pieces(board, candidates, value)


def generate_placement_moves(batch: TensorStateBatch) -> torch.BoolTensor:
    device = batch.board.device
    B, H, W = batch.board.shape
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    states = to_game_states(batch)
    for idx, state in enumerate(states):
        if state.phase != Phase.PLACEMENT:
            continue
        positions = generate_placement_positions(state)
        if positions:
            rows, cols = zip(*positions)
            mask[idx, list(rows), list(cols)] = True
    return mask


@dataclass
class MovementMasks:
    move: torch.BoolTensor
    no_move_remove: torch.BoolTensor


def generate_movement_moves(batch: TensorStateBatch) -> MovementMasks:
    device = batch.board.device
    B, H, W = batch.board.shape
    move_mask = torch.zeros((B, H, W, len(_DIRECTIONS)), dtype=torch.bool, device=device)
    removal_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    states = to_game_states(batch)
    for idx, state in enumerate(states):
        if state.phase != Phase.MOVEMENT:
            continue
        moves = legacy_generate_movement_moves(state)
        if moves:
            for (r_from, c_from), (r_to, c_to) in moves:
                dr = r_to - r_from
                dc = c_to - c_from
                dir_idx = _DIR_TO_INDEX.get((dr, dc))
                if dir_idx is None:
                    continue
                move_mask[idx, r_from, c_from, dir_idx] = True
        else:
            for r, c in _no_moves_removal_targets(state):
                removal_mask[idx, r, c] = True
    return MovementMasks(move=move_mask, no_move_remove=removal_mask)


def generate_selection_mask(batch: TensorStateBatch) -> torch.BoolTensor:
    device = batch.board.device
    B, H, W = batch.board.shape
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    states = to_game_states(batch)
    for idx, state in enumerate(states):
        phase = state.phase
        targets: List[Tuple[int, int]] = []
        if phase == Phase.MARK_SELECTION:
            targets = generate_mark_targets(state)
        elif phase == Phase.CAPTURE_SELECTION:
            targets = generate_capture_targets(state)
        elif phase == Phase.FORCED_REMOVAL:
            targets = _forced_removal_targets(state)
        elif phase == Phase.COUNTER_REMOVAL:
            targets = _counter_removal_targets(state)
        elif phase == Phase.MOVEMENT and not has_legal_moves_phase3(state):
            targets = _no_moves_removal_targets(state)
        if targets:
            rows, cols = zip(*targets)
            mask[idx, list(rows), list(cols)] = True
    return mask


def generate_process_removal_mask(batch: TensorStateBatch) -> torch.BoolTensor:
    device = batch.board.device
    mask = torch.zeros(batch.batch_size, dtype=torch.bool, device=device)

    states = to_game_states(batch)
    for idx, state in enumerate(states):
        if state.phase == Phase.REMOVAL:
            mask[idx] = True
    return mask


def apply_movement(batch: TensorStateBatch, move_indices: torch.Tensor) -> TensorStateBatch:
    states = to_game_states(batch)
    size = batch.config.board_size

    for idx, state in enumerate(states):
        if batch.mask_alive is not None and not bool(batch.mask_alive[idx]):
            continue
        if state.phase != Phase.MOVEMENT:
            continue
        action = int(move_indices[idx].item())
        if action < 0:
            continue
        cell_idx, dir_idx = divmod(action, len(_DIRECTIONS))
        r_from = cell_idx // size
        c_from = cell_idx % size
        dr, dc = _DIRECTIONS[dir_idx]
        r_to = r_from + dr
        c_to = c_from + dc
        states[idx] = apply_movement_move(state, ((r_from, c_from), (r_to, c_to)), quiet=True)

    return _states_to_batch(states, batch)


def resolve_mark_or_capture(batch: TensorStateBatch, selection_indices: torch.Tensor) -> TensorStateBatch:
    states = to_game_states(batch)
    size = batch.config.board_size

    for idx, state in enumerate(states):
        if batch.mask_alive is not None and not bool(batch.mask_alive[idx]):
            continue
        action = int(selection_indices[idx].item())
        if action < 0:
            continue
        r = action // size
        c = action % size

        phase = state.phase
        if phase == Phase.MARK_SELECTION:
            states[idx] = apply_mark_selection(state, (r, c))
        elif phase == Phase.CAPTURE_SELECTION:
            states[idx] = apply_capture_selection(state, (r, c), quiet=True)
        elif phase == Phase.MOVEMENT and not has_legal_moves_phase3(state):
            states[idx] = handle_no_moves_phase3(state, (r, c), quiet=True)
        elif phase == Phase.FORCED_REMOVAL:
            states[idx] = apply_forced_removal(state, (r, c))
        elif phase == Phase.COUNTER_REMOVAL:
            states[idx] = apply_counter_removal_phase3(state, (r, c), quiet=True)

    return _states_to_batch(states, batch)


def apply_process_removal(batch: TensorStateBatch, active_mask: torch.BoolTensor) -> TensorStateBatch:
    states = to_game_states(batch)
    for idx, state in enumerate(states):
        if not bool(active_mask[idx].item()):
            continue
        if state.phase != Phase.REMOVAL:
            continue
        states[idx] = process_phase2_removals(state)
    return _states_to_batch(states, batch)
