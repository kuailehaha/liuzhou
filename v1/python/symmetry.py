"""Exact D4 transforms for 6x6 states and the shared 220-d action space."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch

from src.game_state import GameState
from src.policy_batch import (
    AUXILIARY_DIM,
    DIRECTIONS,
    MOVEMENT_DIM,
    PLACEMENT_DIM,
    SELECTION_DIM,
    TOTAL_DIM,
)

D4_TRANSFORM_COUNT = 8
SYMMETRY_AUGMENTATION_NONE = "none"
SYMMETRY_AUGMENTATION_RANDOM_D4 = "random_d4"


def normalize_symmetry_augmentation(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {
        SYMMETRY_AUGMENTATION_NONE,
        SYMMETRY_AUGMENTATION_RANDOM_D4,
    }:
        raise ValueError(
            f"Unsupported symmetry augmentation {value!r}; "
            "expected none or random_d4."
        )
    return normalized


def transform_coord(
    row: int,
    col: int,
    transform_id: int,
    *,
    board_size: int = GameState.BOARD_SIZE,
) -> tuple[int, int]:
    transform = int(transform_id)
    if transform < 0 or transform >= D4_TRANSFORM_COUNT:
        raise ValueError(f"transform_id must be in [0, 7], got {transform}")
    r, c = int(row), int(col)
    if transform >= 4:
        c = board_size - 1 - c
    for _ in range(transform % 4):
        r, c = c, board_size - 1 - r
    return r, c


@lru_cache(maxsize=None)
def d4_action_permutation(
    transform_id: int,
    board_size: int = GameState.BOARD_SIZE,
) -> tuple[int, ...]:
    if int(board_size) * int(board_size) != PLACEMENT_DIM:
        raise ValueError("D4 action mapping requires the production 6x6 board.")
    direction_to_index = {delta: index for index, delta in enumerate(DIRECTIONS)}
    permutation = list(range(TOTAL_DIM))

    for cell in range(PLACEMENT_DIM):
        row, col = divmod(cell, board_size)
        out_row, out_col = transform_coord(
            row, col, transform_id, board_size=board_size
        )
        out_cell = out_row * board_size + out_col
        permutation[cell] = out_cell
        permutation[PLACEMENT_DIM + MOVEMENT_DIM + cell] = (
            PLACEMENT_DIM + MOVEMENT_DIM + out_cell
        )

        for direction_index, (delta_row, delta_col) in enumerate(DIRECTIONS):
            dest_row, dest_col = row + delta_row, col + delta_col
            source_action = PLACEMENT_DIM + cell * len(DIRECTIONS) + direction_index
            if not (
                0 <= dest_row < board_size and 0 <= dest_col < board_size
            ):
                # Off-board movement slots are never legal, but still need a
                # stable bijection across the full 220-dimensional vector.
                transformed_delta = transform_direction(
                    delta_row,
                    delta_col,
                    transform_id,
                    board_size=board_size,
                )
                transformed_direction = direction_to_index[transformed_delta]
                permutation[source_action] = (
                    PLACEMENT_DIM
                    + out_cell * len(DIRECTIONS)
                    + transformed_direction
                )
                continue
            out_dest_row, out_dest_col = transform_coord(
                dest_row,
                dest_col,
                transform_id,
                board_size=board_size,
            )
            transformed_delta = (
                out_dest_row - out_row,
                out_dest_col - out_col,
            )
            transformed_direction = direction_to_index[transformed_delta]
            permutation[source_action] = (
                PLACEMENT_DIM
                + out_cell * len(DIRECTIONS)
                + transformed_direction
            )

    auxiliary_start = PLACEMENT_DIM + MOVEMENT_DIM + SELECTION_DIM
    for offset in range(AUXILIARY_DIM):
        permutation[auxiliary_start + offset] = auxiliary_start + offset
    if sorted(permutation) != list(range(TOTAL_DIM)):
        raise RuntimeError(f"D4 action transform {transform_id} is not bijective.")
    return tuple(permutation)


def transform_direction(
    delta_row: int,
    delta_col: int,
    transform_id: int,
    *,
    board_size: int = GameState.BOARD_SIZE,
) -> tuple[int, int]:
    origin = (board_size // 2, board_size // 2)
    destination = (origin[0] + int(delta_row), origin[1] + int(delta_col))
    out_origin = transform_coord(
        *origin, transform_id, board_size=board_size
    )
    out_destination = transform_coord(
        *destination, transform_id, board_size=board_size
    )
    return (
        out_destination[0] - out_origin[0],
        out_destination[1] - out_origin[1],
    )


@lru_cache(maxsize=None)
def d4_spatial_permutation(
    transform_id: int,
    board_size: int = GameState.BOARD_SIZE,
) -> tuple[int, ...]:
    cells = board_size * board_size
    permutation = [0] * cells
    for cell in range(cells):
        row, col = divmod(cell, board_size)
        out_row, out_col = transform_coord(
            row, col, transform_id, board_size=board_size
        )
        permutation[cell] = out_row * board_size + out_col
    if sorted(permutation) != list(range(cells)):
        raise RuntimeError(f"D4 spatial transform {transform_id} is not bijective.")
    return tuple(permutation)


def transform_training_batch(
    states: torch.Tensor,
    legal_masks: torch.Tensor,
    policy_targets: torch.Tensor,
    transform_ids: torch.Tensor | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if states.dim() != 4 or tuple(states.shape[-2:]) != (6, 6):
        raise ValueError(f"states must be [B,C,6,6], got {tuple(states.shape)}")
    batch_size = int(states.size(0))
    if tuple(legal_masks.shape) != (batch_size, TOTAL_DIM):
        raise ValueError("legal_masks must be [B,220]")
    if tuple(policy_targets.shape) != (batch_size, TOTAL_DIM):
        raise ValueError("policy_targets must be [B,220]")
    ids = torch.as_tensor(transform_ids, dtype=torch.long, device=states.device).view(-1)
    if int(ids.numel()) != batch_size:
        raise ValueError("transform_ids must contain one id per sample")
    if bool(((ids < 0) | (ids >= D4_TRANSFORM_COUNT)).any().item()):
        raise ValueError("transform_ids must be in [0,7]")

    out_states = torch.empty_like(states)
    out_masks = torch.empty_like(legal_masks)
    out_policy = torch.empty_like(policy_targets)
    for transform in range(D4_TRANSFORM_COUNT):
        rows = torch.nonzero(ids.eq(transform), as_tuple=False).view(-1)
        if int(rows.numel()) == 0:
            continue
        spatial = torch.tensor(
            d4_spatial_permutation(transform),
            dtype=torch.long,
            device=states.device,
        )
        actions = torch.tensor(
            d4_action_permutation(transform),
            dtype=torch.long,
            device=legal_masks.device,
        )
        selected_states = states.index_select(0, rows).flatten(2)
        transformed_states = torch.empty_like(selected_states)
        transformed_states[:, :, spatial] = selected_states
        out_states.index_copy_(
            0,
            rows,
            transformed_states.view(-1, states.size(1), 6, 6),
        )

        selected_masks = legal_masks.index_select(0, rows)
        transformed_masks = torch.empty_like(selected_masks)
        transformed_masks[:, actions] = selected_masks
        out_masks.index_copy_(0, rows, transformed_masks)

        selected_policy = policy_targets.index_select(0, rows)
        transformed_policy = torch.empty_like(selected_policy)
        transformed_policy[:, actions] = selected_policy
        out_policy.index_copy_(0, rows, transformed_policy)
    return out_states, out_masks, out_policy


def transform_game_state(state: GameState, transform_id: int) -> GameState:
    board = [[0] * state.BOARD_SIZE for _ in range(state.BOARD_SIZE)]
    for row in range(state.BOARD_SIZE):
        for col in range(state.BOARD_SIZE):
            out_row, out_col = transform_coord(row, col, transform_id)
            board[out_row][out_col] = state.board[row][col]

    def transform_marks(values):
        return {transform_coord(row, col, transform_id) for row, col in values}

    return GameState(
        board=board,
        phase=state.phase,
        current_player=state.current_player,
        marked_black=transform_marks(state.marked_black),
        marked_white=transform_marks(state.marked_white),
        forced_removals_done=state.forced_removals_done,
        move_count=state.move_count,
        pending_marks_required=state.pending_marks_required,
        pending_marks_remaining=state.pending_marks_remaining,
        pending_captures_required=state.pending_captures_required,
        pending_captures_remaining=state.pending_captures_remaining,
        moves_since_capture=state.moves_since_capture,
    )
