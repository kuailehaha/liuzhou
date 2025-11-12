"""
Move encoding helpers for v0 tensorized self-play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from src.game_state import Phase

from . import fast_legal_mask as _fast_mask
from .rules_tensor import (
    MovementMasks,
    generate_movement_moves,
    generate_placement_moves,
    generate_process_removal_mask,
    generate_selection_mask,
)
from .state_batch import TensorStateBatch

ACTION_KIND_PLACEMENT = 1
ACTION_KIND_MOVEMENT = 2
ACTION_KIND_MARK_SELECTION = 3
ACTION_KIND_CAPTURE_SELECTION = 4
ACTION_KIND_FORCED_REMOVAL_SELECTION = 5
ACTION_KIND_COUNTER_REMOVAL_SELECTION = 6
ACTION_KIND_NO_MOVES_REMOVAL_SELECTION = 7
ACTION_KIND_PROCESS_REMOVAL = 8


@dataclass(frozen=True)
class ActionEncodingSpec:
    placement_dim: int
    movement_dim: int
    selection_dim: int
    auxiliary_dim: int

    @property
    def total_dim(self) -> int:
        return self.placement_dim + self.movement_dim + self.selection_dim + self.auxiliary_dim


DEFAULT_ACTION_SPEC = ActionEncodingSpec(
    placement_dim=36,
    movement_dim=36 * 4,
    selection_dim=36,
    auxiliary_dim=4,
)


def encode_actions_python(
    batch: TensorStateBatch,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
    *,
    return_metadata: bool = False,
) -> Union[torch.BoolTensor, Tuple[torch.BoolTensor, torch.Tensor]]:
    device = batch.board.device
    B = batch.batch_size
    board_size = batch.config.board_size
    cells = board_size * board_size
    dirs = 4

    if spec.placement_dim != cells or spec.selection_dim != cells or spec.movement_dim != cells * dirs:
        raise ValueError(
            f"ActionEncodingSpec does not match board size {board_size}: "
            f"expected placement={cells}, movement={cells*dirs}, selection={cells}."
        )

    mask = torch.zeros((B, spec.total_dim), dtype=torch.bool, device=device)

    placement_mask = generate_placement_moves(batch).view(B, cells)
    movement_masks: MovementMasks = generate_movement_moves(batch)
    movement_mask = movement_masks.move.view(B, cells * dirs)
    selection_mask = generate_selection_mask(batch).view(B, cells)
    process_mask = generate_process_removal_mask(batch).view(B, 1)

    placement_slice = slice(0, spec.placement_dim)
    movement_slice = slice(spec.placement_dim, spec.placement_dim + spec.movement_dim)
    selection_slice = slice(movement_slice.stop, movement_slice.stop + spec.selection_dim)
    auxiliary_slice = slice(selection_slice.stop, selection_slice.stop + spec.auxiliary_dim)

    mask[:, placement_slice] = placement_mask
    mask[:, movement_slice] = movement_mask
    mask[:, selection_slice] = selection_mask
    mask[:, auxiliary_slice.start : auxiliary_slice.start + 1] = process_mask

    if not return_metadata:
        return mask

    metadata = _build_metadata_from_mask(batch, mask, spec, board_size=board_size)
    return mask, metadata


def encode_actions_fast(
    batch: TensorStateBatch,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
    *,
    return_metadata: bool = False,
):
    result = _fast_mask.encode_actions_fast(batch, spec, return_metadata=return_metadata)
    if result is None:
        return encode_actions_python(batch, spec, return_metadata=return_metadata)
    return result


def _build_metadata_from_mask(
    batch: TensorStateBatch,
    mask: torch.BoolTensor,
    spec: ActionEncodingSpec,
    *,
    board_size: int,
) -> torch.Tensor:
    B = batch.batch_size
    metadata = torch.full((B, spec.total_dim, 4), -1, dtype=torch.int32, device=mask.device)

    placement_slice = slice(0, spec.placement_dim)
    movement_slice = slice(spec.placement_dim, spec.placement_dim + spec.movement_dim)
    selection_slice = slice(movement_slice.stop, movement_slice.stop + spec.selection_dim)
    auxiliary_slice = slice(selection_slice.stop, selection_slice.stop + spec.auxiliary_dim)

    placement_coords = torch.nonzero(mask[:, placement_slice], as_tuple=False)
    if placement_coords.numel() > 0:
        batch_idx = placement_coords[:, 0]
        flat = placement_coords[:, 1]
        metadata[batch_idx, flat, 0] = ACTION_KIND_PLACEMENT
        metadata[batch_idx, flat, 1] = flat

    move_coords = torch.nonzero(mask[:, movement_slice], as_tuple=False)
    if move_coords.numel() > 0:
        dirs = 4
        batch_idx = move_coords[:, 0]
        rel = move_coords[:, 1]
        cell_idx = rel // dirs
        dir_idx = rel % dirs
        metadata[batch_idx, movement_slice.start + rel, 0] = ACTION_KIND_MOVEMENT
        metadata[batch_idx, movement_slice.start + rel, 1] = cell_idx
        metadata[batch_idx, movement_slice.start + rel, 2] = dir_idx

    selection_coords = torch.nonzero(mask[:, selection_slice], as_tuple=False)
    if selection_coords.numel() > 0:
        batch_idx = selection_coords[:, 0]
        rel = selection_coords[:, 1]
        flat_idx = selection_slice.start + rel
        metadata[batch_idx, flat_idx, 0] = ACTION_KIND_MARK_SELECTION
        metadata[batch_idx, flat_idx, 1] = rel

    process_coords = torch.nonzero(mask[:, auxiliary_slice], as_tuple=False)
    if process_coords.numel() > 0:
        batch_idx = process_coords[:, 0]
        rel = process_coords[:, 1]
        flat_idx = auxiliary_slice.start + rel
        metadata[batch_idx, flat_idx, 0] = ACTION_KIND_PROCESS_REMOVAL

    return metadata


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(DIRECTIONS)}


def decode_action_indices(
    indices: torch.Tensor,
    batch: TensorStateBatch,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> List[Optional[dict]]:
    board_size = batch.config.board_size
    cells = board_size * board_size
    dirs = len(DIRECTIONS)

    placement_end = spec.placement_dim
    movement_end = placement_end + spec.movement_dim
    selection_end = movement_end + spec.selection_dim
    total = spec.total_dim

    decoded: List[Optional[dict]] = []
    for idx, raw in enumerate(indices.tolist()):
        if raw is None:
            decoded.append(None)
            continue
        action = int(raw)
        if action < 0 or action >= total:
            decoded.append(None)
            continue
        phase = Phase(int(batch.phase[idx].item()))

        if action < placement_end:
            cell = action
            r = cell // board_size
            c = cell % board_size
            move = {
                "phase": Phase.PLACEMENT,
                "action_type": "place",
                "position": (r, c),
            }
        elif action < movement_end:
            rel = action - placement_end
            cell_idx, dir_idx = divmod(rel, dirs)
            r_from = cell_idx // board_size
            c_from = cell_idx % board_size
            dr, dc = DIRECTIONS[dir_idx]
            move = {
                "phase": Phase.MOVEMENT,
                "action_type": "move",
                "from_position": (r_from, c_from),
                "to_position": (r_from + dr, c_from + dc),
            }
        elif action < selection_end:
            rel = action - movement_end
            r = rel // board_size
            c = rel % board_size
            if phase == Phase.MARK_SELECTION:
                action_type = "mark"
                target_phase = Phase.MARK_SELECTION
            elif phase == Phase.CAPTURE_SELECTION:
                action_type = "capture"
                target_phase = Phase.CAPTURE_SELECTION
            elif phase == Phase.FORCED_REMOVAL:
                action_type = "remove"
                target_phase = Phase.FORCED_REMOVAL
            elif phase == Phase.COUNTER_REMOVAL:
                action_type = "counter_remove"
                target_phase = Phase.COUNTER_REMOVAL
            elif phase == Phase.MOVEMENT:
                action_type = "no_moves_remove"
                target_phase = Phase.MOVEMENT
            else:
                action_type = "select"
                target_phase = phase
            move = {
                "phase": target_phase,
                "action_type": action_type,
                "position": (r, c),
            }
        else:
            aux_index = action - selection_end
            if aux_index == 0:
                move = {
                    "phase": Phase.REMOVAL,
                    "action_type": "process_removal",
                }
            else:
                move = None
        decoded.append(move)
    return decoded


def action_to_index(
    action: dict,
    board_size: int,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> Optional[int]:
    placement_end = spec.placement_dim
    movement_end = placement_end + spec.movement_dim
    selection_end = movement_end + spec.selection_dim

    phase = action.get("phase")
    action_type = action.get("action_type")

    if phase == Phase.PLACEMENT and action_type == "place":
        r, c = action["position"]
        return r * board_size + c

    if phase == Phase.MOVEMENT and action_type == "move":
        (r_from, c_from) = action["from_position"]
        (r_to, c_to) = action["to_position"]
        dr = r_to - r_from
        dc = c_to - c_from
        dir_idx = DIR_TO_INDEX.get((dr, dc))
        if dir_idx is None:
            return None
        cell_idx = r_from * board_size + c_from
        return placement_end + cell_idx * len(DIRECTIONS) + dir_idx

    selection_offset = placement_end + spec.movement_dim
    if action_type in {"mark", "capture", "remove", "counter_remove", "no_moves_remove"}:
        r, c = action["position"]
        return selection_offset + r * board_size + c

    if action_type == "process_removal":
        return selection_end

    return None
