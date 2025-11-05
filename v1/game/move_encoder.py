"""
Move encoding helpers for tensorized self-play.

This module will centralize the mapping between abstract game actions and the
fixed-size tensor representations consumed by neural networks and sampling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from src.game_state import Phase

from .rules_tensor import (
    MovementMasks,
    generate_movement_moves,
    generate_placement_moves,
    generate_process_removal_mask,
    generate_selection_mask,
)
from .state_batch import TensorStateBatch
from .fast_legal_mask import encode_actions_fast as _encode_actions_fast


@dataclass(frozen=True)
class ActionEncodingSpec:
    """
    Configuration describing how actions are flattened.

    Attributes
    ----------
    placement_dim : int
        Number of logits reserved for placement moves.
    movement_dim : int
        Number of logits dedicated to movement moves (source × direction).
    selection_dim : int
        Number of logits used for mark/capture selections.
    auxiliary_dim : int
        Slot for special actions (e.g., `process_removal`).
    """

    placement_dim: int
    movement_dim: int
    selection_dim: int
    auxiliary_dim: int

    @property
    def total_dim(self) -> int:
        return self.placement_dim + self.movement_dim + self.selection_dim + self.auxiliary_dim


DEFAULT_ACTION_SPEC = ActionEncodingSpec(
    placement_dim=36,  # 6x6 board
    movement_dim=36 * 4,  # four orthogonal moves per square
    selection_dim=36,
    auxiliary_dim=4,  # placeholder for process_removal, forced removal slots, etc.
)


def encode_actions_python(
    batch: TensorStateBatch,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> torch.BoolTensor:
    """
    Pure-Python/tensor implementation of the legal-action mask.
    """
    device = batch.board.device
    B = batch.batch_size
    board_size = batch.config.board_size
    cells = board_size * board_size
    dirs = 4  # up, down, left, right

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
    # Only first auxiliary slot is currently used for process_removal
    mask[:, auxiliary_slice.start : auxiliary_slice.start + 1] = process_mask

    return mask


def encode_actions(
    batch: TensorStateBatch,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
    *,
    use_fast: bool = True,
    return_metadata: bool = False,
) -> Union[torch.BoolTensor, Tuple[torch.BoolTensor, Optional[torch.Tensor]]]:
    """
    Produce a legal-action mask of shape (B, spec.total_dim).
    """
    device = batch.board.device

    if use_fast:
        fast_mask = None
        fast_metadata = None
        if device.type == "cpu":
            result = _encode_actions_fast(batch, spec, return_metadata=return_metadata)
        else:
            cpu_batch = batch.to(torch.device("cpu"))
            result = _encode_actions_fast(cpu_batch, spec, return_metadata=return_metadata)
        if result is not None:
            if return_metadata:
                fast_mask, fast_metadata = result
            else:
                fast_mask = result
            mask = fast_mask.to(device)
            if return_metadata:
                return mask, fast_metadata
            return mask

    mask = encode_actions_python(batch, spec)
    if return_metadata:
        return mask, None
    return mask


DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(DIRECTIONS)}


def decode_action_indices(
    indices: torch.Tensor,
    batch: TensorStateBatch,
    spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC,
) -> List[Optional[dict]]:
    """
    Map flat action indices back to structured action descriptors.
    """
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
    """
    Map a structured move dictionary (AlphaZero-style) to the flat index used in
    the tensorized encoding. Returns None if the action is not representable.
    """
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
        return selection_end  # occupy first auxiliary slot

    return None
