from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import torch

import v0_core as core
from src.game_state import GameState as PyGameState
from src.game_state import Phase as PyPhase
from src.game_state import Player as PyPlayer

MoveType = Dict[str, Any]

_ACTION_TYPE_TO_STRING = {
    core.ActionType.PLACE: "place",
    core.ActionType.MOVE: "move",
    core.ActionType.MARK: "mark",
    core.ActionType.CAPTURE: "capture",
    core.ActionType.FORCED_REMOVAL: "remove",
    core.ActionType.COUNTER_REMOVAL: "counter_remove",
    core.ActionType.NO_MOVES_REMOVAL: "no_moves_remove",
    core.ActionType.PROCESS_REMOVAL: "process_removal",
}

_STRING_TO_FACTORY = {
    "place": core.MoveRecord.placement,
    "mark": core.MoveRecord.mark,
    "capture": core.MoveRecord.capture,
    "remove": core.MoveRecord.forced_removal,
    "counter_remove": core.MoveRecord.counter_removal,
    "no_moves_remove": core.MoveRecord.no_moves_removal,
}


def generate_all_legal_moves(
    state: Union[core.GameState, PyGameState],
    *,
    return_codes: bool = False,
) -> Union[List[MoveType], Tuple[List[MoveType], torch.Tensor]]:
    """
    Drop-in replacement for ``src.move_generator.generate_all_legal_moves`` backed by C++.
    """
    core_state, _ = _ensure_core_state(state)
    if return_codes:
        moves, codes = core.generate_moves_with_codes(core_state)
        action_tensor = _action_codes_to_tensor(codes)
    else:
        moves = core.generate_all_legal_moves_struct(core_state)
        action_tensor = None
    move_dicts = [_core_move_to_py_dict(move) for move in moves]
    if return_codes:
        return move_dicts, action_tensor
    return move_dicts


def generate_all_legal_moves_with_codes(
    state: Union[core.GameState, PyGameState],
) -> Tuple[List[MoveType], torch.Tensor]:
    return generate_all_legal_moves(state, return_codes=True)  # type: ignore[return-value]


def apply_move(
    state: Union[core.GameState, PyGameState],
    move: MoveType,
    quiet: bool = False,
) -> Union[core.GameState, PyGameState]:
    """
    Apply a move dictionary using the C++ rule helpers.
    Returns a GameState matching the input type (core or src).
    """
    core_state, was_core = _ensure_core_state(state)
    move_record = _dict_to_core_move(move)
    next_state = core.apply_move_struct(core_state, move_record, quiet=quiet)
    if was_core:
        return next_state
    return _core_to_py_state(next_state)


def _generate_moves_forced_removal(
    state: Union[core.GameState, PyGameState],
) -> List[MoveType]:
    core_state, _ = _ensure_core_state(state)
    return [_core_move_to_py_dict(m) for m in core.generate_forced_removal_moves_struct(core_state)]


def _generate_moves_no_moves(
    state: Union[core.GameState, PyGameState],
) -> List[MoveType]:
    core_state, _ = _ensure_core_state(state)
    return [_core_move_to_py_dict(m) for m in core.generate_no_moves_options_struct(core_state)]


def _generate_moves_counter_removal(
    state: Union[core.GameState, PyGameState],
) -> List[MoveType]:
    core_state, _ = _ensure_core_state(state)
    return [_core_move_to_py_dict(m) for m in core.generate_counter_removal_moves_struct(core_state)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_core_state(state: Union[core.GameState, PyGameState]) -> Tuple[core.GameState, bool]:
    if isinstance(state, core.GameState):
        return state, True
    if isinstance(state, PyGameState):
        return _py_to_core_state(state), False
    raise TypeError(f"Unsupported GameState type: {type(state)!r}")


def _py_to_core_state(state: PyGameState) -> core.GameState:
    core_state = core.GameState()
    core_state.board = [row[:] for row in state.board]
    core_state.phase = core.Phase(state.phase.value)
    core_state.current_player = core.Player(state.current_player.value)
    core_state.marked_black = list(state.marked_black)
    core_state.marked_white = list(state.marked_white)
    core_state.forced_removals_done = state.forced_removals_done
    core_state.move_count = state.move_count
    core_state.pending_marks_required = state.pending_marks_required
    core_state.pending_marks_remaining = state.pending_marks_remaining
    core_state.pending_captures_required = state.pending_captures_required
    core_state.pending_captures_remaining = state.pending_captures_remaining
    return core_state


def _core_to_py_state(state: core.GameState) -> PyGameState:
    board = [row[:] for row in state.board]
    marked_black = set(tuple(pos) for pos in state.marked_black)
    marked_white = set(tuple(pos) for pos in state.marked_white)
    py_state = PyGameState(
        board=board,
        phase=PyPhase(int(state.phase.value)),
        current_player=PyPlayer(int(state.current_player.value)),
        marked_black=marked_black,
        marked_white=marked_white,
        forced_removals_done=state.forced_removals_done,
        move_count=state.move_count,
        pending_marks_required=state.pending_marks_required,
        pending_marks_remaining=state.pending_marks_remaining,
        pending_captures_required=state.pending_captures_required,
        pending_captures_remaining=state.pending_captures_remaining,
    )
    return py_state


def _core_move_to_py_dict(move: core.MoveRecord) -> MoveType:
    phase = PyPhase(int(move.phase.value))
    action_name = _ACTION_TYPE_TO_STRING.get(move.action_type)
    if action_name is None:
        raise ValueError(f"Unknown action type: {move.action_type}")

    if action_name == "move":
        return {
            "phase": phase,
            "action_type": "move",
            "from_position": tuple(move.from_position),
            "to_position": tuple(move.to_position),
        }
    if action_name == "process_removal":
        return {"phase": phase, "action_type": "process_removal"}

    position = move.position
    if position is None:
        raise ValueError("Expected position for non-movement action.")
    return {
        "phase": phase,
        "action_type": action_name,
        "position": tuple(position),
    }


def _dict_to_core_move(move: MoveType) -> core.MoveRecord:
    action_type = move.get("action_type")
    if action_type == "process_removal":
        return core.MoveRecord.process_removal()
    if action_type == "move":
        return core.MoveRecord.movement(
            tuple(move["from_position"]),
            tuple(move["to_position"]),
        )
    factory = _STRING_TO_FACTORY.get(action_type)
    if factory is None:
        raise ValueError(f"Unsupported action_type: {action_type}")
    position = move.get("position")
    if position is None:
        raise ValueError(f"Action {action_type} requires 'position'.")
    return factory(tuple(position))


def _action_codes_to_tensor(codes: Sequence[core.ActionCode]) -> torch.Tensor:
    if not codes:
        return torch.zeros((0, 4), dtype=torch.int32)
    data = [[c.kind, c.primary, c.secondary, c.extra] for c in codes]
    return torch.tensor(data, dtype=torch.int32)
