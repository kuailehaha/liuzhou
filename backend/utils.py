from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from src.game_state import GameState, Phase, Player

Coordinate = Tuple[int, int]


def _pos_to_list(position: Coordinate) -> List[int]:
    return [int(position[0]), int(position[1])]


def _maybe_pos_to_list(position: Optional[Coordinate]) -> Optional[List[int]]:
    return None if position is None else _pos_to_list(position)


def _list_to_pos(position: Optional[Iterable[int]]) -> Optional[Coordinate]:
    if position is None:
        return None
    items = list(position)
    if len(items) != 2:
        raise ValueError(f"Positions must have exactly two entries, received {items}")
    return int(items[0]), int(items[1])


def move_to_key(item: Any) -> Any:
    """
    Deterministic, hashable representation of a move dictionary.

    Matches the helper used inside ``src.mcts`` but duplicated here to keep
    the backend lightweight and avoid importing the full MCTS module just
    for this utility.
    """
    if isinstance(item, dict):
        return tuple(sorted((key, move_to_key(value)) for key, value in item.items()))
    if isinstance(item, list):
        return tuple(move_to_key(value) for value in item)
    if isinstance(item, tuple):
        return tuple(move_to_key(value) for value in item)
    if isinstance(item, Phase):
        return item.name
    return item


def serialize_move(move: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Prepare a move dictionary for JSON responses.
    """
    result: Dict[str, Any] = {}
    for key, value in move.items():
        if key == "phase" and isinstance(value, Phase):
            result[key] = value.name
        elif isinstance(value, tuple):
            result[key] = _pos_to_list(value)
        elif isinstance(value, list):
            result[key] = [_pos_to_list(tuple(v)) if isinstance(v, tuple) else v for v in value]
        else:
            result[key] = value
    return result


def serialize_moves(moves: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [serialize_move(move) for move in moves]


def deserialize_move(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert a JSON move payload coming from the UI back into the internal format.
    """
    if "phase" not in payload:
        raise ValueError("Move payload missing 'phase'")
    try:
        phase = Phase[payload["phase"]]
    except KeyError as exc:
        raise ValueError(f"Unknown phase: {payload['phase']}") from exc

    move: Dict[str, Any] = {"phase": phase}
    for key, value in payload.items():
        if key == "phase":
            continue
        if value is None:
            continue
        if key.endswith("position") or key in {"position", "from_position", "to_position"}:
            move[key] = _list_to_pos(value)  # type: ignore[arg-type]
        else:
            move[key] = value
    return move


def serialize_game_state(state: GameState) -> Dict[str, Any]:
    """
    Convert ``GameState`` into a JSON-friendly structure.
    """
    board = [row[:] for row in state.board]
    marked_black = sorted(_pos_to_list(pos) for pos in state.marked_black)
    marked_white = sorted(_pos_to_list(pos) for pos in state.marked_white)

    winner = state.get_winner()

    return {
        "board": board,
        "phase": state.phase.name,
        "currentPlayer": state.current_player.name,
        "marked": {
            "BLACK": marked_black,
            "WHITE": marked_white,
        },
        "pending": {
            "marksRemaining": state.pending_marks_remaining,
            "marksRequired": state.pending_marks_required,
            "capturesRemaining": state.pending_captures_remaining,
            "capturesRequired": state.pending_captures_required,
        },
        "forcedRemovalsDone": state.forced_removals_done,
        "moveCount": state.move_count,
        "isBoardFull": state.is_board_full(),
        "isGameOver": state.is_game_over(),
        "winner": None if winner is None else winner.name,
    }


def build_game_payload(
    game_id: str,
    state: GameState,
    legal_moves: Iterable[Mapping[str, Any]],
    ai_moves: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Combine the serialised game state with auxiliary data for API responses.
    """
    payload: Dict[str, Any] = {
        "gameId": game_id,
        "state": serialize_game_state(state),
        "legalMoves": serialize_moves(legal_moves),
    }
    if ai_moves:
        payload["aiMoves"] = serialize_moves(ai_moves)
    return payload
