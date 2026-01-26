"""Helpers for serializing GameState objects and training samples to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from src.game_state import GameState, Phase, Player

__all__ = [
    "state_to_dict",
    "state_from_dict",
    "sample_to_record",
    "flatten_training_games",
    "iter_sample_records",
    "load_examples_from_files",
    "write_records_to_jsonl",
]


Coord = Tuple[int, int]
# 5-tuple: (GameState, policy, legal_moves, value, soft_value)
SampleTuple = Tuple[GameState, np.ndarray, List[dict], float, float]


def _marks_to_list(marks: Iterable[Coord]) -> List[Tuple[int, int]]:
    return [(int(r), int(c)) for r, c in marks]


def _marks_from_list(items: Sequence[Sequence[int]]) -> set:
    return {(int(pair[0]), int(pair[1])) for pair in items}


def state_to_dict(state: GameState) -> Dict:
    """Convert a GameState into a JSON-serializable dictionary."""
    return {
        "board": [row[:] for row in state.board],
        "phase": int(state.phase.value),
        "current_player": int(state.current_player.value),
        "marked_black": _marks_to_list(state.marked_black),
        "marked_white": _marks_to_list(state.marked_white),
        "forced_removals_done": int(state.forced_removals_done),
        "move_count": int(state.move_count),
        "pending_marks_required": int(state.pending_marks_required),
        "pending_marks_remaining": int(state.pending_marks_remaining),
        "pending_captures_required": int(state.pending_captures_required),
        "pending_captures_remaining": int(state.pending_captures_remaining),
    }


def state_from_dict(data: Dict) -> GameState:
    """Reconstruct a GameState from ``state_to_dict`` output."""
    return GameState(
        board=[[int(cell) for cell in row] for row in data["board"]],
        phase=Phase(int(data["phase"])),
        current_player=Player(int(data["current_player"])),
        marked_black=_marks_from_list(data.get("marked_black", [])),
        marked_white=_marks_from_list(data.get("marked_white", [])),
        forced_removals_done=int(data.get("forced_removals_done", 0)),
        move_count=int(data.get("move_count", 0)),
        pending_marks_required=int(data.get("pending_marks_required", 0)),
        pending_marks_remaining=int(data.get("pending_marks_remaining", 0)),
        pending_captures_required=int(data.get("pending_captures_required", 0)),
        pending_captures_remaining=int(data.get("pending_captures_remaining", 0)),
    )


def sample_to_record(
    state: GameState, 
    policy: np.ndarray, 
    legal_moves: List[dict],
    value: float, 
    soft_value: float
) -> Dict:
    """Convert a training tuple into a JSON-serializable record."""
    return {
        "state": state_to_dict(state),
        "policy": policy.astype(float).tolist(),
        "legal_moves": legal_moves,  # Already JSON-serializable dicts
        "value": float(value),
        "soft_value": float(soft_value),
    }


def flatten_training_games(
    training_games: Iterable[Tuple[List[GameState], List[np.ndarray], List[List[dict]], float, float]]
) -> Iterator[SampleTuple]:
    """
    Expand raw self-play outputs into per-position training tuples.

    Each yielded tuple matches the input expected by ``train_network``:
    ``(GameState, np.ndarray, List[dict], value, soft_value)``.
    """
    for game_states, game_policies, game_legal_moves, result, soft_value in training_games:
        for state, policy, legal_moves in zip(game_states, game_policies, game_legal_moves):
            sign = 1.0 if state.current_player == Player.BLACK else -1.0
            yield (
                state.copy(),
                np.asarray(policy, dtype=float),
                list(legal_moves),
                sign * result,
                sign * soft_value,
            )


def write_records_to_jsonl(records: Iterable[Dict], path: str) -> int:
    """Write sample dictionaries to a JSONL file. Returns number of records written."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path_obj.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")
            count += 1
    return count


def iter_sample_records(path: str) -> Iterator[Dict]:
    """Yield sample dictionaries from a JSONL file."""
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _record_to_sample(record: Dict) -> SampleTuple:
    state = state_from_dict(record["state"])
    policy = np.asarray(record["policy"], dtype=float)
    # Handle legacy records without legal_moves
    legal_moves = record.get("legal_moves", [])
    value = float(record.get("value") if "value" in record else record.get("result"))
    soft_value = float(record.get("soft_value", value))
    return state, policy, legal_moves, value, soft_value


def load_examples_from_files(paths: Sequence[str]) -> List[SampleTuple]:
    """Load all samples from the provided JSONL files."""
    examples: List[SampleTuple] = []
    for path in paths:
        for record in iter_sample_records(path):
            examples.append(_record_to_sample(record))
    return examples
