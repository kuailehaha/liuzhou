"""
Cross-check v0 TensorStateBatch conversions against the legacy Python version.

Usage:
    python tools/verify_v0_state_batch.py --samples 128 --max-moves 60 --device cpu
"""

from __future__ import annotations

import argparse
import random
from typing import List

import torch

import v0.python.state_batch as state_batch_cpp
from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from v1.game.state_batch import from_game_states as py_from_game_states


def _random_state(max_moves: int) -> GameState:
    state = GameState()
    for _ in range(random.randint(0, max_moves)):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = random.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def _states_equal(a: GameState, b: GameState) -> bool:
    return (
        a.phase == b.phase
        and a.current_player == b.current_player
        and a.board == b.board
        and a.marked_black == b.marked_black
        and a.marked_white == b.marked_white
        and a.pending_marks_required == b.pending_marks_required
        and a.pending_marks_remaining == b.pending_marks_remaining
        and a.pending_captures_required == b.pending_captures_required
        and a.pending_captures_remaining == b.pending_captures_remaining
        and a.forced_removals_done == b.forced_removals_done
        and a.move_count == b.move_count
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64, help="Number of random states to sample.")
    parser.add_argument("--max-moves", type=int, default=60, help="Max random plies per sampled state.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device string for the C++ batch (e.g. cpu, cuda:0)."
    )
    args = parser.parse_args()

    random.seed(0)
    states: List[GameState] = [_random_state(args.max_moves) for _ in range(args.samples)]

    batch_cpp = state_batch_cpp.from_game_states(states, device=args.device)
    restored = state_batch_cpp.to_game_states(batch_cpp)

    batch_py = py_from_game_states(states, device=torch.device(args.device))

    mismatches = [
        (i, s_cpp, s_py)
        for i, (s_cpp, s_py) in enumerate(zip(restored, states))
        if not _states_equal(s_cpp, s_py)
    ]

    if mismatches:
        first_idx, got, expected = mismatches[0]
        raise AssertionError(f"State mismatch at index {first_idx}\nGot:\n{got}\nExpected:\n{expected}")

    if batch_cpp.board.device.type != str(torch.device(args.device).type):
        raise AssertionError("C++ batch device mismatch")

    print(
        f"[verify_v0_state_batch] success: {args.samples} samples | device={args.device} | "
        f"C++ board shape={tuple(batch_cpp.board.shape)} | python board shape={tuple(batch_py.board.shape)}"
    )


if __name__ == "__main__":
    main()
