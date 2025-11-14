"""
Compare legacy Python move generation with the v0 action encoder.

This script samples random game states by playing `max_moves` random legal moves
from the root position, then checks whether `generate_all_legal_moves` matches
the moves reconstructed from `v0_core.encode_actions_fast`.  Any mismatches are
printed for inspection along with phase/move-count statistics.

Usage (from repo root, ensure PYTHONPATH includes build/v0/src):

    python tools/check_v0_actions_all.py --states 400 --max-moves 160
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import move_to_key
from v0.python.move_encoder import decode_action_indices, DEFAULT_ACTION_SPEC
from v0.python.state_batch import from_game_states
import v0_core


MoveKey = Tuple


def random_state(
    rng: random.Random,
    max_moves: int,
) -> GameState:
    state = GameState()
    for _ in range(max_moves):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def canonical_moves(moves: Sequence[dict]) -> Dict[MoveKey, dict]:
    return {move_to_key(move): move for move in moves}


def decode_from_mask(state: GameState) -> List[dict]:
    batch = from_game_states([state], torch.device("cpu"))
    mask, _ = v0_core.encode_actions_fast(
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
        DEFAULT_ACTION_SPEC.placement_dim,
        DEFAULT_ACTION_SPEC.movement_dim,
        DEFAULT_ACTION_SPEC.selection_dim,
        DEFAULT_ACTION_SPEC.auxiliary_dim,
    )
    mask_row = mask[0]
    indices = torch.nonzero(mask_row, as_tuple=False).view(-1)
    if indices.numel() == 0:
        return []

    replicated_states = [state.copy() for _ in range(indices.numel())]
    decode_batch = from_game_states(replicated_states, torch.device("cpu"))
    decoded = decode_action_indices(indices, decode_batch, DEFAULT_ACTION_SPEC)
    return [move for move in decoded if move is not None]


def compare_once(state: GameState) -> Tuple[List[MoveKey], List[MoveKey]]:
    legacy_moves = canonical_moves(generate_all_legal_moves(state))
    v0_moves = canonical_moves(decode_from_mask(state))

    missing = sorted(legacy_moves.keys() - v0_moves.keys())
    extra = sorted(v0_moves.keys() - legacy_moves.keys())
    return missing, extra


def run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    phase_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    issues: List[Tuple[int, GameState, List[MoveKey], List[MoveKey]]] = []

    for idx in range(args.states):
        state = random_state(rng, args.max_moves)
        phase_name = state.phase.name
        phase_counter[phase_name] += 1

        missing, extra = compare_once(state)
        if missing or extra:
            issue_counter[phase_name] += 1
            issues.append((idx, state, missing, extra))
            if len(issues) >= args.limit:
                break

    total_states = sum(phase_counter.values())
    print(f"Checked {total_states} random states")
    for phase, count in sorted(phase_counter.items()):
        diff = issue_counter.get(phase, 0)
        print(f"  {phase:18s}: {count:5d} states | mismatches: {diff}")

    if not issues:
        print("No mismatches detected.")
        return 0

    print(f"\nFound {len(issues)} mismatched states (showing up to {args.limit}):")
    for idx, state, missing, extra in issues[: args.limit]:
        print(
            f"\n--- Case #{idx} | phase={state.phase.name} | move_count={state.move_count} "
            f"| legacy={len(missing) + len(canonical_moves(generate_all_legal_moves(state)))} "
            f"| v0={len(decode_from_mask(state))}"
        )
        if missing:
            print("  Missing actions (legacy only):")
            for key in missing:
                print(f"    {key}")
        if extra:
            print("  Extra actions (v0 only):")
            for key in extra:
                print(f"    {key}")

    if args.fail_on_diff:
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-check legacy vs v0 legal action spaces.")
    parser.add_argument("--states", type=int, default=400, help="Number of random states to evaluate.")
    parser.add_argument("--max-moves", type=int, default=160, help="Max random moves per sampled state.")
    parser.add_argument("--seed", type=int, default=7777, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of mismatched cases to print in detail.",
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with code 1 if any mismatch is detected.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))
