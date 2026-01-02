#!/usr/bin/env python3
"""
Check v0 policy index alignment vs legacy move ordering and inspect root stats.
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import move_to_key
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from v0.python.mcts import MCTS as V0MCTS
from v0.python.move_encoder import DEFAULT_ACTION_SPEC, action_to_index, encode_actions_fast
from v0.python.state_batch import from_game_states


def _load_model(checkpoint: str | None, device: str) -> ChessNet:
    model = ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    )
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _random_state(rng: random.Random, max_moves: int) -> GameState:
    state = GameState()
    num_steps = rng.randint(0, max_moves)
    for _ in range(num_steps):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def _order_match_rate(moves_v0: List[dict], moves_py: List[dict]) -> float:
    if len(moves_v0) != len(moves_py) or not moves_py:
        return 0.0
    matches = 0
    for mv0, mpy in zip(moves_v0, moves_py):
        if move_to_key(mv0) == move_to_key(mpy):
            matches += 1
    return matches / len(moves_py)


def _legacy_index_map(moves_py: List[dict]) -> Dict[Tuple, int]:
    mapping: Dict[Tuple, int] = {}
    for idx, move in enumerate(moves_py):
        key = move_to_key(move)
        if key not in mapping:
            mapping[key] = idx
    return mapping


def _inspect_root(
    mcts: V0MCTS,
    state: GameState,
    moves_v0: List[dict],
    policy_v0: np.ndarray,
    moves_py: List[dict],
    top_k: int,
) -> None:
    core = mcts._core  # noqa: SLF001 - intentional for debug
    stats = core.get_root_children_stats()
    root_visits = float(getattr(core, "root_visit_count", 0.0))
    child_visits = [float(s.get("visit_count", 0.0)) for s in stats]
    priors = [float(s.get("prior", 0.0)) for s in stats]

    zero_visit = sum(1 for v in child_visits if v == 0.0)
    nonzero_visit = sum(1 for v in child_visits if v > 0.0)
    zero_prior = sum(1 for p in priors if p == 0.0)
    nonzero_prior = sum(1 for p in priors if p > 0.0)
    sum_child_visits = float(sum(child_visits))

    print("\n[Root stats]")
    print(f"  root_visit_count={root_visits:.1f} sum_child_visits={sum_child_visits:.1f}")
    print(f"  children={len(stats)} priors>0={nonzero_prior} priors==0={zero_prior}")
    print(f"  visits>0={nonzero_visit} visits==0={zero_visit}")
    if priors:
        print(f"  prior_min={min(priors):.6g} prior_max={max(priors):.6g}")
    if child_visits:
        print(f"  visit_min={min(child_visits):.1f} visit_max={max(child_visits):.1f}")

    if policy_v0.size:
        zero_policy = int((policy_v0 == 0).sum())
        print(f"  policy_len={policy_v0.size} policy_zero={zero_policy}")

    legacy_index = _legacy_index_map(moves_py)
    if moves_v0 and policy_v0.size:
        top_idx = np.argsort(policy_v0)[::-1][:top_k]
        print("\n[Top actions v0 -> legacy index]")
        for rank, idx in enumerate(top_idx, start=1):
            mv = moves_v0[int(idx)]
            key = move_to_key(mv)
            legacy_idx = legacy_index.get(key, None)
            prob = float(policy_v0[int(idx)])
            print(f"  #{rank:02d} v0_i={int(idx):02d} p={prob:.6f} legacy_i={legacy_idx} key={key}")


def _inspect_mapping(state: GameState) -> None:
    moves_py = generate_all_legal_moves(state)
    indices: List[int] = []
    missing = 0
    dup = 0
    seen: Dict[int, int] = {}
    for mv in moves_py:
        idx = action_to_index(mv, state.BOARD_SIZE, DEFAULT_ACTION_SPEC)
        if idx is None:
            missing += 1
            continue
        idx_int = int(idx)
        indices.append(idx_int)
        if idx_int in seen:
            dup += 1
        else:
            seen[idx_int] = 1

    batch = from_game_states([state], device=torch.device("cpu"))
    mask = encode_actions_fast(batch, DEFAULT_ACTION_SPEC, return_metadata=False)
    if isinstance(mask, tuple):
        mask = mask[0]
    mask_row = mask[0].to(torch.bool)
    not_in_mask = sum(1 for idx in indices if not bool(mask_row[idx]))

    print("\n[Action index mapping]")
    print(f"  legacy_moves={len(moves_py)} missing_idx={missing} dup_idx={dup}")
    print(f"  action_idx_not_in_mask={not_in_mask}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check v0 policy ordering vs legacy move ordering.",
    )
    parser.add_argument("--num_states", type=int, default=20)
    parser.add_argument("--max_moves", type=int, default=60)
    parser.add_argument("--mcts_sims", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exploration_weight", type=float, default=1.0)
    parser.add_argument("--batch_leaves", type=int, default=256)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--no_dirichlet", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = _load_model(args.checkpoint, args.device)
    mcts = V0MCTS(
        model=model,
        num_simulations=args.mcts_sims,
        exploration_weight=args.exploration_weight,
        temperature=args.temperature,
        device=args.device,
        add_dirichlet_noise=not args.no_dirichlet,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        batch_K=args.batch_leaves,
        seed=args.seed,
        verbose=False,
    )

    states: List[GameState] = [GameState()]
    for _ in range(max(0, args.num_states - 1)):
        states.append(_random_state(rng, args.max_moves))

    print("== v0 policy index alignment check ==")
    print(
        f"states={len(states)} mcts_sims={args.mcts_sims} batch_leaves={args.batch_leaves} "
        f"dirichlet={'off' if args.no_dirichlet else 'on'}"
    )

    order_rates: List[float] = []
    length_mismatches = 0

    for idx, state in enumerate(states):
        moves_py = generate_all_legal_moves(state)
        moves_v0, policy_v0 = mcts.search(state)

        if len(moves_py) != len(moves_v0):
            length_mismatches += 1

        rate = _order_match_rate(moves_v0, moves_py)
        order_rates.append(rate)

        if idx == 0:
            print("\n[Initial state]")
            print(f"  legacy_moves={len(moves_py)} v0_moves={len(moves_v0)} policy_len={policy_v0.size}")
            print(f"  order_match_rate={rate:.3f}")
            _inspect_mapping(state)
            _inspect_root(mcts, state, moves_v0, policy_v0, moves_py, args.top_k)

    if order_rates:
        print("\n[Order match summary]")
        print(f"  length_mismatches={length_mismatches}/{len(order_rates)}")
        print(
            f"  rate_min={min(order_rates):.3f} "
            f"rate_mean={sum(order_rates)/len(order_rates):.3f} "
            f"rate_max={max(order_rates):.3f}"
        )


if __name__ == "__main__":
    main()
