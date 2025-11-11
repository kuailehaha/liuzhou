"""
Compare the legacy Python MCTS implementation with the new v0 C++ MCTS core.

This script samples random game states, runs both implementations for the same
number of simulations, and checks that the move distributions match within a
small numerical tolerance.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import MCTS as LegacyMCTS
from src.mcts import move_to_key
from src.neural_network import ChessNet
from v0.python.mcts import MCTS as V0MCTS


def _random_state(max_moves: int, rng: random.Random) -> GameState:
    state = GameState()
    for _ in range(rng.randint(0, max_moves)):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def _compare_distributions(
    legacy_moves,
    legacy_probs,
    v0_moves,
    v0_probs,
) -> Tuple[float, float]:
    l_map: Dict[Tuple, float] = {move_to_key(move): prob for move, prob in zip(legacy_moves, legacy_probs)}
    v0_map: Dict[Tuple, float] = {move_to_key(move): prob for move, prob in zip(v0_moves, v0_probs)}
    if l_map.keys() != v0_map.keys():
        raise AssertionError(f"Move sets diverged.\nlegacy={l_map.keys()}\nv0={v0_map.keys()}")
    diffs = [abs(l_map[key] - v0_map[key]) for key in l_map.keys()]
    if not diffs:
        return 0.0, 0.0
    return float(max(diffs)), float(sum(diffs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=25, help="Number of random game states to test.")
    parser.add_argument("--max-moves", type=int, default=60, help="Random plies per sampled state.")
    parser.add_argument("--sims", type=int, default=64, help="Number of simulations per search.")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="Maximum allowed L-infinity difference.")
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ChessNet(board_size=GameState.BOARD_SIZE)
    model.eval()

    rng = random.Random(seed + 1)

    max_linf = 0.0
    max_l1 = 0.0

    for i in range(args.samples):
        state = _random_state(args.max_moves, rng)

        legacy = LegacyMCTS(
            model=model,
            num_simulations=args.sims,
            exploration_weight=1.0,
            temperature=1.0,
            device="cpu",
            add_dirichlet_noise=False,
            virtual_loss_weight=0.0,
            batch_K=8,
            verbose=False,
        )
        v0_mcts = V0MCTS(
            model=model,
            num_simulations=args.sims,
            exploration_weight=1.0,
            temperature=1.0,
            device="cpu",
            add_dirichlet_noise=False,
            seed=seed,
            batch_K=8,
        )

        legacy_moves, legacy_policy = legacy.search(state)
        v0_moves, v0_policy = v0_mcts.search(state)

        linf, l1 = _compare_distributions(legacy_moves, legacy_policy, v0_moves, v0_policy)
        max_linf = max(max_linf, linf)
        max_l1 = max(max_l1, l1)

        if linf > args.tolerance:
            raise AssertionError(
                f"Sample {i}: distributions diverged (linf={linf:.6f} > {args.tolerance:.6f})"
            )

    print(
        f"[verify_v0_mcts] success: samples={args.samples} sims={args.sims} "
        f"max_linf={max_linf:.6f} max_l1={max_l1:.6f}"
    )


if __name__ == "__main__":
    main()
