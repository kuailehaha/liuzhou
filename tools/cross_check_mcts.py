"""
Cross-check script comparing legacy MCTS outputs against the tensorized v1 pipeline.

Usage:
    python tools/cross_check_mcts.py --states 8 --max-random-moves 40 --num-simulations 64
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from src.game_state import GameState
from src.mcts import MCTS
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from v1.game.move_encoder import ActionEncodingSpec, DEFAULT_ACTION_SPEC, action_to_index
from v1.game.state_batch import from_game_states
from v1.mcts.vectorized_mcts import VectorizedMCTS, VectorizedMCTSConfig


@dataclass
class CrossCheckConfig:
    num_states: int = 8
    max_random_moves: int = 40
    num_simulations: int = 64
    exploration_weight: float = 1.0
    temperature: float = 1.0
    batch_leaves: int = 16
    seed: int = 0
    device: str = "cpu"
    action_spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC


def sample_states(config: CrossCheckConfig) -> List[GameState]:
    """Generate random game states by playing random legal moves."""
    rng = random.Random(config.seed)
    states: List[GameState] = []

    for _ in range(config.num_states):
        state = GameState()
        steps = rng.randint(0, config.max_random_moves)
        for _ in range(steps):
            legal = generate_all_legal_moves(state)
            if not legal:
                break
            move = rng.choice(legal)
            state = apply_move(state, move, quiet=True)
            if state.is_game_over():
                break
        states.append(state)
    return states


def legacy_policy_map(
    state: GameState,
    mcts: MCTS,
    spec: ActionEncodingSpec,
) -> Dict[int, float]:
    moves, policy = mcts.search(state)
    board_size = state.BOARD_SIZE
    result: Dict[int, float] = {}

    for move, prob in zip(moves, policy):
        idx = action_to_index(move, board_size, spec)
        if idx is None:
            continue
        result[idx] = float(prob)
    return result


def vectorized_policy_map(
    states: Sequence[GameState],
    vmcts: VectorizedMCTS,
    spec: ActionEncodingSpec,
    device: torch.device,
) -> List[Dict[int, float]]:
    tensor_batch = from_game_states(states, device=device)
    policies, legal_mask = vmcts.search(tensor_batch)
    maps: List[Dict[int, float]] = []

    for idx in range(tensor_batch.batch_size):
        row = policies[idx]
        mask = legal_mask[idx]
        legal_indices = mask.nonzero(as_tuple=False).flatten()
        mapping = {int(i.item()): float(row[i].item()) for i in legal_indices}
        maps.append(mapping)
    return maps


def compare_policies(
    legacy: Dict[int, float],
    tensorized: Dict[int, float],
) -> Tuple[float, float]:
    keys = set(legacy) | set(tensorized)
    l1 = 0.0
    max_diff = 0.0
    for key in keys:
        diff = abs(legacy.get(key, 0.0) - tensorized.get(key, 0.0))
        l1 += diff
        max_diff = max(max_diff, diff)
    return l1, max_diff


def run_cross_check(cfg: CrossCheckConfig) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()

    states = sample_states(cfg)

    legacy_mcts = MCTS(
        model=model,
        num_simulations=cfg.num_simulations,
        exploration_weight=cfg.exploration_weight,
        temperature=cfg.temperature,
        device=str(device),
        add_dirichlet_noise=False,
        virtual_loss_weight=0.0,
    )

    vmcts_config = VectorizedMCTSConfig(
        num_simulations=cfg.num_simulations,
        exploration_weight=cfg.exploration_weight,
        temperature=cfg.temperature,
        batch_leaves=cfg.batch_leaves,
        virtual_loss_weight=0.0,
        action_spec=cfg.action_spec,
    )
    vectorized = VectorizedMCTS(model=model, config=vmcts_config, device=str(device))

    tensor_policies = vectorized_policy_map(states, vectorized, cfg.action_spec, device)

    l1_diffs: List[float] = []
    max_diffs: List[float] = []

    for idx, state in enumerate(states):
        legacy_policy = legacy_policy_map(state, legacy_mcts, cfg.action_spec)
        tensor_policy = tensor_policies[idx]
        l1, max_diff = compare_policies(legacy_policy, tensor_policy)
        l1_diffs.append(l1)
        max_diffs.append(max_diff)

        print(f"[State {idx:02d}] L1={l1:.6f}  max|diff|={max_diff:.6f}  legal={len(legacy_policy)}")

    if l1_diffs:
        print("\nSummary:")
        print(f"  Mean L1: {sum(l1_diffs) / len(l1_diffs):.6f}")
        print(f"  Max L1:  {max(l1_diffs):.6f}")
        print(f"  Mean max|diff|: {sum(max_diffs) / len(max_diffs):.6f}")
        print(f"  Max max|diff|:  {max(max_diffs):.6f}")


def parse_args() -> CrossCheckConfig:
    parser = argparse.ArgumentParser(description="Compare legacy MCTS with tensorized VectorizedMCTS.")
    parser.add_argument("--states", type=int, default=8, help="Number of random states to compare.")
    parser.add_argument("--max-random-moves", type=int, default=40, help="Upper bound on random rollouts per state.")
    parser.add_argument("--num-simulations", type=int, default=64, help="MCTS simulations per evaluation.")
    parser.add_argument("--exploration-weight", type=float, default=1.0, help="PUCT exploration weight.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for both searches.")
    parser.add_argument("--batch-leaves", type=int, default=16, help="VectorizedMCTS leaf batch size.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for state generation.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to run on.")

    args = parser.parse_args()
    return CrossCheckConfig(
        num_states=args.states,
        max_random_moves=args.max_random_moves,
        num_simulations=args.num_simulations,
        exploration_weight=args.exploration_weight,
        temperature=args.temperature,
        batch_leaves=args.batch_leaves,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run_cross_check(parse_args())

