"""
Cross-check script comparing legacy MCTS outputs against the tensorized v1 pipeline.

Usage:
    python tools/cross_check_mcts.py --states 8 --max-random-moves 40 --num-simulations 64
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.game_state import GameState
from src.mcts import MCTS
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from v1.game.move_encoder import (
    ActionEncodingSpec,
    DEFAULT_ACTION_SPEC,
    action_to_index,
    decode_action_indices,
)
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


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


DEBUG_L1_THRESHOLD = 1e-3
MISMATCH_EPS = 1e-4


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
    device: torch.device,
) -> Tuple[Dict[int, float], float]:
    _sync_if_needed(device)
    start = time.perf_counter()
    # Always rebuild the tree so each sample starts from a clean root.
    mcts.root = None
    moves, policy = mcts.search(state)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    board_size = state.BOARD_SIZE
    result: Dict[int, float] = {}

    for move, prob in zip(moves, policy):
        idx = action_to_index(move, board_size, spec)
        if idx is None:
            continue
        result[idx] = float(prob)
    return result, elapsed


def vectorized_policy_map(
    states: Sequence[GameState],
    vmcts: VectorizedMCTS,
    spec: ActionEncodingSpec,
    device: torch.device,
) -> Tuple[List[Dict[int, float]], float]:
    _sync_if_needed(device)
    start = time.perf_counter()
    tensor_batch = from_game_states(states, device=device)
    policies, legal_mask = vmcts.search(tensor_batch)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    maps: List[Dict[int, float]] = []

    for idx in range(tensor_batch.batch_size):
        row = policies[idx]
        mask = legal_mask[idx]
        legal_indices = mask.nonzero(as_tuple=False).flatten()
        mapping = {int(i.item()): float(row[i].item()) for i in legal_indices}
        maps.append(mapping)
    return maps, elapsed


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


def _decode_indices_for_state(
    state: GameState,
    indices: List[int],
    spec: ActionEncodingSpec,
    device: torch.device,
):
    if not indices:
        return []
    clones = [state.copy() for _ in indices]
    batch = from_game_states(clones, device=device)
    tensor = torch.tensor(indices, dtype=torch.long)
    moves = decode_action_indices(tensor, batch, spec)
    return moves


def _print_debug_details(
    state_idx: int,
    state: GameState,
    legacy_map: Dict[int, float],
    tensor_map: Dict[int, float],
    spec: ActionEncodingSpec,
    device: torch.device,
    l1: float,
    max_diff: float,
) -> None:
    print(f"\n[Debug] State {state_idx}: phase={state.phase} player={state.current_player}")
    missing_indices = sorted(set(legacy_map) - set(tensor_map))
    extra_indices = sorted(set(tensor_map) - set(legacy_map))
    common = set(legacy_map) & set(tensor_map)
    mismatched = sorted(
        (
            (idx, legacy_map[idx], tensor_map[idx], abs(legacy_map[idx] - tensor_map[idx]))
            for idx in common
        ),
        key=lambda item: item[3],
        reverse=True,
    )

    def describe(label: str, indices: List[int]):
        if not indices:
            return
        moves = _decode_indices_for_state(state, indices, spec, device)
        print(f"  {label} ({len(indices)}):")
        for idx_val, move in zip(indices, moves):
            print(f"    idx {idx_val:4d}: {move}")

    describe("Missing in tensorized", missing_indices)
    describe("Extra in tensorized", extra_indices)

    relevant_mismatches = [info for info in mismatched if info[3] > MISMATCH_EPS]
    if relevant_mismatches:
        print("  Mismatched probabilities:")
        for idx_val, legacy_prob, tensor_prob, diff in relevant_mismatches:
            move = _decode_indices_for_state(state, [idx_val], spec, device)[0]
            print(
                f"    idx {idx_val:4d}: legacy={legacy_prob:.6f} tensorized={tensor_prob:.6f} diff={diff:.6f} move={move}"
            )
    else:
        print("  No mismatched probabilities above threshold.")

    print("  GameState snapshot:")
    print(state)


def run_cross_check(cfg: CrossCheckConfig) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

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
        batch_K=cfg.batch_leaves,
    )

    vmcts_config = VectorizedMCTSConfig(
        num_simulations=cfg.num_simulations,
        exploration_weight=cfg.exploration_weight,
        temperature=cfg.temperature,
        batch_leaves=cfg.batch_leaves,
        virtual_loss_weight=0.0,
        action_spec=cfg.action_spec,
        log_stats=True,
    )
    vectorized = VectorizedMCTS(model=model, config=vmcts_config, device=str(device))

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    tensor_policies, vector_elapsed = vectorized_policy_map(states, vectorized, cfg.action_spec, device)
    vectorized._roots.clear()

    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)

    l1_diffs: List[float] = []
    max_diffs: List[float] = []
    legacy_durations: List[float] = []

    for idx, state in enumerate(states):
        legacy_policy, elapsed = legacy_policy_map(state, legacy_mcts, cfg.action_spec, device)
        legacy_durations.append(elapsed)
        tensor_policy = tensor_policies[idx]
        l1, max_diff = compare_policies(legacy_policy, tensor_policy)
        l1_diffs.append(l1)
        max_diffs.append(max_diff)

        print(f"[State {idx:02d}] L1={l1:.6f}  max|diff|={max_diff:.6f}  legal={len(legacy_policy)}")
        # if l1 > DEBUG_L1_THRESHOLD:
        #     _print_debug_details(
        #         state_idx=idx,
        #         state=state,
        #         legacy_map=legacy_policy,
        #         tensor_map=tensor_policy,
        #         spec=cfg.action_spec,
        #         device=device,
        #         l1=l1,
        #         max_diff=max_diff,
        #     )

    if l1_diffs:
        print("\nSummary:")
        print(f"  Mean L1: {sum(l1_diffs) / len(l1_diffs):.6f}")
        print(f"  Max L1:  {max(l1_diffs):.6f}")
        print(f"  Mean max|diff|: {sum(max_diffs) / len(max_diffs):.6f}")
        print(f"  Max max|diff|:  {max(max_diffs):.6f}")
        if legacy_durations:
            total_legacy = sum(legacy_durations)
            mean_legacy = total_legacy / len(legacy_durations)
            print(f"  Legacy search avg: {mean_legacy * 1_000:.3f} ms  (total {total_legacy:.3f}s)")
        per_state_vector = vector_elapsed / max(len(states), 1)
        print(f"  Vectorized search avg: {per_state_vector * 1_000:.3f} ms  (batch total {vector_elapsed:.3f}s)")
        if legacy_durations and per_state_vector > 0:
            print(f"  Speedup (legacy/vectorized): {mean_legacy / per_state_vector:.2f}x")


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


# python -m tools.cross_check_mcts --states 200 --max-random-moves 40 --num-simulations 64 --device cuda --batch-leaves 16 > check_log.txt      