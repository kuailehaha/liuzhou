"""
Cross-check network policy projection against the legacy pipeline.

Usage:
    python tools/cross_check_policy_projection.py --states 32 --max-random-moves 50 --device cpu
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import (
    ChessNet,
    NUM_INPUT_CHANNELS,
    get_move_probabilities,
    state_to_tensor,
)

from v1.game.move_encoder import (
    ActionEncodingSpec,
    DEFAULT_ACTION_SPEC,
    action_to_index,
    encode_actions,
)
from v1.game.state_batch import from_game_states
from v1.net.encoding import project_policy_logits, states_to_model_input


@dataclass
class CrossCheckArgs:
    states: int = 16
    max_random_moves: int = 50
    seed: int = 0
    device: str = "cpu"
    tolerance: float = 1e-6
    action_spec: ActionEncodingSpec = DEFAULT_ACTION_SPEC


def sample_states(num_states: int, max_moves: int, seed: int) -> List[GameState]:
    rng = random.Random(seed)
    states: List[GameState] = []
    for _ in range(num_states):
        state = GameState()
        steps = rng.randint(0, max_moves)
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


def legacy_policy_vector(
    state: GameState,
    net: ChessNet,
    spec: ActionEncodingSpec,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    legacy_input = state_to_tensor(state, state.current_player).to(device)
    with torch.no_grad():
        log_p1, log_p2, log_pmc, _ = net(legacy_input)
    legal_moves = generate_all_legal_moves(state)
    probs, raw = get_move_probabilities(
        log_p1.squeeze(0),
        log_p2.squeeze(0),
        log_pmc.squeeze(0),
        legal_moves,
        board_size=state.BOARD_SIZE,
        device=str(device),
    )
    indices = [
        action_to_index(move, state.BOARD_SIZE, spec)
        for move in legal_moves
    ]
    tensor_probs = torch.zeros(spec.total_dim, dtype=log_p1.dtype, device=device)
    tensor_logits = torch.full(
        (spec.total_dim,), float("-inf"), dtype=log_p1.dtype, device=device
    )
    for prob, logit, index in zip(probs, raw, indices):
        if index is None:
            continue
        tensor_probs[index] = float(prob)
        tensor_logits[index] = logit.to(device)
    return tensor_probs, tensor_logits


def vectorized_policy_vector(
    states: Sequence[GameState],
    net: ChessNet,
    spec: ActionEncodingSpec,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = from_game_states(states, device=device)
    inputs = states_to_model_input(batch)
    with torch.no_grad():
        log_p1, log_p2, log_pmc, _ = net(inputs)
    legal_mask = encode_actions(batch, spec)
    probs, logits = project_policy_logits((log_p1, log_p2, log_pmc), legal_mask, spec)
    return probs, logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check policy projection alignment.")
    parser.add_argument("--states", type=int, default=CrossCheckArgs.states)
    parser.add_argument("--max-random-moves", type=int, default=CrossCheckArgs.max_random_moves)
    parser.add_argument("--seed", type=int, default=CrossCheckArgs.seed)
    parser.add_argument("--device", type=str, default=CrossCheckArgs.device)
    parser.add_argument("--tolerance", type=float, default=CrossCheckArgs.tolerance)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    states = sample_states(args.states, args.max_random_moves, args.seed)

    net = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS).to(device)
    net.eval()

    spec = DEFAULT_ACTION_SPEC

    projected_probs, projected_logits = vectorized_policy_vector(states, net, spec, device)

    max_prob_diff = 0.0
    max_logit_diff = 0.0
    mismatched_states: List[int] = []

    for idx, state in enumerate(states):
        legacy_probs, legacy_logits = legacy_policy_vector(state, net, spec, device)
        diff_probs = torch.abs(projected_probs[idx] - legacy_probs)
        diff_logits = torch.abs(projected_logits[idx] - legacy_logits)

        state_prob_diff = diff_probs.max().item()
        state_logit_diff = diff_logits.max().item()

        max_prob_diff = max(max_prob_diff, state_prob_diff)
        max_logit_diff = max(max_logit_diff, state_logit_diff)

        if state_prob_diff > args.tolerance or state_logit_diff > args.tolerance:
            mismatched_states.append(idx)

    print(f"Checked {len(states)} states on device {device.type}.")
    print(f"Max probability difference: {max_prob_diff:.3e}")
    print(f"Max logit difference:       {max_logit_diff:.3e}")

    if mismatched_states:
        indices = ", ".join(str(i) for i in mismatched_states)
        print(f"States exceeding tolerance ({args.tolerance}): {indices}")
    else:
        print(f"All states within tolerance ({args.tolerance}).")


if __name__ == "__main__":
    main()
