"""
Benchmark legacy rule-engine legal mask generation against the tensorized encoder.

Usage:
    python -m tools.benchmark_legal_mask --states 1000 --batch-size 64 --runs 5 --max-random-moves 80 --seed 0
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Sequence

import torch

from src.game_state import GameState
from src.move_generator import generate_all_legal_moves

from v1.game.move_encoder import (
    DEFAULT_ACTION_SPEC,
    ActionEncodingSpec,
    action_to_index,
    encode_actions,
    encode_actions_python,
)
from v1.game.fast_legal_mask import encode_actions_fast
from v1.game.state_batch import from_game_states

from tools.cross_check_mcts import CrossCheckConfig, sample_states


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _legacy_mask(state: GameState, spec: ActionEncodingSpec) -> torch.BoolTensor:
    mask = torch.zeros(spec.total_dim, dtype=torch.bool)
    legal_moves = generate_all_legal_moves(state)
    board_size = state.BOARD_SIZE
    for move in legal_moves:
        idx = action_to_index(move, board_size, spec)
        if idx is not None:
            mask[idx] = True
    return mask


def benchmark_legacy(
    states: Sequence[GameState],
    spec: ActionEncodingSpec,
) -> float:
    start = time.perf_counter()
    for state in states:
        _legacy_mask(state, spec)
    return time.perf_counter() - start


def benchmark_tensor_python(
    states: Sequence[GameState],
    batch_size: int,
    spec: ActionEncodingSpec,
    device: torch.device,
) -> float:
    start = time.perf_counter()
    for offset in range(0, len(states), batch_size):
        chunk = states[offset : offset + batch_size]
        tensor_batch = from_game_states(chunk, device=device)
        encode_actions_python(tensor_batch, spec)
    _sync_if_needed(device)
    return time.perf_counter() - start


def benchmark_tensor_fast(
    states: Sequence[GameState],
    batch_size: int,
    spec: ActionEncodingSpec,
) -> float:
    start = time.perf_counter()
    for offset in range(0, len(states), batch_size):
        chunk = states[offset : offset + batch_size]
        tensor_batch = from_game_states(chunk, device=torch.device("cpu"))
        mask = encode_actions_fast(tensor_batch, spec)
        if mask is None:
            raise RuntimeError("Fast encoder unavailable.")
    return time.perf_counter() - start


def verify_masks(
    states: Sequence[GameState],
    batch_size: int,
    spec: ActionEncodingSpec,
    device: torch.device,
) -> bool:
    python_masks: List[torch.BoolTensor] = []
    fast_masks: List[torch.BoolTensor] = []
    fast_available = True

    for offset in range(0, len(states), batch_size):
        chunk = states[offset : offset + batch_size]
        tensor_batch = from_game_states(chunk, device=device)
        mask_python = encode_actions_python(tensor_batch, spec)
        python_masks.extend(mask_python.to("cpu"))

        fast_batch = tensor_batch if tensor_batch.board.device.type == "cpu" else from_game_states(chunk, device=torch.device("cpu"))
        fast_mask = encode_actions_fast(fast_batch, spec)
        if fast_mask is None:
            fast_available = False
        else:
            fast_masks.extend(fast_mask)

    for state, tensor_mask in zip(states, python_masks):
        legacy_mask = _legacy_mask(state, spec)
        if not torch.equal(legacy_mask, tensor_mask):
            raise AssertionError("Mask mismatch detected between legacy and tensor outputs.")

    if fast_available and fast_masks:
        for tensor_mask, fast_mask in zip(python_masks, fast_masks):
            if not torch.equal(tensor_mask, fast_mask):
                raise AssertionError("Mask mismatch detected between Python and fast encoder outputs.")

    return fast_available

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legal action mask generation.")
    parser.add_argument("--states", type=int, default=1000, help="Number of random states to sample.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for tensor encoder.")
    parser.add_argument("--max-random-moves", type=int, default=80, help="Random rollout depth for state sampling.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for tensor benchmark (cpu or cuda).")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark repetitions.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for state sampling.")
    parser.add_argument("--verify", action="store_true", help="Also verify masks match between implementations.")
    args = parser.parse_args()

    spec = DEFAULT_ACTION_SPEC
    device = torch.device(args.device)

    cfg = CrossCheckConfig(
        num_states=args.states,
        max_random_moves=args.max_random_moves,
        seed=args.seed,
    )
    states = sample_states(cfg)

    fast_available = False
    if args.verify:
        print("Verifying tensor masks against legacy rule engine...")
        fast_available = verify_masks(states, args.batch_size, spec, device)
        print("Verification succeeded.")
    else:
        probe_batch = from_game_states(states[: min(len(states), args.batch_size)], device=torch.device("cpu"))
        fast_available = encode_actions_fast(probe_batch, spec) is not None

    legacy_times = []
    python_times = []
    fast_times: List[float] = []

    for run in range(args.runs):
        legacy_elapsed = benchmark_legacy(states, spec)
        legacy_times.append(legacy_elapsed)

        python_elapsed = benchmark_tensor_python(states, args.batch_size, spec, device)
        python_times.append(python_elapsed)

        fast_elapsed_str = "N/A"
        if fast_available:
            fast_elapsed = benchmark_tensor_fast(states, args.batch_size, spec)
            fast_times.append(fast_elapsed)
            fast_elapsed_str = f"{fast_elapsed*1000:.2f} ms"

        print(
            f"Run {run+1}/{args.runs}: legacy={legacy_elapsed*1000:.2f} ms, "
            f"tensor-python={python_elapsed*1000:.2f} ms, "
            f"tensor-fast={fast_elapsed_str}"
        )

    def summarize(samples: List[float]) -> str:
        mean = statistics.mean(samples)
        stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        return f"{mean*1000:.2f} ± {stdev*1000:.2f} ms"

    print("\nSummary:")
    print(f"  Legacy rule engine: {summarize(legacy_times)}")
    print(f"  Tensor encode_actions (Python): {summarize(python_times)}")
    if fast_available and fast_times:
        print(f"  Tensor encode_actions (C++ fast): {summarize(fast_times)}")
    else:
        print("  Tensor encode_actions (C++ fast): unavailable")


if __name__ == "__main__":
    main()


# python -m tools.benchmark_legal_mask --states 1000 --batch-size 128 --device cuda --runs 3 --verify  