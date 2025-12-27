"""
Benchmark v0 MCTS on multiple devices (e.g., CPU vs CUDA) without running the legacy pipeline.

This is useful for isolating the impact of CUDA kernels (fast_legal_mask, fast_apply_moves, etc.)
without the noise of the legacy baseline.

Usage:
    python -m tools.benchmark_cuda --samples 10 --sims 128 --devices cpu,cuda
    python -m tools.benchmark_cuda --samples 20 --sims 256 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import random
import time
from statistics import fmean
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
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


def _run_device(
    device: torch.device,
    model: ChessNet,
    states: Sequence[GameState],
    sims: int,
    batch_size: int,
) -> Tuple[List[float], Dict[str, float]]:
    model = model.to(device)
    model.eval()
    v0 = V0MCTS(
        model=model,
        num_simulations=sims,
        exploration_weight=1.0,
        temperature=1.0,
        device=str(device),
        add_dirichlet_noise=False,
        seed=0,
        batch_K=batch_size,
    )

    elapsed: List[float] = []
    for state in states:
        start = time.perf_counter()
        v0.search(state)
        elapsed.append(time.perf_counter() - start)

    arr = np.array(elapsed, dtype=float)
    summary = {
        "avg": arr.mean(),
        "median": float(np.median(arr)),
        "std": arr.std(),
        "min": arr.min(),
        "max": arr.max(),
    }
    return elapsed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark v0 MCTS on multiple devices (CPU vs CUDA)."
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of random states to evaluate.")
    parser.add_argument("--max-moves", type=int, default=60, help="Maximum random plies per sampled state.")
    parser.add_argument("--sims", type=int, default=128, help="Number of simulations per run.")
    parser.add_argument("--batch-size", type=int, default=16, help="Leaf batch size.")
    parser.add_argument(
        "--devices",
        type=str,
        default="cpu,cuda",
        help="Comma-separated list of devices to benchmark (e.g., 'cpu,cuda' or 'cuda:0').",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed + 1)
    np.random.seed(args.seed + 2)

    devices = []
    for name in args.devices.split(","):
        name = name.strip()
        if not name:
            continue
        if "cuda" in name and not torch.cuda.is_available():
            print(f"Warning: skipping {name} (CUDA not available)")
            continue
        devices.append(torch.device(name))

    if not devices:
        raise ValueError("No valid devices specified via --devices.")

    model = ChessNet(board_size=GameState.BOARD_SIZE)
    base_rng = random.Random(args.seed + 3)
    states = [_random_state(args.max_moves, base_rng) for _ in range(args.samples)]

    all_results: Dict[str, Dict[str, float]] = {}
    timings: Dict[str, List[float]] = {}

    for device in devices:
        print(f"[benchmark] device={device} samples={args.samples} sims={args.sims} batch={args.batch_size}")
        elapsed, summary = _run_device(device, model, states, args.sims, args.batch_size)
        all_results[str(device)] = summary
        timings[str(device)] = elapsed
        print(
            f"  avg={summary['avg']:.3f}s median={summary['median']:.3f}s "
            f"std={summary['std']:.3f}s min={summary['min']:.3f}s max={summary['max']:.3f}s"
        )

    if len(devices) >= 2:
        base = timings[str(devices[0])]
        for device in devices[1:]:
            other = timings[str(device)]
            ratios = [a / b for a, b in zip(base, other) if b > 0]
            if not ratios:
                continue
            print(
                f"[speedup] {devices[0]} vs {device}: "
                f"avg={fmean(ratios):.2f}x median={np.median(ratios):.2f}x "
                f"min={min(ratios):.2f}x max={max(ratios):.2f}x"
            )


if __name__ == "__main__":
    main()

