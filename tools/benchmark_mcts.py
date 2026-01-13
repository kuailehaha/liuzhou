"""
Benchmark legacy (Python) vs v0 (C++ core) MCTS search throughput.

Samples random states, runs both implementations with identical parameters,
and reports per-sample as well as aggregate timing stats. Optionally prints the
resulting policies for manual sanity checks.

Usage:
    python -m tools.benchmark_mcts --samples 10 --sims 128 --device cpu
    python -m tools.benchmark_mcts --samples 20 --sims 256 --device cuda --timing
"""

from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from statistics import fmean

import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import MCTS as LegacyMCTS
from src.neural_network import ChessNet
from v0.python.mcts import MCTS as V0MCTS

_DEBUG_ENABLED = bool(os.environ.get("V0_MCTS_DEBUG"))

if _DEBUG_ENABLED:
    import faulthandler
    faulthandler.enable()


def debug(msg: str) -> None:
    if _DEBUG_ENABLED:
        print(f"[debug] {msg}")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark legacy vs v0 MCTS search throughput."
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of random game states to benchmark.")
    parser.add_argument("--max-moves", type=int, default=60, help="Random plies per sampled state.")
    parser.add_argument("--sims", type=int, default=128, help="Number of simulations per search.")
    parser.add_argument("--batch-size", type=int, default=16, help="Leaf batch size for both implementations.")
    parser.add_argument("--dump-policies", action="store_true", help="Print move/prob pairs for each run.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for both legacy and v0 (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--inference-backend",
        "--inference_backend",
        type=str,
        default="graph",
        choices=["graph", "ts", "py"],
        help="Inference backend for v0 MCTS: graph|ts|py.",
    )
    parser.add_argument(
        "--torchscript-path",
        "--torchscript_path",
        type=str,
        default=None,
        help="Optional TorchScript path for v0 inference backends.",
    )
    parser.add_argument(
        "--torchscript-dtype",
        "--torchscript_dtype",
        type=str,
        default=None,
        help="Optional TorchScript dtype override (float16/float32/bfloat16).",
    )
    parser.add_argument(
        "--inference-batch-size",
        "--inference_batch_size",
        type=int,
        default=512,
        help="Fixed batch size for graph inference backend.",
    )
    parser.add_argument(
        "--inference-warmup-iters",
        "--inference_warmup_iters",
        type=int,
        default=5,
        help="Warmup iterations inside the graph inference engine.",
    )
    parser.add_argument("--timing", action="store_true", help="Include per-sample timing lines (default summary only).")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for reproducibility.")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip legacy MCTS (benchmark v0 only).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = random.Random(args.seed + 1)
    device = torch.device(args.device)
    model = ChessNet(board_size=GameState.BOARD_SIZE).to(device)
    model.eval()

    legacy_times: list[float] = []
    v0_times: list[float] = []
    speedups: list[float] = []

    for i in range(args.samples):
        state = _random_state(args.max_moves, rng)
        if args.timing:
            print(f"[sample {i}] start (move_count={state.move_count}, phase={state.phase})")
        debug(
            f"sample {i} state info: move_count={state.move_count} phase={state.phase} "
            f"pending_marks={state.pending_marks_remaining}/{state.pending_marks_required} "
            f"pending_captures={state.pending_captures_remaining}/{state.pending_captures_required}"
        )

        legacy_elapsed = None
        if not args.skip_legacy:
            legacy = LegacyMCTS(
                model=model,
                num_simulations=args.sims,
                exploration_weight=1.0,
                temperature=1.0,
                device=args.device,
                add_dirichlet_noise=False,
                virtual_loss_weight=0.0,
                batch_K=args.batch_size,
                verbose=False,
            )

            try:
                debug(f"sample {i} legacy search begin")
                start = time.perf_counter()
                legacy_moves, legacy_policy = legacy.search(state)
                legacy_elapsed = time.perf_counter() - start
                debug(f"sample {i} legacy search end ({legacy_elapsed:.3f}s)")
            except Exception as exc:
                print(f"[sample {i}] legacy search failed: {exc}")
                traceback.print_exc()
                continue

        v0_mcts = V0MCTS(
            model=model,
            num_simulations=args.sims,
            exploration_weight=1.0,
            temperature=1.0,
            device=args.device,
            add_dirichlet_noise=False,
            seed=args.seed,
            batch_K=args.batch_size,
            inference_backend=args.inference_backend,
            torchscript_path=args.torchscript_path,
            torchscript_dtype=args.torchscript_dtype,
            inference_batch_size=args.inference_batch_size,
            inference_warmup_iters=args.inference_warmup_iters,
        )

        try:
            debug(f"sample {i} v0 search begin")
            start = time.perf_counter()
            v0_moves, v0_policy = v0_mcts.search(state)
            v0_elapsed = time.perf_counter() - start
            debug(f"sample {i} v0 search end ({v0_elapsed:.3f}s)")
        except Exception as exc:
            print(f"[sample {i}] v0 search failed: {exc}")
            traceback.print_exc()
            continue

        if legacy_elapsed is not None:
            legacy_times.append(legacy_elapsed)
        v0_times.append(v0_elapsed)

        if legacy_elapsed is not None and v0_elapsed > 0:
            speedup = legacy_elapsed / v0_elapsed
            speedups.append(speedup)

        if args.timing:
            legacy_str = f"legacy={legacy_elapsed:.3f}s " if legacy_elapsed else ""
            speedup_str = f"speedup={speedup:.2f}x" if legacy_elapsed else ""
            print(f"[sample {i}] {legacy_str}v0={v0_elapsed:.3f}s {speedup_str}")

        if args.dump_policies:
            if legacy_elapsed is not None:
                print("  legacy policy:")
                for move, prob in zip(legacy_moves, legacy_policy):
                    print(f"    {prob:.6f} -> {move}")
            print("  v0 policy:")
            for move, prob in zip(v0_moves, v0_policy):
                print(f"    {prob:.6f} -> {move}")

    def summarize(label: str, values: list[float]) -> None:
        if not values:
            print(f"{label}: n/a")
            return
        arr = np.array(values, dtype=float)
        print(
            f"{label}: avg={arr.mean():.3f}s median={np.median(arr):.3f}s "
            f"std={arr.std():.3f}s min={arr.min():.3f}s max={arr.max():.3f}s"
        )

    print("\n[benchmark_mcts] timing summary")
    if legacy_times:
        summarize("legacy", legacy_times)
    summarize("v0", v0_times)
    if speedups:
        speed_arr = np.array(speedups, dtype=float)
        print(
            f"speedup: avg={fmean(speedups):.2f}x median={np.median(speed_arr):.2f}x "
            f"min={speed_arr.min():.2f}x max={speed_arr.max():.2f}x"
        )


if __name__ == "__main__":
    main()
