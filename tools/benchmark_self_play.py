"""Benchmark legacy vs v0 self-play pipelines under identical settings."""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from src.mcts import self_play as legacy_self_play

from v0.python.self_play_runner import self_play_v0


def _load_model(device: str, checkpoint: Optional[str]) -> ChessNet:
    """Load a ChessNet (optionally from a checkpoint) and move it to device."""
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    torch_device = torch.device(device)
    model.to(torch_device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=torch_device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _auto_games_per_worker(num_games: int, num_workers: int, override: Optional[int]) -> Optional[int]:
    if num_workers <= 1:
        return override
    if override is not None and override > 0:
        return override
    return max(1, math.ceil(num_games / num_workers))


def _summarize_result(
    label: str,
    start_time: float,
    end_time: float,
    training_data: Sequence[Tuple[List[GameState], List[Any], float, float]],
) -> Dict[str, Any]:
    duration = end_time - start_time
    num_games = len(training_data)
    total_positions = sum(len(game_states) for game_states, _, _, _ in training_data)
    return {
        "label": label,
        "duration_sec": duration,
        "num_games": num_games,
        "positions": total_positions,
        "games_per_sec": (num_games / duration) if duration > 0 else None,
        "positions_per_sec": (total_positions / duration) if duration > 0 else None,
    }


def run_legacy(args: argparse.Namespace, model: ChessNet, games_per_worker: Optional[int]) -> Dict[str, Any]:
    if args.skip_legacy:
        return {}
    start = time.perf_counter()
    training_data = legacy_self_play(
        model=model,
        num_games=args.num_games,
        mcts_simulations=args.mcts_simulations,
        temperature_init=args.temperature_init,
        temperature_final=args.temperature_final,
        temperature_threshold=args.temperature_threshold,
        exploration_weight=args.exploration_weight,
        device=args.device,
        add_dirichlet_noise=not args.disable_dirichlet_noise,
        mcts_verbose=args.legacy_mcts_verbose,
        verbose=args.verbose,
        num_workers=args.num_workers,
        games_per_worker=games_per_worker,
        base_seed=(None if args.base_seed == 0 else args.base_seed),
        virtual_loss_weight=args.legacy_virtual_loss,
        soft_value_k=args.soft_value_k,
    )
    end = time.perf_counter()
    return _summarize_result("legacy", start, end, training_data)


def run_v0(args: argparse.Namespace, model: ChessNet, games_per_worker: Optional[int]) -> Dict[str, Any]:
    if args.skip_v0:
        return {}
    start = time.perf_counter()
    training_data = self_play_v0(
        model=model,
        num_games=args.num_games,
        mcts_simulations=args.mcts_simulations,
        temperature_init=args.temperature_init,
        temperature_final=args.temperature_final,
        temperature_threshold=args.temperature_threshold,
        exploration_weight=args.exploration_weight,
        device=args.device,
        add_dirichlet_noise=not args.disable_dirichlet_noise,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        batch_leaves=args.batch_leaves,
        virtual_loss=args.v0_virtual_loss,
        opening_random_moves=args.opening_random_moves,
        resign_threshold=args.resign_threshold,
        resign_min_moves=args.resign_min_moves,
        resign_consecutive=args.resign_consecutive,
        num_workers=args.num_workers,
        games_per_worker=games_per_worker,
        base_seed=(None if args.base_seed == 0 else args.base_seed),
        soft_value_k=args.soft_value_k,
        mcts_verbose=args.v0_mcts_verbose,
        verbose=args.verbose,
    )
    end = time.perf_counter()
    return _summarize_result("v0", start, end, training_data)


def print_summary(results: Sequence[Dict[str, Any]]) -> None:
    print("\nBenchmark results:")
    print("=" * 72)
    for res in results:
        if not res:
            continue
        print(f"[{res['label']}] duration={res['duration_sec']:.2f}s | "
              f"games={res['num_games']} ({res['games_per_sec']:.2f} games/s) | "
              f"positions={res['positions']} ({res['positions_per_sec']:.2f} positions/s)")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare legacy and v0 self-play runtimes under identical settings."
    )
    parser.add_argument("--num-games", type=int, default=16, help="Number of self-play games per run.")
    parser.add_argument("--mcts-simulations", type=int, default=200, help="MCTS simulations per move.")
    parser.add_argument("--temperature-init", type=float, default=1.0, help="Initial sampling temperature.")
    parser.add_argument("--temperature-final", type=float, default=0.1, help="Final sampling temperature.")
    parser.add_argument("--temperature-threshold", type=int, default=10, help="Move threshold for temperature drop.")
    parser.add_argument("--exploration-weight", type=float, default=1.0, help="PUCT exploration constant.")
    parser.add_argument("--soft-value-k", type=float, default=2.0, help="Soft value scaling factor.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers (applied to both pipelines).")
    parser.add_argument(
        "--games-per-worker",
        type=int,
        default=None,
        help="Override games per worker (auto-computed if omitted).",
    )
    parser.add_argument("--batch-leaves", type=int, default=256, help="v0 leaf batch size.")
    parser.add_argument("--opening-random-moves", type=int, default=4, help="Opening random moves for v0.")
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=-0.8,
        help="Resign when root value <= threshold (set >=0 to disable).",
    )
    parser.add_argument(
        "--resign-min-moves",
        type=int,
        default=10,
        help="Disable resign for the first N moves.",
    )
    parser.add_argument(
        "--resign-consecutive",
        type=int,
        default=3,
        help="Require this many consecutive low-value steps to resign.",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for v0.")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Dirichlet epsilon for v0.")
    parser.add_argument("--disable-dirichlet-noise", action="store_true", help="Disable root Dirichlet noise.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-seed", type=int, default=0, help="Base RNG seed (0 = auto).")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Optional checkpoint for the model.")
    parser.add_argument("--legacy-virtual-loss", type=float, default=0.0, help="Legacy virtual loss weight.")
    parser.add_argument("--v0-virtual-loss", type=float, default=1.0, help="v0 virtual loss weight.")
    parser.add_argument("--legacy-mcts-verbose", action="store_true", help="Verbose logging for legacy MCTS.")
    parser.add_argument("--v0-mcts-verbose", action="store_true", help="Verbose logging for v0 MCTS.")
    parser.add_argument("--verbose", action="store_true", help="Print per-move logging (both pipelines).")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip the legacy baseline run.")
    parser.add_argument("--skip-v0", action="store_true", help="Skip the v0 run.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_games <= 0:
        raise ValueError("num-games must be positive.")
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive.")

    games_per_worker = _auto_games_per_worker(args.num_games, args.num_workers, args.games_per_worker)
    if args.num_workers > 1 and games_per_worker is None:
        raise ValueError("games-per-worker must be provided (or auto-computable) when num-workers > 1.")

    model = _load_model(args.device, args.model_checkpoint)

    results: List[Dict[str, Any]] = []
    legacy_summary = run_legacy(args, model, games_per_worker)
    if legacy_summary:
        results.append(legacy_summary)

    v0_summary = run_v0(args, model, games_per_worker)
    if v0_summary:
        results.append(v0_summary)

    if not results:
        print("No benchmark executed (both runs skipped).")
        return

    print_summary(results)


if __name__ == "__main__":
    main()
