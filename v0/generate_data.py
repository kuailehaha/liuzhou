"""Standalone script to generate self-play samples and store them on disk."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from v0.python.self_play_runner import self_play_v0
from v0.python.state_io import flatten_training_games, sample_to_record, write_records_to_jsonl


def _default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "self_play")


def generate_samples(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)

    metadata: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "device": str(device),
        "num_games": args.num_games,
        "args": vars(args),
    }

    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        metadata["model_checkpoint"] = os.path.abspath(args.model_checkpoint)
        metadata["model_iteration"] = checkpoint.get("iteration")
    model.to(device)
    model.eval()

    games_per_worker = args.games_per_worker
    if games_per_worker is None:
        games_per_worker = max(1, math.ceil(args.num_games / max(1, args.self_play_workers)))

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
        virtual_loss=args.virtual_loss,
        opening_random_moves=args.opening_random_moves,
        resign_threshold=args.resign_threshold,
        resign_min_moves=args.resign_min_moves,
        resign_consecutive=args.resign_consecutive,
        num_workers=args.self_play_workers,
        games_per_worker=games_per_worker,
        base_seed=(None if args.base_seed == 0 else args.base_seed),
        soft_value_k=args.soft_value_k,
        mcts_verbose=args.mcts_verbose,
        verbose=args.verbose,
        inference_backend=args.inference_backend,
        torchscript_path=args.torchscript_path,
        torchscript_dtype=args.torchscript_dtype,
        inference_batch_size=args.inference_batch_size,
        inference_warmup_iters=args.inference_warmup_iters,
    )

    samples = list(flatten_training_games(training_data))
    record_iter = (sample_to_record(*sample) for sample in samples)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.output_prefix}_{timestamp}"
    data_path = os.path.join(args.output_dir, base_name + ".jsonl")
    count = write_records_to_jsonl(record_iter, data_path)
    metadata["samples"] = count

    meta_path = data_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=False)

    print(f"Wrote {count} samples to {data_path}")
    print(f"Metadata saved to {meta_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v0 self-play data to JSONL files.")
    parser.add_argument("--num_games", type=int, default=32, help="Total number of self-play games to generate.")
    parser.add_argument("--mcts_simulations", type=int, default=800, help="MCTS simulations per move.")
    parser.add_argument("--temperature_init", type=float, default=1.0, help="Initial temperature for move sampling.")
    parser.add_argument("--temperature_final", type=float, default=0.1, help="Final temperature after threshold.")
    parser.add_argument("--temperature_threshold", type=int, default=10, help="Move count before lowering temperature.")
    parser.add_argument("--exploration_weight", type=float, default=1.0, help="PUCT exploration constant.")
    parser.add_argument("--soft_value_k", type=float, default=2.0, help="Soft value scaling factor.")
    parser.add_argument("--self_play_workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--games_per_worker",
        type=int,
        default=None,
        help="Override number of games per worker (default auto-computed).",
    )
    parser.add_argument("--batch_leaves", type=int, default=256, help="Number of batched leaf evaluations per search.")
    parser.add_argument("--virtual_loss", type=float, default=1.0, help="Virtual loss weight for v0 MCTS.")
    parser.add_argument(
        "--opening_random_moves",
        type=int,
        default=4,
        help="Number of opening moves to sample uniformly at random.",
    )
    parser.add_argument(
        "--resign_threshold",
        type=float,
        default=-0.8,
        help="Resign when root value <= threshold (set >=0 to disable).",
    )
    parser.add_argument(
        "--resign_min_moves",
        type=int,
        default=10,
        help="Disable resign for the first N moves.",
    )
    parser.add_argument(
        "--resign_consecutive",
        type=int,
        default=3,
        help="Require this many consecutive low-value steps to resign.",
    )
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet epsilon blend.")
    parser.add_argument(
        "--disable_dirichlet_noise",
        action="store_true",
        help="Disable Dirichlet noise injection at the root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument("--base_seed", type=int, default=0, help="Base RNG seed (0 = auto).")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Optional model checkpoint to load.")
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(), help="Destination directory for JSONL.")
    parser.add_argument("--output_prefix", type=str, default="self_play", help="Prefix for output filenames.")
    parser.add_argument("--verbose", action="store_true", help="Print per-move information during generation.")
    parser.add_argument("--mcts_verbose", action="store_true", help="Enable verbose logging inside v0 MCTS.")
    parser.add_argument(
        "--inference_backend",
        "--inference-backend",
        type=str,
        default="graph",
        choices=["graph", "ts", "py"],
        help="Inference backend for v0 MCTS: graph|ts|py.",
    )
    parser.add_argument(
        "--torchscript_path",
        "--torchscript-path",
        type=str,
        default=None,
        help="Optional TorchScript path for v0 inference backends.",
    )
    parser.add_argument(
        "--torchscript_dtype",
        "--torchscript-dtype",
        type=str,
        default=None,
        help="Optional TorchScript dtype override (float16/float32/bfloat16).",
    )
    parser.add_argument(
        "--inference_batch_size",
        "--inference-batch-size",
        type=int,
        default=512,
        help="Fixed batch size for graph inference backend.",
    )
    parser.add_argument(
        "--inference_warmup_iters",
        "--inference-warmup-iters",
        type=int,
        default=5,
        help="Warmup iterations inside the graph inference engine.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    if cli_args.num_games <= 0:
        raise ValueError("num_games must be positive.")
    start = time.time()
    generate_samples(cli_args)
    elapsed = time.time() - start
    print(f"Generation finished in {elapsed:.2f}s")
