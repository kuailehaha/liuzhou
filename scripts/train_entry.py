#!/usr/bin/env python
"""Shared training entry for v0/v1 pipelines."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
_EXTRA_PATHS = [
    os.path.join(ROOT_DIR, "build", "v0", "src"),
    os.path.join(ROOT_DIR, "v0", "build", "src"),
]
for _p in _EXTRA_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _run_v0(args: argparse.Namespace) -> int:
    cmd: List[str] = [
        sys.executable,
        "scripts/train_loop.py",
        "--iterations",
        str(int(args.iterations)),
        "--games-per-iter",
        str(int(args.self_play_games)),
        "--mcts-sims",
        str(int(args.mcts_simulations)),
        "--batch-leaves",
        str(int(args.v0_batch_leaves)),
        "--train-epochs",
        str(int(args.epochs)),
        "--train-batch-size",
        str(int(args.batch_size)),
        "--train-lr",
        str(float(args.lr)),
        "--device",
        str(args.device),
        "--data-dir",
        str(args.v0_data_dir),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--eval-games",
        str(int(args.v0_eval_games)),
    ]
    print("[train_entry] dispatch pipeline=v0")
    print("[train_entry] command:", " ".join(cmd))
    return int(subprocess.call(cmd))


def _run_v1(args: argparse.Namespace) -> int:
    from v1.train import train_pipeline_v1

    print("[train_entry] dispatch pipeline=v1")
    train_pipeline_v1(
        stage=str(args.stage),
        iterations=int(args.iterations),
        self_play_games=int(args.self_play_games),
        mcts_simulations=int(args.mcts_simulations),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        temperature_init=float(args.temperature_init),
        temperature_final=float(args.temperature_final),
        temperature_threshold=int(args.temperature_threshold),
        exploration_weight=float(args.exploration_weight),
        dirichlet_alpha=float(args.dirichlet_alpha),
        dirichlet_epsilon=float(args.dirichlet_epsilon),
        checkpoint_dir=str(args.checkpoint_dir),
        device=str(args.device),
        devices=args.devices,
        train_devices=args.train_devices,
        train_strategy=str(args.train_strategy),
        soft_value_k=float(args.soft_value_k),
        max_game_plies=int(args.max_game_plies),
        load_checkpoint=args.load_checkpoint,
        self_play_output=args.self_play_output,
        self_play_input=args.self_play_input,
        self_play_stats_json=args.self_play_stats_json,
        checkpoint_name=args.checkpoint_name,
        metrics_output=args.metrics_output,
        infer_devices=args.infer_devices,
        infer_batch_size=int(args.infer_batch_size),
        infer_warmup_iters=int(args.infer_warmup_iters),
        infer_iters=int(args.infer_iters),
        infer_output=args.infer_output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared training entry for v0/v1 pipelines.")
    parser.add_argument("--pipeline", type=str, choices=["v0", "v1"], required=True)
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "selfplay", "train", "infer"],
        help="v1 stage selector; ignored by v0.",
    )

    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--self_play_games", type=int, default=6400)
    parser.add_argument("--mcts_simulations", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--temperature_init", type=float, default=1.0)
    parser.add_argument("--temperature_final", type=float, default=0.1)
    parser.add_argument("--temperature_threshold", type=int, default=10)
    parser.add_argument("--exploration_weight", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--soft_value_k", type=float, default=2.0)
    parser.add_argument("--max_game_plies", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_shared")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional v1 self-play device list, e.g. cuda:0,cuda:1,cuda:2,cuda:3.",
    )
    parser.add_argument(
        "--train_devices",
        type=str,
        default=None,
        help="Optional v1 training device list; default is single --device.",
    )
    parser.add_argument(
        "--train_strategy",
        type=str,
        default="data_parallel",
        choices=["none", "data_parallel", "ddp"],
        help="v1 training parallel strategy.",
    )
    parser.add_argument(
        "--self_play_output",
        type=str,
        default=None,
        help="v1 self-play payload output path.",
    )
    parser.add_argument(
        "--self_play_input",
        type=str,
        default=None,
        help="v1 self-play payload input path.",
    )
    parser.add_argument(
        "--self_play_stats_json",
        type=str,
        default=None,
        help="v1 self-play stats JSON output path.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="v1 stage=train checkpoint file name.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="v1 metrics output path.",
    )
    parser.add_argument(
        "--infer_devices",
        type=str,
        default=None,
        help="v1 infer stage device list.",
    )
    parser.add_argument("--infer_batch_size", type=int, default=4096)
    parser.add_argument("--infer_warmup_iters", type=int, default=20)
    parser.add_argument("--infer_iters", type=int, default=100)
    parser.add_argument(
        "--infer_output",
        type=str,
        default=None,
        help="v1 infer stage output JSON path.",
    )

    parser.add_argument("--v0_batch_leaves", type=int, default=512)
    parser.add_argument("--v0_eval_games", type=int, default=20)
    parser.add_argument("--v0_data_dir", type=str, default="./v0/data/self_play")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.makedirs(str(args.checkpoint_dir), exist_ok=True)
    if args.pipeline == "v0":
        return _run_v0(args)
    return _run_v1(args)


if __name__ == "__main__":
    raise SystemExit(main())
