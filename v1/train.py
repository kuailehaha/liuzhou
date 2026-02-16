"""v1 training entrypoint: GPU-first self-play + tensor-native training."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.self_play_gpu_runner import self_play_v1_gpu
from v1.python.train_bridge import train_network_from_tensors


def train_pipeline_v1(
    iterations: int = 10,
    self_play_games: int = 8,
    mcts_simulations: int = 64,
    batch_size: int = 512,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    temperature_threshold: int = 10,
    exploration_weight: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    checkpoint_dir: str = "./checkpoints_v1",
    device: str = "cpu",
    soft_value_k: float = 2.0,
    max_game_plies: int = 512,
    load_checkpoint: Optional[str] = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    device_obj = torch.device(device)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)

    if load_checkpoint and os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint, map_location=device_obj)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[v1.train] loaded checkpoint: {load_checkpoint}")

    model.to(device_obj)
    metrics = []

    for iteration in range(iterations):
        it_idx = iteration + 1
        print(f"\n[v1.train] ===== Iteration {it_idx}/{iterations} =====")

        sp_start = time.perf_counter()
        model.eval()
        samples, sp_stats = self_play_v1_gpu(
            model=model,
            num_games=self_play_games,
            mcts_simulations=mcts_simulations,
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            temperature_threshold=temperature_threshold,
            exploration_weight=exploration_weight,
            device=str(device_obj),
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            soft_value_k=soft_value_k,
            max_game_plies=max_game_plies,
            sample_moves=True,
            verbose=False,
        )
        sp_elapsed = time.perf_counter() - sp_start

        train_start = time.perf_counter()
        model.train()
        model, train_metrics = train_network_from_tensors(
            model=model,
            samples=samples,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            soft_label_alpha=0.0,
            policy_draw_weight=1.0,
            device=str(device_obj),
            use_amp=(device_obj.type == "cuda"),
        )
        train_elapsed = time.perf_counter() - train_start

        ckpt_path = os.path.join(checkpoint_dir, f"model_iter_{it_idx:03d}.pt")
        torch.save(
            {
                "iteration": it_idx,
                "model_state_dict": model.state_dict(),
                "board_size": GameState.BOARD_SIZE,
                "num_input_channels": NUM_INPUT_CHANNELS,
            },
            ckpt_path,
        )

        last_epoch = (train_metrics.get("epoch_stats") or [{}])[-1]
        entry: Dict[str, Any] = {
            "iteration": it_idx,
            "self_play_games": sp_stats.num_games,
            "self_play_positions": sp_stats.num_positions,
            "self_play_time_sec": sp_elapsed,
            "self_play_games_per_sec": sp_stats.games_per_sec,
            "self_play_positions_per_sec": sp_stats.positions_per_sec,
            "black_wins": sp_stats.black_wins,
            "white_wins": sp_stats.white_wins,
            "draws": sp_stats.draws,
            "train_time_sec": train_elapsed,
            "train_avg_loss": last_epoch.get("avg_loss"),
            "train_avg_policy_loss": last_epoch.get("avg_policy_loss"),
            "train_avg_value_loss": last_epoch.get("avg_value_loss"),
            "checkpoint": ckpt_path,
        }
        metrics.append(entry)

        print(
            "[v1.train] "
            f"games={sp_stats.num_games} positions={sp_stats.num_positions} "
            f"self_play={sp_elapsed:.2f}s train={train_elapsed:.2f}s "
            f"loss={float(last_epoch.get('avg_loss') or 0.0):.4f}"
        )

    metrics_path = os.path.join(checkpoint_dir, "training_metrics_v1.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[v1.train] metrics saved: {metrics_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train with v1 GPU-first self-play pipeline.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--self_play_games", type=int, default=4)
    parser.add_argument("--mcts_simulations", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature_init", type=float, default=1.0)
    parser.add_argument("--temperature_final", type=float, default=0.1)
    parser.add_argument("--temperature_threshold", type=int, default=10)
    parser.add_argument("--exploration_weight", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--soft_value_k", type=float, default=2.0)
    parser.add_argument("--max_game_plies", type=int, default=512)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_v1")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_pipeline_v1(
        iterations=args.iterations,
        self_play_games=args.self_play_games,
        mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature_init=args.temperature_init,
        temperature_final=args.temperature_final,
        temperature_threshold=args.temperature_threshold,
        exploration_weight=args.exploration_weight,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        soft_value_k=args.soft_value_k,
        max_game_plies=args.max_game_plies,
        load_checkpoint=args.load_checkpoint,
    )
