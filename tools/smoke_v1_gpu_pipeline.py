"""Smoke benchmark for v1 GPU-first self-play + tensor-native training."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.self_play_gpu_runner import self_play_v1_gpu
from v1.python.train_bridge import train_network_from_tensors


@dataclass
class PowerStats:
    avg_util: float
    avg_power: float
    samples: int


class GPUPowerSampler:
    def __init__(self, gpu_index: int, interval_sec: float = 0.5) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_sec = max(0.1, float(interval_sec))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.util_samples: List[float] = []
        self.power_samples: List[float] = []

    def _sample_once(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) < 2:
            return
        try:
            util = float(parts[0])
            power = float(parts[1])
        except ValueError:
            return
        self.util_samples.append(util)
        self.power_samples.append(power)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> PowerStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        util = statistics.mean(self.util_samples) if self.util_samples else 0.0
        power = statistics.mean(self.power_samples) if self.power_samples else 0.0
        return PowerStats(avg_util=util, avg_power=power, samples=len(self.util_samples))


def _load_model(checkpoint: Optional[str], device: torch.device) -> ChessNet:
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _run_once(args: argparse.Namespace, model: ChessNet) -> Dict[str, float]:
    sampler = None
    if args.sample_power and torch.cuda.is_available() and str(args.device).startswith("cuda"):
        gpu_idx = 0
        if ":" in str(args.device):
            gpu_idx = int(str(args.device).split(":")[1])
        sampler = GPUPowerSampler(gpu_index=gpu_idx, interval_sec=args.power_interval)
        sampler.start()

    t0 = time.perf_counter()
    batch, sp_stats = self_play_v1_gpu(
        model=model,
        num_games=args.num_games,
        mcts_simulations=args.mcts_simulations,
        temperature_init=args.temperature_init,
        temperature_final=args.temperature_final,
        temperature_threshold=args.temperature_threshold,
        exploration_weight=args.exploration_weight,
        device=str(args.device),
        add_dirichlet_noise=(not args.disable_dirichlet),
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        soft_value_k=args.soft_value_k,
        max_game_plies=args.max_game_plies,
        sample_moves=(not args.greedy),
        verbose=args.verbose,
    )
    self_play_elapsed = time.perf_counter() - t0

    train_elapsed = 0.0
    train_metrics = {"epoch_stats": []}
    if args.epochs > 0 and batch.num_samples > 0:
        t1 = time.perf_counter()
        model.train()
        model, train_metrics = train_network_from_tensors(
            model=model,
            samples=batch,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            soft_label_alpha=args.soft_label_alpha,
            policy_draw_weight=args.policy_draw_weight,
            device=str(args.device),
            use_amp=(not args.disable_amp),
        )
        model.eval()
        train_elapsed = time.perf_counter() - t1

    power_stats = sampler.stop() if sampler is not None else PowerStats(0.0, 0.0, 0)

    last_epoch = train_metrics["epoch_stats"][-1] if train_metrics["epoch_stats"] else {}
    result = {
        "games": float(sp_stats.num_games),
        "positions": float(sp_stats.num_positions),
        "self_play_sec": float(self_play_elapsed),
        "train_sec": float(train_elapsed),
        "games_per_sec": float(sp_stats.games_per_sec),
        "positions_per_sec": float(sp_stats.positions_per_sec),
        "black_wins": float(sp_stats.black_wins),
        "white_wins": float(sp_stats.white_wins),
        "draws": float(sp_stats.draws),
        "avg_game_length": float(sp_stats.avg_game_length),
        "train_avg_loss": float(last_epoch.get("avg_loss", 0.0) or 0.0),
        "train_avg_policy_loss": float(last_epoch.get("avg_policy_loss", 0.0) or 0.0),
        "train_avg_value_loss": float(last_epoch.get("avg_value_loss", 0.0) or 0.0),
        "gpu_util_avg": float(power_stats.avg_util),
        "gpu_power_avg": float(power_stats.avg_power),
        "gpu_power_samples": float(power_stats.samples),
    }
    return result


def _parse_sweep(value: str) -> List[int]:
    return [max(1, int(item.strip())) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke benchmark for v1 GPU-first pipeline.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_games", type=int, default=4)
    parser.add_argument("--mcts_simulations", type=int, default=32)
    parser.add_argument("--temperature_init", type=float, default=1.0)
    parser.add_argument("--temperature_final", type=float, default=0.1)
    parser.add_argument("--temperature_threshold", type=int, default=10)
    parser.add_argument("--exploration_weight", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--disable_dirichlet", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--soft_value_k", type=float, default=2.0)
    parser.add_argument("--max_game_plies", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--soft_label_alpha", type=float, default=0.0)
    parser.add_argument("--policy_draw_weight", type=float, default=1.0)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--sample_power", action="store_true")
    parser.add_argument("--power_interval", type=float, default=0.5)
    parser.add_argument("--thread_sweep", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)

    if args.thread_sweep:
        thread_values = _parse_sweep(args.thread_sweep)
        for threads in thread_values:
            torch.set_num_threads(threads)
            result = _run_once(args, model)
            print(
                f"[threads={threads}] games/s={result['games_per_sec']:.3f} "
                f"pos/s={result['positions_per_sec']:.1f} "
                f"gpu_util={result['gpu_util_avg']:.1f}% "
                f"gpu_power={result['gpu_power_avg']:.1f}W "
                f"self_play={result['self_play_sec']:.2f}s train={result['train_sec']:.2f}s"
            )
        return

    result = _run_once(args, model)
    print(
        f"games={int(result['games'])} positions={int(result['positions'])} "
        f"games/s={result['games_per_sec']:.3f} pos/s={result['positions_per_sec']:.1f}"
    )
    print(
        f"self_play={result['self_play_sec']:.2f}s train={result['train_sec']:.2f}s "
        f"W/L/D={int(result['black_wins'])}/{int(result['white_wins'])}/{int(result['draws'])} "
        f"avg_len={result['avg_game_length']:.1f}"
    )
    print(
        f"train_loss={result['train_avg_loss']:.4f} "
        f"policy_loss={result['train_avg_policy_loss']:.4f} "
        f"value_loss={result['train_avg_value_loss']:.4f}"
    )
    if args.sample_power:
        print(
            f"gpu_util_avg={result['gpu_util_avg']:.1f}% "
            f"gpu_power_avg={result['gpu_power_avg']:.1f}W "
            f"samples={int(result['gpu_power_samples'])}"
        )


if __name__ == "__main__":
    main()

