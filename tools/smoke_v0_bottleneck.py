#!/usr/bin/env python
"""Quick smoke benchmark for v0 self-play CPU/GPU bottleneck signals.

This script measures:
1) Self-play throughput under different CPU worker counts.
2) GPU utilization/power during self-play.
3) Optional pure inference baseline (same model) to contrast GPU headroom.

Typical usage on Windows:
    set PYTHONPATH=d:/CODES/liuzhou/build/v0/src;d:/CODES/liuzhou
    conda run -n torchenv python tools/smoke_v0_bottleneck.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import threading
import time
from typing import Dict, List

import torch

import v0_core
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v0.python.self_play_runner import self_play_v0


def _ensure_v0_binary_compat() -> bool:
    """Patch old v0_core binaries missing `MCTSConfig.max_actions_per_batch`.

    Returns True if a runtime compatibility patch was applied.
    """
    if hasattr(v0_core.MCTSConfig, "max_actions_per_batch"):
        return False
    v0_core.MCTSConfig.max_actions_per_batch = property(  # type: ignore[attr-defined]
        lambda self: 0,
        lambda self, value: None,
    )
    return True


COMPAT_PATCH_APPLIED = _ensure_v0_binary_compat()


class GPUSampler:
    def __init__(self, interval_sec: float = 0.2) -> None:
        self.interval_sec = max(0.05, float(interval_sec))
        self.samples: List[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 3:
                        util = float(parts[0])
                        power = float(parts[1])
                        mem = float(parts[2])
                        self.samples.append((util, power, mem))
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {
                "gpu_samples": 0.0,
                "gpu_util_avg": float("nan"),
                "gpu_util_max": float("nan"),
                "gpu_power_avg_w": float("nan"),
                "gpu_power_max_w": float("nan"),
                "gpu_mem_avg_mib": float("nan"),
                "gpu_mem_max_mib": float("nan"),
            }

        util = [x[0] for x in self.samples]
        power = [x[1] for x in self.samples]
        mem = [x[2] for x in self.samples]
        return {
            "gpu_samples": float(len(self.samples)),
            "gpu_util_avg": statistics.fmean(util),
            "gpu_util_max": max(util),
            "gpu_power_avg_w": statistics.fmean(power),
            "gpu_power_max_w": max(power),
            "gpu_mem_avg_mib": statistics.fmean(mem),
            "gpu_mem_max_mib": max(mem),
        }


def _parse_workers(raw: str) -> List[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    values: List[int] = []
    for item in items:
        try:
            w = int(item)
        except ValueError:
            continue
        if w > 0:
            values.append(w)
    seen = set()
    dedup: List[int] = []
    for w in values:
        if w not in seen:
            dedup.append(w)
            seen.add(w)
    return dedup or [1, 2, 4]


def run_self_play_case(model: ChessNet, args: argparse.Namespace, workers: int) -> Dict[str, float]:
    games_per_worker = max(1, int(math.ceil(args.total_games / workers)))
    sampler = GPUSampler(interval_sec=args.gpu_sample_interval)
    sampler.start()
    start = time.perf_counter()
    games = self_play_v0(
        model=model,
        num_games=args.total_games,
        mcts_simulations=args.mcts_simulations,
        temperature_init=1.0,
        temperature_final=0.2,
        temperature_threshold=8,
        exploration_weight=1.0,
        device=args.device,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_leaves=args.batch_leaves,
        virtual_loss=1.0,
        num_workers=workers,
        games_per_worker=games_per_worker,
        base_seed=12345,
        soft_value_k=2.0,
        mcts_verbose=False,
        verbose=False,
        opening_random_moves=2,
        resign_threshold=-0.95,
        resign_min_moves=30,
        resign_consecutive=3,
        inference_backend=args.inference_backend,
        torchscript_path=None,
        torchscript_dtype=args.torchscript_dtype,
        inference_batch_size=args.inference_batch_size,
        inference_warmup_iters=args.inference_warmup_iters,
        devices=[args.device],
    )
    elapsed = time.perf_counter() - start
    sampler.stop()
    positions = sum(len(g[0]) for g in games)
    row: Dict[str, float] = {
        "workers": float(workers),
        "games": float(len(games)),
        "positions": float(positions),
        "elapsed_sec": elapsed,
        "games_per_sec": len(games) / elapsed if elapsed > 0 else 0.0,
        "positions_per_sec": positions / elapsed if elapsed > 0 else 0.0,
    }
    row.update(sampler.summary())
    return row


def run_inference_baseline(model: ChessNet, args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    batch_size = int(args.inference_baseline_batch)
    iters = int(args.inference_baseline_iters)
    x = torch.randn(
        batch_size,
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=device,
    )
    for _ in range(10):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(x)
    torch.cuda.synchronize(device)

    sampler = GPUSampler(interval_sec=args.gpu_sample_interval)
    sampler.start()
    start = time.perf_counter()
    for _ in range(iters):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(x)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    sampler.stop()

    out: Dict[str, float] = {
        "batch_size": float(batch_size),
        "iters": float(iters),
        "elapsed_sec": elapsed,
        "iters_per_sec": iters / elapsed if elapsed > 0 else 0.0,
        "samples_per_sec": (batch_size * iters) / elapsed if elapsed > 0 else 0.0,
    }
    out.update(sampler.summary())
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke benchmark for v0 bottleneck diagnosis.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=str, default="1,2,4")
    parser.add_argument("--total_games", type=int, default=8)
    parser.add_argument("--mcts_simulations", type=int, default=12)
    parser.add_argument("--batch_leaves", type=int, default=32)
    parser.add_argument("--inference_backend", type=str, default="py", choices=["py", "graph", "ts"])
    parser.add_argument("--torchscript_dtype", type=str, default="float16")
    parser.add_argument("--inference_batch_size", type=int, default=64)
    parser.add_argument("--inference_warmup_iters", type=int, default=1)
    parser.add_argument("--gpu_sample_interval", type=float, default=0.2)
    parser.add_argument("--with_inference_baseline", action="store_true")
    parser.add_argument("--inference_baseline_batch", type=int, default=4096)
    parser.add_argument("--inference_baseline_iters", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke benchmark.")

    workers = _parse_workers(args.workers)

    device = torch.device(args.device)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS).to(device).eval()

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={args.device}")
    print(f"compat_patch_applied={COMPAT_PATCH_APPLIED}")
    print(f"workers={workers} total_games={args.total_games} sims={args.mcts_simulations}")

    rows = [run_self_play_case(model, args, w) for w in workers]
    print("\nSelf-play scale:")
    for row in rows:
        print(
            "workers={w:.0f} games={g:.0f} sec={t:.3f} games/s={gps:.3f} pos/s={pps:.1f} "
            "gpu_util_avg={u:.1f}% gpu_power_avg={p:.1f}W".format(
                w=row["workers"],
                g=row["games"],
                t=row["elapsed_sec"],
                gps=row["games_per_sec"],
                pps=row["positions_per_sec"],
                u=row["gpu_util_avg"],
                p=row["gpu_power_avg_w"],
            )
        )

    if args.with_inference_baseline:
        baseline = run_inference_baseline(model, args)
        print("\nPure inference baseline:")
        print(
            "batch={b:.0f} iters={i:.0f} sec={t:.3f} samples/s={sps:.1f} "
            "gpu_util_avg={u:.1f}% gpu_power_avg={p:.1f}W".format(
                b=baseline["batch_size"],
                i=baseline["iters"],
                t=baseline["elapsed_sec"],
                sps=baseline["samples_per_sec"],
                u=baseline["gpu_util_avg"],
                p=baseline["gpu_power_avg_w"],
            )
        )


if __name__ == "__main__":
    main()
