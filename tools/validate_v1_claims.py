#!/usr/bin/env python
"""Validation matrix for v1 GPU-first claims.

This script compares v0 and v1 on the same machine and reports:
1) Throughput change (games/s, positions/s).
2) GPU util/power behavior.
3) CPU-scaling sensitivity (v0 workers vs v1 CPU threads).

It targets the acceptance criteria documented in `v1/Design.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import v0_core
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v0.python.self_play_runner import self_play_v0
from v1.python.self_play_gpu_runner import self_play_v1_gpu


def _ensure_v0_binary_compat() -> bool:
    """Patch old v0 binaries missing `MCTSConfig.max_actions_per_batch`."""
    if hasattr(v0_core.MCTSConfig, "max_actions_per_batch"):
        return False
    v0_core.MCTSConfig.max_actions_per_batch = property(  # type: ignore[attr-defined]
        lambda self: 0,
        lambda self, value: None,
    )
    return True


@dataclass
class RunRow:
    mode: str
    scale: int
    games: int
    positions: int
    elapsed_sec: float
    games_per_sec: float
    positions_per_sec: float
    gpu_util_avg: float
    gpu_util_max: float
    gpu_power_avg_w: float
    gpu_power_max_w: float
    gpu_mem_avg_mib: float
    gpu_mem_max_mib: float


class GPUSampler:
    def __init__(self, gpu_index: int, interval_sec: float = 0.2) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_sec = max(0.05, float(interval_sec))
        self.samples: List[Tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 3:
                        self.samples.append((float(parts[0]), float(parts[1]), float(parts[2])))
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
            nan = float("nan")
            return {
                "gpu_util_avg": nan,
                "gpu_util_max": nan,
                "gpu_power_avg_w": nan,
                "gpu_power_max_w": nan,
                "gpu_mem_avg_mib": nan,
                "gpu_mem_max_mib": nan,
            }
        util = [x[0] for x in self.samples]
        power = [x[1] for x in self.samples]
        mem = [x[2] for x in self.samples]
        return {
            "gpu_util_avg": float(statistics.fmean(util)),
            "gpu_util_max": float(max(util)),
            "gpu_power_avg_w": float(statistics.fmean(power)),
            "gpu_power_max_w": float(max(power)),
            "gpu_mem_avg_mib": float(statistics.fmean(mem)),
            "gpu_mem_max_mib": float(max(mem)),
        }


def _parse_scale_list(raw: str, default: Sequence[int]) -> List[int]:
    values: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            num = int(token)
        except ValueError:
            continue
        if num > 0:
            values.append(num)
    if not values:
        values = list(default)
    seen = set()
    dedup = []
    for value in values:
        if value not in seen:
            dedup.append(value)
            seen.add(value)
    return dedup


def _query_gpu_processes(gpu_index: int) -> List[Tuple[int, str, float]]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows: List[Tuple[int, str, float]] = []
    for line in out.splitlines():
        text = line.strip()
        if not text or text.lower() == "no running processes found":
            continue
        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            mem_mib = float(parts[2])
        except ValueError:
            continue
        rows.append((pid, parts[1], mem_mib))
    return rows


def _current_pid() -> int:
    return int(os.getpid())


def _check_background_load(gpu_index: int, allow_external_gpu_procs: int) -> None:
    rows = _query_gpu_processes(gpu_index)
    current = _current_pid()
    external = [row for row in rows if row[0] != current]
    if len(external) > int(allow_external_gpu_procs):
        formatted = ", ".join(f"{pid}:{name}:{mem:.0f}MiB" for pid, name, mem in external)
        raise RuntimeError(
            "GPU seems busy before benchmark. "
            f"external_processes={len(external)}, allowed={allow_external_gpu_procs}, detail=[{formatted}]"
        )


def _resolve_gpu_index(device: str) -> int:
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0
    return 0 if dev.index is None else int(dev.index)


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a / b)


def _gain_ratio(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    first = float(values[0])
    if first <= 0:
        return float("nan")
    best = max(float(v) for v in values)
    return (best - first) / first


def _run_v0_case(
    model: ChessNet,
    device: str,
    workers: int,
    total_games: int,
    mcts_simulations: int,
    batch_leaves: int,
    inference_backend: str,
    sampler_interval: float,
    gpu_index: int,
) -> RunRow:
    games_per_worker = max(1, int(math.ceil(total_games / workers)))
    sampler = GPUSampler(gpu_index=gpu_index, interval_sec=sampler_interval)
    sampler.start()
    t0 = time.perf_counter()
    games = self_play_v0(
        model=model,
        num_games=total_games,
        mcts_simulations=mcts_simulations,
        temperature_init=1.0,
        temperature_final=0.2,
        temperature_threshold=8,
        exploration_weight=1.0,
        device=device,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_leaves=batch_leaves,
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
        inference_backend=inference_backend,
        torchscript_path=None,
        torchscript_dtype="float16",
        inference_batch_size=64,
        inference_warmup_iters=1,
        devices=[device],
    )
    elapsed = time.perf_counter() - t0
    sampler.stop()
    positions = int(sum(len(g[0]) for g in games))
    summ = sampler.summary()
    return RunRow(
        mode="v0",
        scale=int(workers),
        games=int(len(games)),
        positions=positions,
        elapsed_sec=float(elapsed),
        games_per_sec=_safe_div(float(len(games)), float(elapsed)),
        positions_per_sec=_safe_div(float(positions), float(elapsed)),
        gpu_util_avg=float(summ["gpu_util_avg"]),
        gpu_util_max=float(summ["gpu_util_max"]),
        gpu_power_avg_w=float(summ["gpu_power_avg_w"]),
        gpu_power_max_w=float(summ["gpu_power_max_w"]),
        gpu_mem_avg_mib=float(summ["gpu_mem_avg_mib"]),
        gpu_mem_max_mib=float(summ["gpu_mem_max_mib"]),
    )


def _run_v1_case(
    model: ChessNet,
    device: str,
    threads: int,
    total_games: int,
    mcts_simulations: int,
    sampler_interval: float,
    gpu_index: int,
) -> RunRow:
    torch.set_num_threads(int(threads))
    sampler = GPUSampler(gpu_index=gpu_index, interval_sec=sampler_interval)
    sampler.start()
    t0 = time.perf_counter()
    _batch, stats = self_play_v1_gpu(
        model=model,
        num_games=total_games,
        mcts_simulations=mcts_simulations,
        temperature_init=1.0,
        temperature_final=0.2,
        temperature_threshold=8,
        exploration_weight=1.0,
        device=device,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        max_game_plies=512,
        sample_moves=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    sampler.stop()
    summ = sampler.summary()
    return RunRow(
        mode="v1",
        scale=int(threads),
        games=int(stats.num_games),
        positions=int(stats.num_positions),
        elapsed_sec=float(elapsed),
        games_per_sec=float(stats.games_per_sec),
        positions_per_sec=float(stats.positions_per_sec),
        gpu_util_avg=float(summ["gpu_util_avg"]),
        gpu_util_max=float(summ["gpu_util_max"]),
        gpu_power_avg_w=float(summ["gpu_power_avg_w"]),
        gpu_power_max_w=float(summ["gpu_power_max_w"]),
        gpu_mem_avg_mib=float(summ["gpu_mem_avg_mib"]),
        gpu_mem_max_mib=float(summ["gpu_mem_max_mib"]),
    )


def _run_inference_baseline(
    model: ChessNet,
    device: str,
    batch_size: int,
    iters: int,
    sampler_interval: float,
    gpu_index: int,
) -> Dict[str, float]:
    dev = torch.device(device)
    x = torch.randn(
        int(batch_size),
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=dev,
    )
    for _ in range(8):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(x)
    torch.cuda.synchronize(dev)

    sampler = GPUSampler(gpu_index=gpu_index, interval_sec=sampler_interval)
    sampler.start()
    t0 = time.perf_counter()
    for _ in range(int(iters)):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(x)
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    sampler.stop()
    summ = sampler.summary()
    return {
        "batch_size": float(batch_size),
        "iters": float(iters),
        "elapsed_sec": float(elapsed),
        "samples_per_sec": _safe_div(float(batch_size * iters), float(elapsed)),
        **summ,
    }


def _print_rows(title: str, rows: Sequence[RunRow], scale_name: str) -> None:
    print(f"\n{title}")
    for row in rows:
        print(
            f"{scale_name}={row.scale} games={row.games} sec={row.elapsed_sec:.3f} "
            f"games/s={row.games_per_sec:.3f} pos/s={row.positions_per_sec:.1f} "
            f"gpu_util_avg={row.gpu_util_avg:.1f}% gpu_power_avg={row.gpu_power_avg_w:.1f}W"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate v1 GPU-first claims against v0 baseline.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--v0-workers", type=str, default="1,2,4")
    parser.add_argument("--v1-threads", type=str, default="1,2,4")
    parser.add_argument("--total-games", type=int, default=8)
    parser.add_argument("--v0-mcts-simulations", type=int, default=12)
    parser.add_argument("--v1-mcts-simulations", type=int, default=12)
    parser.add_argument("--v0-batch-leaves", type=int, default=32)
    parser.add_argument("--v0-inference-backend", type=str, default="py", choices=["py", "graph", "ts"])
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)
    parser.add_argument("--allow-external-gpu-procs", type=int, default=0)
    parser.add_argument("--with-inference-baseline", action="store_true")
    parser.add_argument("--inference-baseline-batch", type=int, default=4096)
    parser.add_argument("--inference-baseline-iters", type=int, default=120)
    parser.add_argument("--min-v1-speedup", type=float, default=1.20)
    parser.add_argument("--min-v1-power-delta-w", type=float, default=5.0)
    parser.add_argument("--max-v1-thread-gain", type=float, default=0.15)
    parser.add_argument("--min-v0-worker-gain", type=float, default=0.15)
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join("results", f"v1_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero when any criterion fails.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation script.")
    _ensure_v0_binary_compat()

    gpu_index = _resolve_gpu_index(args.device)
    _check_background_load(gpu_index=gpu_index, allow_external_gpu_procs=args.allow_external_gpu_procs)

    v0_workers = _parse_scale_list(args.v0_workers, [1, 2, 4])
    v1_threads = _parse_scale_list(args.v1_threads, [1, 2, 4])

    device = torch.device(args.device)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device).eval()

    print(f"device={args.device} v0_workers={v0_workers} v1_threads={v1_threads}")
    print(f"games={args.total_games} v0_sims={args.v0_mcts_simulations} v1_sims={args.v1_mcts_simulations}")

    rows_v0 = [
        _run_v0_case(
            model=model,
            device=args.device,
            workers=workers,
            total_games=int(args.total_games),
            mcts_simulations=int(args.v0_mcts_simulations),
            batch_leaves=int(args.v0_batch_leaves),
            inference_backend=str(args.v0_inference_backend),
            sampler_interval=float(args.gpu_sample_interval),
            gpu_index=gpu_index,
        )
        for workers in v0_workers
    ]
    rows_v1 = [
        _run_v1_case(
            model=model,
            device=args.device,
            threads=threads,
            total_games=int(args.total_games),
            mcts_simulations=int(args.v1_mcts_simulations),
            sampler_interval=float(args.gpu_sample_interval),
            gpu_index=gpu_index,
        )
        for threads in v1_threads
    ]

    _print_rows("[v0] worker scaling", rows_v0, "workers")
    _print_rows("[v1] CPU-thread sensitivity", rows_v1, "threads")

    baseline = None
    if args.with_inference_baseline:
        baseline = _run_inference_baseline(
            model=model,
            device=args.device,
            batch_size=int(args.inference_baseline_batch),
            iters=int(args.inference_baseline_iters),
            sampler_interval=float(args.gpu_sample_interval),
            gpu_index=gpu_index,
        )
        print(
            "\n[inference baseline] "
            f"samples/s={baseline['samples_per_sec']:.1f} "
            f"gpu_util_avg={baseline['gpu_util_avg']:.1f}% "
            f"gpu_power_avg={baseline['gpu_power_avg_w']:.1f}W"
        )

    v0_games_per_sec = [row.games_per_sec for row in rows_v0]
    v1_games_per_sec = [row.games_per_sec for row in rows_v1]

    v0_worker_gain = _gain_ratio(v0_games_per_sec)
    v1_thread_gain = _gain_ratio(v1_games_per_sec)
    speedup = _safe_div(max(v1_games_per_sec), v0_games_per_sec[0])
    power_delta = max(row.gpu_power_avg_w for row in rows_v1) - rows_v0[0].gpu_power_avg_w

    criteria = {
        "v0_worker_gain_ge_threshold": bool(v0_worker_gain >= float(args.min_v0_worker_gain)),
        "v1_speedup_ge_threshold": bool(speedup >= float(args.min_v1_speedup)),
        "v1_power_delta_ge_threshold": bool(power_delta >= float(args.min_v1_power_delta_w)),
        "v1_thread_gain_le_threshold": bool(v1_thread_gain <= float(args.max_v1_thread_gain)),
    }

    summary = {
        "v0_worker_gain": float(v0_worker_gain),
        "v1_thread_gain": float(v1_thread_gain),
        "speedup_best_v1_vs_v0_worker1": float(speedup),
        "power_delta_best_v1_minus_v0_worker1_w": float(power_delta),
        "thresholds": {
            "min_v0_worker_gain": float(args.min_v0_worker_gain),
            "min_v1_speedup": float(args.min_v1_speedup),
            "min_v1_power_delta_w": float(args.min_v1_power_delta_w),
            "max_v1_thread_gain": float(args.max_v1_thread_gain),
        },
        "criteria": criteria,
    }

    print("\n[summary]")
    print(
        f"v0_worker_gain={summary['v0_worker_gain']:.3f}, "
        f"v1_thread_gain={summary['v1_thread_gain']:.3f}, "
        f"speedup={summary['speedup_best_v1_vs_v0_worker1']:.3f}, "
        f"power_delta_w={summary['power_delta_best_v1_minus_v0_worker1_w']:.2f}"
    )
    print("[criteria]")
    for key, passed in criteria.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "v0_rows": [asdict(row) for row in rows_v0],
        "v1_rows": [asdict(row) for row in rows_v1],
        "inference_baseline": baseline,
        "summary": summary,
    }

    out_path = args.output_json
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nreport_saved={out_path}")

    if args.strict and not all(criteria.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

