#!/usr/bin/env python
"""Run fixed self-play workloads (v0/v1) and export telemetry/stability artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import v0_core
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v0.python.self_play_runner import self_play_v0
from v1.python.self_play_gpu_runner import self_play_v1_gpu


def _ensure_v0_binary_compat() -> bool:
    if hasattr(v0_core.MCTSConfig, "max_actions_per_batch"):
        return False
    v0_core.MCTSConfig.max_actions_per_batch = property(  # type: ignore[attr-defined]
        lambda self: 0,
        lambda self, value: None,
    )
    return True


@dataclass
class IterRow:
    index: int
    mode: str
    started_sec: float
    elapsed_sec: float
    games: int
    positions: int
    games_per_sec: float
    positions_per_sec: float
    step_timing_ms: Dict[str, float]
    step_timing_ratio: Dict[str, float]
    step_timing_calls: Dict[str, int]
    mcts_counters: Dict[str, int]


class GPUSampler:
    def __init__(self, gpu_index: int, interval_sec: float = 0.2) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_sec = max(0.05, float(interval_sec))
        self.samples: List[Tuple[float, float, float, float, str, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,power.draw,memory.used,pstate,clocks.current.graphics,clocks.current.sm",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            ts = time.perf_counter()
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 6:
                        self.samples.append(
                            (
                                ts,
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                str(parts[3]),
                                float(parts[4]),
                                float(parts[5]),
                            )
                        )
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
                "gpu_power_min_w": nan,
                "gpu_power_max_w": nan,
                "gpu_mem_avg_mib": nan,
                "gpu_mem_max_mib": nan,
                "gpu_p0_ratio": nan,
                "gpu_graphics_clock_avg_mhz": nan,
                "gpu_graphics_clock_min_mhz": nan,
                "gpu_graphics_clock_max_mhz": nan,
                "gpu_sm_clock_avg_mhz": nan,
                "gpu_sm_clock_min_mhz": nan,
                "gpu_sm_clock_max_mhz": nan,
                "samples": 0.0,
            }
        util = [x[1] for x in self.samples]
        power = [x[2] for x in self.samples]
        mem = [x[3] for x in self.samples]
        pstate = [str(x[4]).strip().upper() for x in self.samples]
        graphics_clk = [x[5] for x in self.samples]
        sm_clk = [x[6] for x in self.samples]
        p0_ratio = float(sum(1 for x in pstate if x == "P0") / max(1, len(pstate)))
        return {
            "gpu_util_avg": float(statistics.fmean(util)),
            "gpu_util_max": float(max(util)),
            "gpu_power_avg_w": float(statistics.fmean(power)),
            "gpu_power_min_w": float(min(power)),
            "gpu_power_max_w": float(max(power)),
            "gpu_mem_avg_mib": float(statistics.fmean(mem)),
            "gpu_mem_max_mib": float(max(mem)),
            "gpu_p0_ratio": p0_ratio,
            "gpu_graphics_clock_avg_mhz": float(statistics.fmean(graphics_clk)),
            "gpu_graphics_clock_min_mhz": float(min(graphics_clk)),
            "gpu_graphics_clock_max_mhz": float(max(graphics_clk)),
            "gpu_sm_clock_avg_mhz": float(statistics.fmean(sm_clk)),
            "gpu_sm_clock_min_mhz": float(min(sm_clk)),
            "gpu_sm_clock_max_mhz": float(max(sm_clk)),
            "samples": float(len(self.samples)),
        }


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a / b)


def _parse_bool_flag(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean flag value: {value!r}")


def _resolve_gpu_index(device: str) -> int:
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0
    return 0 if dev.index is None else int(dev.index)


def _export_torchscript_for_v1(model: ChessNet, device: str, output_path: str, batch_size: int) -> str:
    dev = torch.device(device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(
        int(batch_size),
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=dev,
    )
    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced.save(str(output))
    return str(output)


def _build_v1_inference_engine(
    model: ChessNet,
    *,
    backend: str,
    device: str,
    batch_size: int,
    warmup_iters: int,
) -> Tuple[Optional[object], Optional[str]]:
    mode = str(backend).strip().lower()
    if mode == "py":
        return None, None
    if mode != "graph":
        raise ValueError(f"Unsupported v1 inference backend: {backend}")
    ts_path = os.path.join("results", "v1_workload_temp_model.ts")
    _export_torchscript_for_v1(model=model, device=device, output_path=ts_path, batch_size=max(1, int(batch_size)))
    dtype = "float16" if torch.device(device).type == "cuda" else "float32"
    engine = v0_core.InferenceEngine(
        ts_path,
        str(device),
        dtype,
        int(batch_size),
        int(NUM_INPUT_CHANNELS),
        int(GameState.BOARD_SIZE),
        int(GameState.BOARD_SIZE),
        int(warmup_iters),
        True,
    )
    return engine, ts_path


def _run_v0_once(model: ChessNet, args: argparse.Namespace, iter_seed: int) -> IterRow:
    workers = int(args.v0_workers)
    games_per_worker = max(1, int(math.ceil(int(args.num_games_per_iter) / max(1, workers))))
    t0 = time.perf_counter()
    games = self_play_v0(
        model=model,
        num_games=int(args.num_games_per_iter),
        mcts_simulations=int(args.mcts_simulations),
        temperature_init=1.0,
        temperature_final=0.2,
        temperature_threshold=8,
        exploration_weight=1.0,
        device=args.device,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_leaves=int(args.v0_batch_leaves),
        virtual_loss=1.0,
        num_workers=workers,
        games_per_worker=games_per_worker,
        base_seed=int(iter_seed),
        soft_value_k=2.0,
        mcts_verbose=False,
        verbose=False,
        opening_random_moves=int(args.v0_opening_random_moves),
        resign_threshold=float(args.v0_resign_threshold),
        resign_min_moves=int(args.v0_resign_min_moves),
        resign_consecutive=int(args.v0_resign_consecutive),
        inference_backend=str(args.v0_inference_backend),
        torchscript_path=None,
        torchscript_dtype="float16",
        inference_batch_size=int(args.v0_inference_batch_size),
        inference_warmup_iters=int(args.v0_inference_warmup_iters),
        devices=[args.device],
    )
    elapsed = max(1e-9, time.perf_counter() - t0)
    positions = int(sum(len(g[0]) for g in games))
    return IterRow(
        index=0,
        mode="v0",
        started_sec=t0,
        elapsed_sec=float(elapsed),
        games=int(len(games)),
        positions=positions,
        games_per_sec=_safe_div(float(len(games)), float(elapsed)),
        positions_per_sec=_safe_div(float(positions), float(elapsed)),
        step_timing_ms={},
        step_timing_ratio={},
        step_timing_calls={},
        mcts_counters={},
    )


def _run_v1_once(
    model: ChessNet,
    args: argparse.Namespace,
    inference_engine,
    iter_seed: int,
) -> IterRow:
    torch.set_num_threads(int(args.v1_threads))
    prev_finalize_graph = os.environ.get("V1_FINALIZE_GRAPH")
    finalize_mode = str(args.v1_finalize_graph).strip().lower()
    if finalize_mode == "auto":
        if "V1_FINALIZE_GRAPH" in os.environ:
            del os.environ["V1_FINALIZE_GRAPH"]
    elif finalize_mode == "on":
        os.environ["V1_FINALIZE_GRAPH"] = "1"
    elif finalize_mode == "off":
        os.environ["V1_FINALIZE_GRAPH"] = "0"
    else:
        raise ValueError(f"Unsupported v1-finalize-graph mode: {args.v1_finalize_graph}")
    t0 = time.perf_counter()
    try:
        _batch, stats = self_play_v1_gpu(
            model=model,
            num_games=int(args.num_games_per_iter),
            mcts_simulations=int(args.mcts_simulations),
            temperature_init=1.0,
            temperature_final=0.2,
            temperature_threshold=8,
            exploration_weight=1.0,
            device=args.device,
            add_dirichlet_noise=True,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            soft_value_k=2.0,
            max_game_plies=int(args.max_game_plies),
            sample_moves=_parse_bool_flag(str(args.v1_sample_moves)),
            concurrent_games=int(args.v1_concurrent_games),
            child_eval_mode=str(args.v1_child_eval_mode),
            inference_engine=inference_engine,
            collect_step_timing=bool(args.collect_step_timing),
            verbose=False,
        )
    finally:
        if prev_finalize_graph is None:
            os.environ.pop("V1_FINALIZE_GRAPH", None)
        else:
            os.environ["V1_FINALIZE_GRAPH"] = prev_finalize_graph
    elapsed = max(1e-9, time.perf_counter() - t0)
    return IterRow(
        index=0,
        mode="v1",
        started_sec=t0,
        elapsed_sec=float(elapsed),
        games=int(stats.num_games),
        positions=int(stats.num_positions),
        games_per_sec=float(stats.games_per_sec),
        positions_per_sec=float(stats.positions_per_sec),
        step_timing_ms={k: float(v) for k, v in stats.step_timing_ms.items()},
        step_timing_ratio={k: float(v) for k, v in stats.step_timing_ratio.items()},
        step_timing_calls={k: int(v) for k, v in stats.step_timing_calls.items()},
        mcts_counters={k: int(v) for k, v in stats.mcts_counters.items()},
    )


def _save_step_breakdown_plots(step_timing_ms: Dict[str, float], step_timing_ratio: Dict[str, float], output_prefix: Path) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    keys = ["root_puct_ms", "pack_writeback_ms", "self_play_step_ms", "finalize_ms"]
    labels = [
        "root_puct_ms",
        "pack_writeback_ms",
        "self_play_step_ms",
        "finalize_ms",
    ]
    values = [float(step_timing_ms.get(k, 0.0)) for k in keys]
    ratios = [float(step_timing_ratio.get(k, 0.0)) for k in keys]

    bar_path = output_prefix.with_name(output_prefix.name + "_step_breakdown_bar.png")
    pie_path = output_prefix.with_name(output_prefix.name + "_step_breakdown_pie.png")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("V1 Step Segment Time (Total ms)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(bar_path)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=120)
    total = sum(ratios)
    if total <= 0.0:
        ax2.text(0.5, 0.5, "No timing data", ha="center", va="center")
        ax2.axis("off")
    else:
        ax2.pie(ratios, labels=labels, autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
    ax2.set_title("V1 Step Segment Share")
    fig2.tight_layout()
    fig2.savefig(pie_path)
    plt.close(fig2)

    return {
        "step_breakdown_bar": str(bar_path),
        "step_breakdown_pie": str(pie_path),
    }


def _save_stability_plot(rows: List[IterRow], gpu_samples: List[Tuple[float, float, float, float, str, float, float]], output_prefix: Path) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    throughput_path = output_prefix.with_name(output_prefix.name + "_stable_throughput.png")
    gpu_path = output_prefix.with_name(output_prefix.name + "_stable_gpu.png")

    if rows:
        t0 = rows[0].started_sec
    elif gpu_samples:
        t0 = gpu_samples[0][0]
    else:
        t0 = time.perf_counter()

    iter_times = [max(0.0, float(r.started_sec - t0)) for r in rows]
    gps = [float(r.games_per_sec) for r in rows]
    pps = [float(r.positions_per_sec) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    if iter_times:
        ax.plot(iter_times, gps, marker="o", label="games/s", color="#4C78A8")
        ax.plot(iter_times, pps, marker="x", label="positions/s", color="#F58518")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("throughput")
    ax.set_title("Stable Run Throughput")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(throughput_path)
    plt.close(fig)

    sample_times = [max(0.0, float(s[0] - t0)) for s in gpu_samples]
    util = [float(s[1]) for s in gpu_samples]
    power = [float(s[2]) for s in gpu_samples]
    gfx = [float(s[5]) for s in gpu_samples]

    fig2, ax_left = plt.subplots(figsize=(9, 4.8), dpi=120)
    if sample_times:
        ax_left.plot(sample_times, power, label="power(W)", color="#E45756")
        ax_left.plot(sample_times, util, label="util(%)", color="#54A24B")
    ax_left.set_xlabel("time (s)")
    ax_left.set_ylabel("power/util")
    ax_left.grid(alpha=0.3)

    ax_right = ax_left.twinx()
    if sample_times:
        ax_right.plot(sample_times, gfx, label="graphics_clock(MHz)", color="#72B7B2", alpha=0.9)
    ax_right.set_ylabel("graphics clock (MHz)")

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")
    ax_left.set_title("Stable Run GPU Telemetry")
    fig2.tight_layout()
    fig2.savefig(gpu_path)
    plt.close(fig2)

    return {
        "stable_throughput_plot": str(throughput_path),
        "stable_gpu_plot": str(gpu_path),
    }


def _aggregate_timing(rows: List[IterRow]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    keys = ["root_puct_ms", "pack_writeback_ms", "self_play_step_ms", "finalize_ms"]
    totals = {k: 0.0 for k in keys}
    calls = {k: 0 for k in keys}
    for row in rows:
        for key in keys:
            totals[key] += float(row.step_timing_ms.get(key, 0.0))
            calls[key] += int(row.step_timing_calls.get(key, 0))
    total_ms = float(sum(totals.values()))
    ratio = {k: (totals[k] / total_ms if total_ms > 0.0 else 0.0) for k in keys}
    return totals, ratio, calls


def _aggregate_mcts_counters(rows: List[IterRow]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        for key, value in row.mcts_counters.items():
            out[key] = int(out.get(key, 0) + int(value))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed self-play workload and export telemetry.")
    parser.add_argument("--mode", type=str, choices=["v0", "v1"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-games-per-iter", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    parser.add_argument("--max-game-plies", type=int, default=512)
    parser.add_argument("--sampler-interval-sec", type=float, default=0.2)

    parser.add_argument("--v0-workers", type=int, default=1)
    parser.add_argument("--v0-batch-leaves", type=int, default=512)
    parser.add_argument("--v0-inference-backend", type=str, default="graph", choices=["graph", "py", "ts"])
    parser.add_argument("--v0-inference-batch-size", type=int, default=512)
    parser.add_argument("--v0-inference-warmup-iters", type=int, default=5)
    parser.add_argument("--v0-opening-random-moves", type=int, default=2)
    parser.add_argument("--v0-resign-threshold", type=float, default=-0.8)
    parser.add_argument("--v0-resign-min-moves", type=int, default=36)
    parser.add_argument("--v0-resign-consecutive", type=int, default=3)

    parser.add_argument("--v1-threads", type=int, default=1)
    parser.add_argument("--v1-concurrent-games", type=int, default=8)
    parser.add_argument("--v1-child-eval-mode", type=str, default="value_only", choices=["value_only", "full"])
    parser.add_argument("--v1-sample-moves", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--v1-finalize-graph", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--v1-inference-backend", type=str, default="py", choices=["py", "graph"])
    parser.add_argument("--v1-inference-batch-size", type=int, default=512)
    parser.add_argument("--v1-inference-warmup-iters", type=int, default=5)
    parser.add_argument("--collect-step-timing", action="store_true")

    parser.add_argument("--plot-step-breakdown", action="store_true")
    parser.add_argument("--plot-stability", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join("results", f"selfplay_workload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_v0_binary_compat()
    if bool(args.collect_step_timing) and any(str(k).upper().startswith("NSYS_") for k in os.environ.keys()):
        print("[warn] collect-step-timing is disabled under nsys to avoid sync-polluted traces.")
        args.collect_step_timing = False

    if int(args.num_games_per_iter) <= 0:
        raise ValueError("num-games-per-iter must be positive")
    if int(args.iterations) <= 0:
        raise ValueError("iterations must be positive")

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=dev)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(dev)
    model.eval()

    inference_engine = None
    ts_path = None
    if args.mode == "v1":
        inference_engine, ts_path = _build_v1_inference_engine(
            model=model,
            backend=str(args.v1_inference_backend),
            device=str(args.device),
            batch_size=int(args.v1_inference_batch_size),
            warmup_iters=int(args.v1_inference_warmup_iters),
        )

    gpu_index = _resolve_gpu_index(str(args.device))
    sampler = GPUSampler(gpu_index=gpu_index, interval_sec=float(args.sampler_interval_sec))
    sampler.start()

    rows: List[IterRow] = []
    run_started = time.perf_counter()
    deadline = run_started + float(args.duration_sec) if float(args.duration_sec) > 0.0 else None

    try:
        iter_idx = 0
        while True:
            iter_seed = seed + 1009 * (iter_idx + 1)
            if args.mode == "v0":
                row = _run_v0_once(model=model, args=args, iter_seed=iter_seed)
            else:
                row = _run_v1_once(model=model, args=args, inference_engine=inference_engine, iter_seed=iter_seed)
            row.index = int(iter_idx)
            rows.append(row)
            iter_idx += 1

            if deadline is not None:
                if time.perf_counter() >= deadline:
                    break
            else:
                if iter_idx >= int(args.iterations):
                    break
    finally:
        sampler.stop()
        if inference_engine is not None:
            del inference_engine
        if ts_path and os.path.exists(ts_path):
            try:
                os.remove(ts_path)
            except OSError:
                pass

    run_elapsed = max(1e-9, time.perf_counter() - run_started)
    total_games = int(sum(r.games for r in rows))
    total_positions = int(sum(r.positions for r in rows))
    step_timing_ms, step_timing_ratio, step_timing_calls = _aggregate_timing(rows)
    mcts_counters = _aggregate_mcts_counters(rows)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    output_prefix = output_path.with_suffix("")
    if args.mode == "v1" and bool(args.collect_step_timing) and bool(args.plot_step_breakdown):
        artifacts.update(_save_step_breakdown_plots(step_timing_ms, step_timing_ratio, output_prefix))
    if bool(args.plot_stability):
        artifacts.update(_save_stability_plot(rows, sampler.samples, output_prefix))

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "mode": args.mode,
            "device": args.device,
            "seed": int(args.seed),
            "checkpoint": args.checkpoint,
            "num_games_per_iter": int(args.num_games_per_iter),
            "iterations": int(args.iterations),
            "duration_sec": float(args.duration_sec),
            "mcts_simulations": int(args.mcts_simulations),
            "max_game_plies": int(args.max_game_plies),
            "v0_workers": int(args.v0_workers),
            "v0_batch_leaves": int(args.v0_batch_leaves),
            "v0_inference_backend": str(args.v0_inference_backend),
            "v0_inference_batch_size": int(args.v0_inference_batch_size),
            "v1_threads": int(args.v1_threads),
            "v1_concurrent_games": int(args.v1_concurrent_games),
            "v1_child_eval_mode": str(args.v1_child_eval_mode),
            "v1_sample_moves": _parse_bool_flag(str(args.v1_sample_moves)),
            "v1_finalize_graph": str(args.v1_finalize_graph),
            "v1_inference_backend": str(args.v1_inference_backend),
            "v1_inference_batch_size": int(args.v1_inference_batch_size),
            "collect_step_timing": bool(args.collect_step_timing),
        },
        "summary": {
            "iterations": int(len(rows)),
            "run_elapsed_sec": float(run_elapsed),
            "games": int(total_games),
            "positions": int(total_positions),
            "games_per_sec": _safe_div(float(total_games), float(run_elapsed)),
            "positions_per_sec": _safe_div(float(total_positions), float(run_elapsed)),
            "step_timing_ms": {k: float(v) for k, v in step_timing_ms.items()},
            "step_timing_ratio": {k: float(v) for k, v in step_timing_ratio.items()},
            "step_timing_calls": {k: int(v) for k, v in step_timing_calls.items()},
            "mcts_counters": {k: int(v) for k, v in mcts_counters.items()},
        },
        "gpu_summary": sampler.summary(),
        "gpu_samples": [
            {
                "time_sec": float(s[0] - run_started),
                "gpu_util": float(s[1]),
                "gpu_power_w": float(s[2]),
                "gpu_mem_mib": float(s[3]),
                "pstate": str(s[4]),
                "graphics_clock_mhz": float(s[5]),
                "sm_clock_mhz": float(s[6]),
            }
            for s in sampler.samples
        ],
        "iterations": [asdict(row) for row in rows],
        "artifacts": artifacts,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report_saved={output_path}")
    print(
        f"mode={args.mode} iterations={len(rows)} games={total_games} elapsed={run_elapsed:.3f}s "
        f"games/s={_safe_div(float(total_games), float(run_elapsed)):.3f}"
    )


if __name__ == "__main__":
    main()
