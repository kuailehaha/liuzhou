#!/usr/bin/env python
"""Run v1-only GPU matrix sweeps for concurrency and backend sensitivity."""

from __future__ import annotations

import argparse
import json
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
from v1.python.self_play_gpu_runner import self_play_v1_gpu


def _parse_bool_flag(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean flag value: {value!r}")


def _parse_int_list(raw: str, default: Sequence[int]) -> List[int]:
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
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _parse_backend_list(raw: str) -> List[str]:
    values = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not values:
        return ["py"]
    out = []
    seen = set()
    for value in values:
        if value not in ("py", "graph"):
            raise ValueError(f"Unsupported backend: {value}")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _ensure_v0_binary_compat() -> None:
    if hasattr(v0_core.MCTSConfig, "max_actions_per_batch"):
        return
    v0_core.MCTSConfig.max_actions_per_batch = property(  # type: ignore[attr-defined]
        lambda self: 0,
        lambda self, value: None,
    )


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


def _check_background_load(gpu_index: int, allow_external_gpu_procs: int) -> None:
    rows = _query_gpu_processes(gpu_index)
    current = int(os.getpid())
    external = [row for row in rows if row[0] != current]
    if len(external) > int(allow_external_gpu_procs):
        detail = ", ".join(f"{pid}:{name}:{mem:.0f}MiB" for pid, name, mem in external)
        raise RuntimeError(
            "GPU seems busy before benchmark. "
            f"external_processes={len(external)}, allowed={allow_external_gpu_procs}, detail=[{detail}]"
        )


class GPUSampler:
    def __init__(self, gpu_index: int, interval_sec: float) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_sec = max(0.05, float(interval_sec))
        self.samples: List[Tuple[float, float, float, str, float, float]] = []
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
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 6:
                        self.samples.append(
                            (
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                str(parts[3]).upper(),
                                float(parts[4]),
                                float(parts[5]),
                            )
                        )
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self.samples:
            nan = float("nan")
            return {
                "gpu_util_avg": nan,
                "gpu_power_avg_w": nan,
                "gpu_power_max_w": nan,
                "gpu_mem_avg_mib": nan,
                "gpu_mem_max_mib": nan,
                "gpu_p0_ratio": nan,
                "gpu_graphics_clock_avg_mhz": nan,
                "gpu_sm_clock_avg_mhz": nan,
            }
        util = [x[0] for x in self.samples]
        power = [x[1] for x in self.samples]
        mem = [x[2] for x in self.samples]
        pstate = [x[3] for x in self.samples]
        graphics = [x[4] for x in self.samples]
        sm = [x[5] for x in self.samples]
        return {
            "gpu_util_avg": float(statistics.fmean(util)),
            "gpu_power_avg_w": float(statistics.fmean(power)),
            "gpu_power_max_w": float(max(power)),
            "gpu_mem_avg_mib": float(statistics.fmean(mem)),
            "gpu_mem_max_mib": float(max(mem)),
            "gpu_p0_ratio": float(sum(1 for x in pstate if x == "P0") / max(1, len(pstate))),
            "gpu_graphics_clock_avg_mhz": float(statistics.fmean(graphics)),
            "gpu_sm_clock_avg_mhz": float(statistics.fmean(sm)),
        }


def _resolve_gpu_index(device: str) -> int:
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0
    return 0 if dev.index is None else int(dev.index)


def _load_model(checkpoint: Optional[str], device: torch.device) -> ChessNet:
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    return model


def _export_torchscript(model: ChessNet, device: torch.device, output_path: str, batch_size: int) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    example = torch.randn(
        int(batch_size),
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced.save(output_path)
    return output_path


def _build_inference_engine(
    *,
    model: ChessNet,
    backend: str,
    device: str,
    batch_size: int,
    warmup_iters: int,
) -> Tuple[Optional[object], Optional[str]]:
    backend = str(backend).strip().lower()
    if backend == "py":
        return None, None
    if backend != "graph":
        raise ValueError(f"Unsupported backend={backend}")
    ts_path = os.path.join("results", f"v1_sweep_temp_model_{backend}.ts")
    _export_torchscript(model=model, device=torch.device(device), output_path=ts_path, batch_size=batch_size)
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


@dataclass
class SweepRow:
    backend: str
    concurrent_games: int
    threads: int
    rounds: int
    games: int
    positions: int
    elapsed_sec: float
    games_per_sec: float
    positions_per_sec: float
    gpu_util_avg: float
    gpu_power_avg_w: float
    gpu_power_max_w: float
    gpu_mem_avg_mib: float
    gpu_mem_max_mib: float
    gpu_p0_ratio: float
    gpu_graphics_clock_avg_mhz: float
    gpu_sm_clock_avg_mhz: float
    finalize_graph_capture_count: int
    finalize_graph_replay_count: int
    finalize_graph_fallback_count: int


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(float(v) for v in values))


def _run_case(
    *,
    model: ChessNet,
    device: str,
    backend: str,
    inference_engine,
    threads: int,
    concurrent_games: int,
    rounds: int,
    base_seed: int,
    total_games: int,
    mcts_simulations: int,
    child_eval_mode: str,
    sample_moves: bool,
    finalize_graph_mode: str,
    sampler_interval: float,
    gpu_index: int,
) -> SweepRow:
    games = []
    positions = []
    elapsed = []
    games_per_sec = []
    pos_per_sec = []
    gpu_util = []
    gpu_power = []
    gpu_power_max = []
    gpu_mem = []
    gpu_mem_max = []
    p0_ratio = []
    gfx_clk = []
    sm_clk = []
    capture_counts = []
    replay_counts = []
    fallback_counts = []
    prev_finalize_graph = os.environ.get("V1_FINALIZE_GRAPH")
    graph_mode = str(finalize_graph_mode).strip().lower()
    if graph_mode == "auto":
        if "V1_FINALIZE_GRAPH" in os.environ:
            del os.environ["V1_FINALIZE_GRAPH"]
    elif graph_mode == "on":
        os.environ["V1_FINALIZE_GRAPH"] = "1"
    elif graph_mode == "off":
        os.environ["V1_FINALIZE_GRAPH"] = "0"
    else:
        raise ValueError(f"Unsupported finalize_graph_mode: {finalize_graph_mode}")
    try:
        for ridx in range(int(rounds)):
            seed = int(base_seed) + 500003 * (ridx + 1) + 97 * int(concurrent_games)
            torch.manual_seed(seed)
            torch.set_num_threads(int(threads))
            sampler = GPUSampler(gpu_index=gpu_index, interval_sec=sampler_interval)
            sampler.start()
            t0 = time.perf_counter()
            _batch, stats = self_play_v1_gpu(
                model=model,
                num_games=int(total_games),
                mcts_simulations=int(mcts_simulations),
                temperature_init=1.0,
                temperature_final=0.2,
                temperature_threshold=8,
                exploration_weight=1.0,
                device=str(device),
                add_dirichlet_noise=True,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                soft_value_k=2.0,
                max_game_plies=512,
                sample_moves=bool(sample_moves),
                concurrent_games=int(concurrent_games),
                child_eval_mode=str(child_eval_mode),
                inference_engine=inference_engine,
                verbose=False,
            )
            elapsed_sec = time.perf_counter() - t0
            sampled = sampler.stop()

            games.append(float(stats.num_games))
            positions.append(float(stats.num_positions))
            elapsed.append(float(elapsed_sec))
            games_per_sec.append(float(stats.games_per_sec))
            pos_per_sec.append(float(stats.positions_per_sec))
            gpu_util.append(float(sampled["gpu_util_avg"]))
            gpu_power.append(float(sampled["gpu_power_avg_w"]))
            gpu_power_max.append(float(sampled["gpu_power_max_w"]))
            gpu_mem.append(float(sampled["gpu_mem_avg_mib"]))
            gpu_mem_max.append(float(sampled["gpu_mem_max_mib"]))
            p0_ratio.append(float(sampled["gpu_p0_ratio"]))
            gfx_clk.append(float(sampled["gpu_graphics_clock_avg_mhz"]))
            sm_clk.append(float(sampled["gpu_sm_clock_avg_mhz"]))
            capture_counts.append(int(stats.mcts_counters.get("finalize_graph_capture_count", 0)))
            replay_counts.append(int(stats.mcts_counters.get("finalize_graph_replay_count", 0)))
            fallback_counts.append(int(stats.mcts_counters.get("finalize_graph_fallback_count", 0)))
    finally:
        if prev_finalize_graph is None:
            os.environ.pop("V1_FINALIZE_GRAPH", None)
        else:
            os.environ["V1_FINALIZE_GRAPH"] = prev_finalize_graph

    return SweepRow(
        backend=str(backend),
        concurrent_games=int(concurrent_games),
        threads=int(threads),
        rounds=int(rounds),
        games=int(sum(games)),
        positions=int(sum(positions)),
        elapsed_sec=float(sum(elapsed)),
        games_per_sec=_mean(games_per_sec),
        positions_per_sec=_mean(pos_per_sec),
        gpu_util_avg=_mean(gpu_util),
        gpu_power_avg_w=_mean(gpu_power),
        gpu_power_max_w=_mean(gpu_power_max),
        gpu_mem_avg_mib=_mean(gpu_mem),
        gpu_mem_max_mib=_mean(gpu_mem_max),
        gpu_p0_ratio=_mean(p0_ratio),
        gpu_graphics_clock_avg_mhz=_mean(gfx_clk),
        gpu_sm_clock_avg_mhz=_mean(sm_clk),
        finalize_graph_capture_count=int(sum(capture_counts)),
        finalize_graph_replay_count=int(sum(replay_counts)),
        finalize_graph_fallback_count=int(sum(fallback_counts)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v1 GPU matrix sweep (backend x concurrent_games).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--concurrent-games", type=str, default="8,16,32")
    parser.add_argument("--backends", type=str, default="py,graph")
    parser.add_argument("--total-games", type=int, default=8)
    parser.add_argument("--mcts-simulations", type=int, default=256)
    parser.add_argument("--child-eval-mode", type=str, default="value_only", choices=["value_only", "full"])
    parser.add_argument("--sample-moves", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--finalize-graph", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--inference-warmup-iters", type=int, default=5)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)
    parser.add_argument("--allow-external-gpu-procs", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join("results", f"v1_gpu_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    _ensure_v0_binary_compat()

    device = torch.device(args.device)
    gpu_index = _resolve_gpu_index(args.device)
    _check_background_load(gpu_index=gpu_index, allow_external_gpu_procs=int(args.allow_external_gpu_procs))
    model = _load_model(checkpoint=args.checkpoint, device=device)
    backends = _parse_backend_list(args.backends)
    conc_values = _parse_int_list(args.concurrent_games, [8, 16, 32])
    rows: List[SweepRow] = []
    backend_paths: Dict[str, Optional[str]] = {}

    print(
        f"device={args.device} backends={backends} concurrent_games={conc_values} "
        f"threads={int(args.threads)} rounds={int(args.rounds)} mcts_sims={int(args.mcts_simulations)} "
        f"child_eval_mode={str(args.child_eval_mode)} sample_moves={str(args.sample_moves)} "
        f"finalize_graph={str(args.finalize_graph)}"
    )

    for backend in backends:
        engine, ts_path = _build_inference_engine(
            model=model,
            backend=backend,
            device=args.device,
            batch_size=int(args.inference_batch_size),
            warmup_iters=int(args.inference_warmup_iters),
        )
        backend_paths[backend] = ts_path
        for cg in conc_values:
            print(f"[run] backend={backend} concurrent_games={cg}")
            row = _run_case(
                model=model,
                device=args.device,
                backend=backend,
                inference_engine=engine,
                threads=int(args.threads),
                concurrent_games=int(cg),
                rounds=int(max(1, args.rounds)),
                base_seed=int(args.seed),
                total_games=int(max(1, args.total_games)),
                mcts_simulations=int(max(1, args.mcts_simulations)),
                child_eval_mode=str(args.child_eval_mode),
                sample_moves=_parse_bool_flag(str(args.sample_moves)),
                finalize_graph_mode=str(args.finalize_graph),
                sampler_interval=float(args.gpu_sample_interval),
                gpu_index=gpu_index,
            )
            rows.append(row)

    print("\n[v1_matrix]")
    for row in rows:
        print(
            f"backend={row.backend} cg={row.concurrent_games} "
            f"games/s={row.games_per_sec:.3f} pos/s={row.positions_per_sec:.1f} "
            f"gpu_util={row.gpu_util_avg:.1f}% power={row.gpu_power_avg_w:.1f}W "
            f"mem={row.gpu_mem_avg_mib:.0f}MiB p0={100.0 * row.gpu_p0_ratio:.1f}% "
            f"gfx_clk={row.gpu_graphics_clock_avg_mhz:.0f}MHz sm_clk={row.gpu_sm_clock_avg_mhz:.0f}MHz "
            f"capture/replay/fallback={row.finalize_graph_capture_count}/"
            f"{row.finalize_graph_replay_count}/{row.finalize_graph_fallback_count}"
        )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(args.device),
        "config": {
            "seed": int(args.seed),
            "rounds": int(args.rounds),
            "threads": int(args.threads),
            "concurrent_games": conc_values,
            "backends": backends,
            "total_games": int(args.total_games),
            "mcts_simulations": int(args.mcts_simulations),
            "child_eval_mode": str(args.child_eval_mode),
            "sample_moves": _parse_bool_flag(str(args.sample_moves)),
            "finalize_graph": str(args.finalize_graph),
            "inference_batch_size": int(args.inference_batch_size),
            "inference_warmup_iters": int(args.inference_warmup_iters),
            "torchscript_paths": backend_paths,
        },
        "rows": [asdict(row) for row in rows],
    }

    out_path = str(args.output_json)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nreport_saved={out_path}")


if __name__ == "__main__":
    main()
