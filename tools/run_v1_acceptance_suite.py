#!/usr/bin/env python
"""Run phase-A v1 acceptance suite and emit one consolidated report."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RESULTS = ROOT / "results"


@dataclass
class RunArtifact:
    kind: str
    backend: str
    repeat: int
    seed: int
    path: Path


def _pythonpath_with_project() -> str:
    current = os.environ.get("PYTHONPATH", "")
    entries = [str(ROOT / "build" / "v0" / "src"), str(ROOT)]
    if current:
        entries.append(current)
    sep = ";" if os.name == "nt" else ":"
    return sep.join(entries)


def _run_cmd(args: List[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_project()
    print("[run]", " ".join(args))
    subprocess.run(args, check=True, cwd=str(ROOT), env=env)


def _float_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_ab(
    *,
    device: str,
    seed: int,
    sims: int,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(TOOLS / "ab_v1_child_value_only.py"),
        "--device",
        device,
        "--seed",
        str(seed),
        "--num-states",
        "32",
        "--state-plies",
        "8",
        "--mcts-simulations",
        str(int(sims)),
        "--self-play-games",
        "8",
        "--self-play-concurrent-games",
        "8",
        "--strict",
        "--output-json",
        str(output_path),
    ]
    _run_cmd(cmd)


def _run_validate(
    *,
    device: str,
    seed: int,
    rounds: int,
    backend: str,
    total_games: int,
    mcts_simulations: int,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(TOOLS / "validate_v1_claims.py"),
        "--device",
        device,
        "--seed",
        str(seed),
        "--rounds",
        str(int(rounds)),
        "--v0-workers",
        "1,2,4",
        "--v1-threads",
        "1,2,4",
        "--v1-concurrent-games",
        "8",
        "--v1-child-eval-mode",
        "value_only",
        "--total-games",
        str(int(total_games)),
        "--v0-mcts-simulations",
        str(int(mcts_simulations)),
        "--v1-mcts-simulations",
        str(int(mcts_simulations)),
        "--v0-batch-leaves",
        "512",
        "--v0-inference-backend",
        "graph",
        "--v0-inference-batch-size",
        "512",
        "--v0-inference-warmup-iters",
        "5",
        "--v1-inference-backend",
        backend,
        "--v1-inference-batch-size",
        "512",
        "--v1-inference-warmup-iters",
        "5",
        "--v0-opening-random-moves",
        "2",
        "--v0-resign-threshold",
        "-0.8",
        "--v0-resign-min-moves",
        "36",
        "--v0-resign-consecutive",
        "3",
        "--output-json",
        str(output_path),
    ]
    _run_cmd(cmd)


def _run_matrix(
    *,
    device: str,
    seed: int,
    total_games: int,
    mcts_simulations: int,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(TOOLS / "sweep_v1_gpu_matrix.py"),
        "--device",
        device,
        "--seed",
        str(seed),
        "--rounds",
        "1",
        "--threads",
        "1",
        "--concurrent-games",
        "8,16,32,64",
        "--backends",
        "py,graph",
        "--total-games",
        str(int(total_games)),
        "--mcts-simulations",
        str(int(mcts_simulations)),
        "--child-eval-mode",
        "value_only",
        "--inference-batch-size",
        "512",
        "--inference-warmup-iters",
        "5",
        "--output-json",
        str(output_path),
    ]
    _run_cmd(cmd)


def _run_smoke() -> bool:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/v1/test_v1_tensor_pipeline_smoke.py",
        "-q",
    ]
    try:
        _run_cmd(cmd)
    except subprocess.CalledProcessError:
        return False
    return True


def _collect_validate_stats(paths: List[Path]) -> Dict[str, object]:
    runs: List[Dict[str, object]] = []
    by_scale: Dict[str, List[float]] = {}
    fixed_min_values: List[float] = []
    best_values: List[float] = []
    thread_gain_values: List[float] = []
    p0_values: List[float] = []
    power_delta_values: List[float] = []

    for path in paths:
        data = _load_json(path)
        summary = data["summary"]
        fixed_min = float(summary["speedup_fixed_worker_min"])
        best = float(summary["speedup_best_v1_vs_v0_worker1"])
        thread_gain = float(summary["v1_thread_gain"])
        p0_ratio = float(summary["v1_p0_ratio_min"])
        power_delta = float(summary["power_delta_best_v1_minus_v0_worker1_w"])
        fixed_min_values.append(fixed_min)
        best_values.append(best)
        thread_gain_values.append(thread_gain)
        p0_values.append(p0_ratio)
        power_delta_values.append(power_delta)

        v0_rows = sorted(data["v0_rows"], key=lambda r: int(r["scale"]))
        v1_rows = sorted(data["v1_rows"], key=lambda r: int(r["scale"]))
        same_scale: Dict[str, float] = {}
        for v0_row, v1_row in zip(v0_rows, v1_rows):
            scale = str(int(v0_row["scale"]))
            v0_gps = float(v0_row["games_per_sec"])
            v1_gps = float(v1_row["games_per_sec"])
            speedup = float(v1_gps / v0_gps) if v0_gps > 0 else float("nan")
            same_scale[scale] = speedup
            by_scale.setdefault(scale, []).append(speedup)

        runs.append(
            {
                "path": str(path),
                "fixed_worker_min": fixed_min,
                "best_speedup": best,
                "thread_gain": thread_gain,
                "p0_ratio_min": p0_ratio,
                "power_delta_w": power_delta,
                "same_scale_speedup": same_scale,
            }
        )

    scale_stats = {scale: _float_stats(values) for scale, values in sorted(by_scale.items())}
    return {
        "runs": runs,
        "fixed_worker_min": _float_stats(fixed_min_values),
        "best_speedup_vs_v0_worker1": _float_stats(best_values),
        "thread_gain": _float_stats(thread_gain_values),
        "p0_ratio_min": _float_stats(p0_values),
        "power_delta_w": _float_stats(power_delta_values),
        "same_scale_speedup": scale_stats,
    }


def _collect_matrix_stats(paths: List[Path]) -> Dict[str, object]:
    table: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for path in paths:
        data = _load_json(path)
        for row in data["rows"]:
            key = (str(row["backend"]), int(row["concurrent_games"]))
            slot = table.setdefault(
                key,
                {"games_per_sec": [], "positions_per_sec": [], "gpu_power_avg_w": [], "gpu_util_avg": []},
            )
            slot["games_per_sec"].append(float(row["games_per_sec"]))
            slot["positions_per_sec"].append(float(row["positions_per_sec"]))
            slot["gpu_power_avg_w"].append(float(row["gpu_power_avg_w"]))
            slot["gpu_util_avg"].append(float(row["gpu_util_avg"]))

    rows: List[Dict[str, object]] = []
    for (backend, cg), vals in sorted(table.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "backend": backend,
                "concurrent_games": cg,
                "games_per_sec": _float_stats(vals["games_per_sec"]),
                "positions_per_sec": _float_stats(vals["positions_per_sec"]),
                "gpu_power_avg_w": _float_stats(vals["gpu_power_avg_w"]),
                "gpu_util_avg": _float_stats(vals["gpu_util_avg"]),
            }
        )
    return {"rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v1 acceptance suite (Phase-A).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--total-games", type=int, default=8)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(RESULTS / f"v1_acceptance_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch_cuda_available():
        raise RuntimeError("CUDA is required for this suite.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = time.perf_counter()

    artifacts: List[RunArtifact] = []

    ab_256 = RESULTS / f"v1_accept_ab_256_{tag}.json"
    _run_ab(device=args.device, seed=int(args.seed), sims=256, output_path=ab_256)
    artifacts.append(RunArtifact(kind="ab", backend="n/a", repeat=0, seed=int(args.seed), path=ab_256))

    ab_512 = RESULTS / f"v1_accept_ab_512_{tag}.json"
    _run_ab(device=args.device, seed=int(args.seed) + 17, sims=512, output_path=ab_512)
    artifacts.append(RunArtifact(kind="ab", backend="n/a", repeat=0, seed=int(args.seed) + 17, path=ab_512))

    validate_py_paths: List[Path] = []
    validate_graph_paths: List[Path] = []
    matrix_paths: List[Path] = []

    for rep in range(max(1, int(args.repeats))):
        rep_seed = int(args.seed) + 1000003 * rep

        py_path = RESULTS / f"v1_accept_fixed_py_r{rep + 1}_{tag}.json"
        _run_validate(
            device=args.device,
            seed=rep_seed,
            rounds=max(1, int(args.rounds)),
            backend="py",
            total_games=max(1, int(args.total_games)),
            mcts_simulations=max(1, int(args.mcts_simulations)),
            output_path=py_path,
        )
        validate_py_paths.append(py_path)
        artifacts.append(RunArtifact(kind="validate", backend="py", repeat=rep + 1, seed=rep_seed, path=py_path))

        graph_path = RESULTS / f"v1_accept_fixed_graph_r{rep + 1}_{tag}.json"
        _run_validate(
            device=args.device,
            seed=rep_seed + 101,
            rounds=max(1, int(args.rounds)),
            backend="graph",
            total_games=max(1, int(args.total_games)),
            mcts_simulations=max(1, int(args.mcts_simulations)),
            output_path=graph_path,
        )
        validate_graph_paths.append(graph_path)
        artifacts.append(RunArtifact(kind="validate", backend="graph", repeat=rep + 1, seed=rep_seed + 101, path=graph_path))

        matrix_path = RESULTS / f"v1_accept_matrix_r{rep + 1}_{tag}.json"
        _run_matrix(
            device=args.device,
            seed=rep_seed + 211,
            total_games=max(1, int(args.total_games)),
            mcts_simulations=max(1, int(args.mcts_simulations)),
            output_path=matrix_path,
        )
        matrix_paths.append(matrix_path)
        artifacts.append(RunArtifact(kind="matrix", backend="py,graph", repeat=rep + 1, seed=rep_seed + 211, path=matrix_path))

    smoke_passed = _run_smoke()

    ab_reports = [_load_json(ab_256), _load_json(ab_512)]
    ab_pass = all(all(bool(v) for v in report.get("criteria", {}).values()) for report in ab_reports)

    py_stats = _collect_validate_stats(validate_py_paths)
    graph_stats = _collect_validate_stats(validate_graph_paths)
    matrix_stats = _collect_matrix_stats(matrix_paths)

    gates = {
        "ab_all_pass": bool(ab_pass),
        "smoke_passed": bool(smoke_passed),
        "graph_fixed_worker_min_mean_ge_10": bool(graph_stats["fixed_worker_min"]["mean"] >= 10.0),
        "graph_thread_gain_mean_le_0.15": bool(graph_stats["thread_gain"]["mean"] <= 0.15),
        "graph_p0_ratio_min_mean_ge_0.9": bool(graph_stats["p0_ratio_min"]["mean"] >= 0.9),
    }

    elapsed = max(1e-9, time.perf_counter() - started)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "device": str(args.device),
            "seed": int(args.seed),
            "rounds": int(args.rounds),
            "repeats": int(args.repeats),
            "total_games": int(args.total_games),
            "mcts_simulations": int(args.mcts_simulations),
            "aligned_v0_batch_leaves": 512,
            "aligned_v1_inference_batch_size": 512,
            "v0_workers": [1, 2, 4],
            "v1_threads": [1, 2, 4],
            "backends": ["py", "graph"],
            "concurrent_games_matrix": [8, 16, 32, 64],
        },
        "artifacts": [
            {
                "kind": x.kind,
                "backend": x.backend,
                "repeat": x.repeat,
                "seed": x.seed,
                "path": str(x.path),
            }
            for x in artifacts
        ],
        "phase_a": {
            "ab": {
                "reports": [str(ab_256), str(ab_512)],
                "all_pass": bool(ab_pass),
            },
            "fixed_worker_py": py_stats,
            "fixed_worker_graph": graph_stats,
            "matrix": matrix_stats,
            "smoke": {"passed": bool(smoke_passed)},
        },
        "gates": gates,
        "elapsed_sec": float(elapsed),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_saved={output_path}")
    for key, val in gates.items():
        print(f"{key}: {'PASS' if val else 'FAIL'}")


def torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    main()
