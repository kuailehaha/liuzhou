#!/usr/bin/env python3
"""Reproducible Python-vs-threaded-C++ portable self-play benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.portable_cpp_self_play import self_play_v1_portable_cpp
from v1.python.portable_self_play import self_play_v1_portable


TRAINING_PATTERNS = re.compile(
    r"(long_train_portable_mps|big_train_v1|train_entry\.py|eval_checkpoint\.py)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint: Optional[Path]) -> ChessNet:
    model = ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    )
    if checkpoint is not None:
        payload = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(payload, dict) and isinstance(
            payload.get("model_state_dict"), dict
        ):
            state = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state = payload
        else:
            raise RuntimeError(
                f"unsupported checkpoint payload: {type(payload)!r}"
            )
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _competing_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[str] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid() or "benchmark_portable_cpp.py" in command:
            continue
        if TRAINING_PATTERNS.search(command):
            rows.append(line)
    return rows


def _read_gpu_utilization() -> tuple[Optional[float], Optional[int]]:
    if platform.system() != "Darwin":
        return None, None
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', result.stdout)
    pid_match = re.search(r'"fLastSubmissionPID"\s*=\s*(\d+)', result.stdout)
    return (
        float(match.group(1)) if match else None,
        int(pid_match.group(1)) if pid_match else None,
    )


class _GpuSampler:
    def __init__(self, interval_sec: float = 0.25) -> None:
        self.interval_sec = max(0.1, float(interval_sec))
        self.values: list[float] = []
        self.last_submission_pids: list[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_GpuSampler":
        def sample() -> None:
            while not self._stop.is_set():
                value, last_pid = _read_gpu_utilization()
                if value is not None:
                    self.values.append(float(value))
                if last_pid is not None:
                    self.last_submission_pids.append(int(last_pid))
                self._stop.wait(self.interval_sec)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def _fingerprint(samples: Any) -> str:
    digest = hashlib.sha256()
    for tensor in (
        samples.state_tensors,
        samples.legal_masks,
        samples.policy_targets,
        samples.value_targets,
        samples.soft_value_targets,
    ):
        cpu = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _run_case(
    *,
    runner: Callable[..., Any],
    model: ChessNet,
    backend: str,
    threads: int,
    args: argparse.Namespace,
    seed: int,
    max_plies_override: Optional[int] = None,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    kwargs = {
        "model": model,
        "num_games": int(args.games),
        "mcts_simulations": int(args.simulations),
        "temperature_init": float(args.temperature),
        "temperature_final": float(args.temperature),
        "temperature_threshold": int(
            (max_plies_override or args.max_plies) + 1
        ),
        "exploration_weight": float(args.exploration_weight),
        "device": str(args.device),
        "add_dirichlet_noise": False,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "soft_value_k": 2.0,
        "opening_random_moves": 0,
        "max_game_plies": int(max_plies_override or args.max_plies),
        "sample_moves": False,
        "concurrent_games": int(args.concurrency),
        "verbose": False,
    }
    if backend == "cpp":
        kwargs["cpu_threads"] = int(threads)
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    wall_started = time.perf_counter()
    with _GpuSampler() as gpu:
        samples, stats = runner(**kwargs)
    wall_elapsed = max(1e-9, time.perf_counter() - wall_started)
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (
        usage_after.ru_utime
        + usage_after.ru_stime
        - usage_before.ru_utime
        - usage_before.ru_stime
    )
    policy_non_finite = int(
        (~torch.isfinite(samples.policy_targets)).sum().item()
    )
    value_non_finite = int(
        (~torch.isfinite(samples.value_targets)).sum().item()
    )
    return {
        "backend": backend,
        "threads": int(threads),
        "seed": int(seed),
        "games": int(stats.num_games),
        "positions": int(stats.num_positions),
        "elapsed_sec": float(stats.elapsed_sec),
        "wall_elapsed_sec": float(wall_elapsed),
        "positions_per_sec": float(stats.positions_per_sec),
        "games_per_sec": float(stats.games_per_sec),
        "cpu_seconds": float(cpu_seconds),
        "process_cpu_percent": float(100.0 * cpu_seconds / wall_elapsed),
        "gpu_device_utilization_median_percent": (
            float(statistics.median(gpu.values)) if gpu.values else None
        ),
        "gpu_device_utilization_max_percent": (
            float(max(gpu.values)) if gpu.values else None
        ),
        "gpu_samples": int(len(gpu.values)),
        "gpu_last_submission_pid_match_ratio": (
            float(
                sum(pid == os.getpid() for pid in gpu.last_submission_pids)
                / len(gpu.last_submission_pids)
            )
            if gpu.last_submission_pids
            else None
        ),
        "device": str(stats.device),
        "fallback_count": int(stats.fallback_count),
        "fallback_reasons": list(stats.fallback_reasons),
        "illegal_actions": int(
            stats.mcts_counters.get("portable_cpp_illegal_actions", 0)
        ),
        "non_finite_count": int(
            stats.mcts_counters.get("portable_cpp_non_finite", 0)
            + policy_non_finite
            + value_non_finite
        ),
        "inference_batches": int(stats.inference_batches),
        "device_inference_ms": float(
            stats.step_timing_ms.get("device_inference", 0.0)
        ),
        "fingerprint": _fingerprint(samples),
    }


def _median_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    return {
        "runs": int(len(values)),
        "positions_per_sec_median": float(
            statistics.median(row["positions_per_sec"] for row in values)
        ),
        "elapsed_sec_median": float(
            statistics.median(row["elapsed_sec"] for row in values)
        ),
        "process_cpu_percent_median": float(
            statistics.median(row["process_cpu_percent"] for row in values)
        ),
        "gpu_device_utilization_median_percent": (
            float(
                statistics.median(
                    row["gpu_device_utilization_median_percent"]
                    for row in values
                    if row["gpu_device_utilization_median_percent"] is not None
                )
            )
            if any(
                row["gpu_device_utilization_median_percent"] is not None
                for row in values
            )
            else None
        ),
        "gpu_device_utilization_max_percent_median": (
            float(
                statistics.median(
                    row["gpu_device_utilization_max_percent"]
                    for row in values
                    if row["gpu_device_utilization_max_percent"] is not None
                )
            )
            if any(
                row["gpu_device_utilization_max_percent"] is not None
                for row in values
            )
            else None
        ),
        "gpu_last_submission_pid_match_ratio_median": (
            float(
                statistics.median(
                    row["gpu_last_submission_pid_match_ratio"]
                    for row in values
                    if row["gpu_last_submission_pid_match_ratio"] is not None
                )
            )
            if any(
                row["gpu_last_submission_pid_match_ratio"] is not None
                for row in values
            )
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--allow-competing-load", action="store_true")
    parser.add_argument("--no-enforce-speedup", action="store_true")
    args = parser.parse_args()
    thread_values = [int(value) for value in str(args.threads).split(",")]
    if (
        int(args.games) <= 0
        or int(args.concurrency) <= 0
        or int(args.simulations) <= 0
        or int(args.max_plies) <= 0
        or int(args.repeats) <= 0
        or any(value <= 0 for value in thread_values)
    ):
        raise ValueError("games/concurrency/simulations/max-plies/repeats/threads must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    competing = _competing_processes()
    if competing and not bool(args.allow_competing_load):
        raise RuntimeError(
            "controlled benchmark refused because training/evaluation processes are active:\n"
            + "\n".join(competing)
        )
    if str(args.device) == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS benchmark requested but MPS is unavailable")

    model = _load_model(args.checkpoint)
    rows: list[Dict[str, Any]] = []
    cases = [("python", 1), *(("cpp", value) for value in thread_values)]
    warmup_max_plies = min(2, int(args.max_plies))
    print(
        json.dumps(
            {
                "event": "warmup_start",
                "cases": cases,
                "games": int(args.games),
                "concurrency": int(args.concurrency),
                "simulations": int(args.simulations),
                "max_plies": int(warmup_max_plies),
                "seed": int(args.seed),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for backend, threads in cases:
        runner = (
            self_play_v1_portable
            if backend == "python"
            else self_play_v1_portable_cpp
        )
        _run_case(
            runner=runner,
            model=model,
            backend=backend,
            threads=threads,
            args=args,
            seed=int(args.seed),
            max_plies_override=warmup_max_plies,
        )
    print(json.dumps({"event": "warmup_complete"}, sort_keys=True), flush=True)

    sequence = 0
    for repeat in range(int(args.repeats)):
        offset = repeat % len(cases)
        repeat_cases = cases[offset:] + cases[:offset]
        for backend, threads in repeat_cases:
            runner = (
                self_play_v1_portable
                if backend == "python"
                else self_play_v1_portable_cpp
            )
            row = _run_case(
                runner=runner,
                model=model,
                backend=backend,
                threads=threads,
                args=args,
                seed=int(args.seed),
            )
            row["repeat"] = int(repeat)
            row["sequence"] = int(sequence)
            sequence += 1
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    baseline_rows = [row for row in rows if row["backend"] == "python"]
    baseline = _median_rows(baseline_rows)
    summaries: Dict[str, Dict[str, Any]] = {"python": baseline}
    baseline_fingerprints = {
        int(row["repeat"]): str(row["fingerprint"]) for row in baseline_rows
    }
    semantic_match = True
    for threads in thread_values:
        selected = [
            row
            for row in rows
            if row["backend"] == "cpp" and int(row["threads"]) == threads
        ]
        summary = _median_rows(selected)
        summary["speedup_vs_python"] = float(
            summary["positions_per_sec_median"]
            / baseline["positions_per_sec_median"]
        )
        summary["semantic_fingerprint_match"] = all(
            str(row["fingerprint"]) == baseline_fingerprints[int(row["repeat"])]
            for row in selected
        )
        semantic_match = semantic_match and bool(
            summary["semantic_fingerprint_match"]
        )
        summaries[f"cpp_threads_{threads}"] = summary
    best_key = max(
        (key for key in summaries if key.startswith("cpp_threads_")),
        key=lambda key: summaries[key]["positions_per_sec_median"],
    )
    all_audits_clean = all(
        int(row["fallback_count"]) == 0
        and int(row["illegal_actions"]) == 0
        and int(row["non_finite_count"]) == 0
        for row in rows
    )
    report = {
        "schema_version": 1,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": _sha256(args.checkpoint) if args.checkpoint else None,
        "conditions": {
            "device": str(args.device),
            "games": int(args.games),
            "concurrency": int(args.concurrency),
            "simulations": int(args.simulations),
            "max_plies": int(args.max_plies),
            "temperature": float(args.temperature),
            "exploration_weight": float(args.exploration_weight),
            "seed": int(args.seed),
            "repeats": int(args.repeats),
            "dirichlet_noise": False,
            "sample_moves": False,
            "warmup": {
                "cases": [f"{backend}:{threads}" for backend, threads in cases],
                "games": int(args.games),
                "concurrency": int(args.concurrency),
                "simulations": int(args.simulations),
                "max_plies": int(warmup_max_plies),
                "seed": int(args.seed),
            },
            "case_order_rotated_per_repeat": True,
            "load_average_before": list(os.getloadavg()),
            "competing_processes": competing,
        },
        "rows": rows,
        "summaries": summaries,
        "best_cpp_configuration": best_key,
        "best_speedup": float(summaries[best_key]["speedup_vs_python"]),
        "semantic_fingerprint_match": bool(semantic_match),
        "audit_clean": bool(all_audits_clean),
        "target_speedup": float(args.min_speedup),
        "target_met": bool(
            semantic_match
            and all_audits_clean
            and float(summaries[best_key]["speedup_vs_python"])
            >= float(args.min_speedup)
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not bool(args.no_enforce_speedup) and not bool(report["target_met"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
