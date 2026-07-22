#!/usr/bin/env python
"""Run the portable V1 self-play -> train -> eval -> reload smoke workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_checkpoint import evaluate_against_agent_parallel_v1  # noqa: E402
from src.game_state import GameState  # noqa: E402
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS  # noqa: E402
from v1.python.portable_device import resolve_portable_device  # noqa: E402
from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig  # noqa: E402
from v1.train import train_pipeline_v1  # noqa: E402


def _command_text(argv: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _model_name() -> Optional[str]:
    text = _command_text(["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"])
    if not text:
        return None
    for line in text.splitlines():
        if "Model Name:" in line:
            return line.split(":", 1)[1].strip()
    return None


def _load_model(checkpoint_path: Path, device: torch.device) -> ChessNet:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = ChessNet(
        board_size=int(payload.get("board_size", GameState.BOARD_SIZE)),
        num_input_channels=int(payload.get("num_input_channels", NUM_INPUT_CHANNELS)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--work_dir", default="tmp/v1_portable_smoke")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--self_play_games", type=int, default=2)
    parser.add_argument("--mcts_simulations", type=int, default=4)
    parser.add_argument("--max_game_plies", type=int, default=12)
    parser.add_argument("--eval_games", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolution = resolve_portable_device(args.device)
    resolved_device = resolution.device
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = work_dir / "training_metrics_v1.json"
    self_play_stats_path = work_dir / "selfplay_stats.json"
    batch_path = work_dir / "selfplay_batch_v1.pt"
    checkpoint_path = work_dir / "model_iter_001.pt"
    output_path = Path(args.output_json) if args.output_json else work_dir / "smoke_report.json"

    started = time.perf_counter()
    common_pipeline_args = {
        "iterations": 1,
        "self_play_games": max(1, int(args.self_play_games)),
        "mcts_simulations": max(1, int(args.mcts_simulations)),
        "batch_size": 8,
        "epochs": 1,
        "lr": 1e-3,
        "temperature_init": 1.0,
        "temperature_final": 0.1,
        "temperature_threshold": 4,
        "self_play_concurrent_games": max(1, int(args.self_play_games)),
        "search_backend": "portable",
        "checkpoint_dir": str(work_dir),
        "device": str(args.device),
        "devices": str(args.device),
        "train_devices": str(args.device),
        "train_strategy": "none",
        "max_game_plies": max(1, int(args.max_game_plies)),
        "model_init_seed": int(args.seed),
    }
    train_pipeline_v1(
        **common_pipeline_args,
        stage="selfplay",
        self_play_output=str(batch_path),
        self_play_stats_json=str(self_play_stats_path),
    )
    train_pipeline_v1(
        **common_pipeline_args,
        stage="train",
        self_play_input=str(batch_path),
        checkpoint_name=checkpoint_path.name,
        metrics_output=str(metrics_path),
    )
    if not checkpoint_path.exists():
        raise RuntimeError(f"Portable smoke checkpoint was not written: {checkpoint_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    self_play_stats = json.loads(self_play_stats_path.read_text(encoding="utf-8"))
    metric_rows = metrics if isinstance(metrics, list) else [metrics]
    pipeline_fallback_count = sum(
        int(row.get("device_fallback_count", 0))
        for row in metric_rows
        if isinstance(row, dict)
    ) + int(self_play_stats.get("fallback_count", 0))
    pipeline_fallback_reasons = [
        str(reason)
        for row in metric_rows
        if isinstance(row, dict)
        for key in ("device_fallback_reasons",)
        for reason in row.get(key, [])
    ] + [str(reason) for reason in self_play_stats.get("fallback_reasons", [])]
    train_elapsed = time.perf_counter() - started

    eval_started = time.perf_counter()
    eval_stats = evaluate_against_agent_parallel_v1(
        challenger_checkpoint=str(checkpoint_path),
        opponent_checkpoint=None,
        num_games=max(2, int(args.eval_games)),
        device=str(resolved_device),
        mcts_simulations=max(1, int(args.mcts_simulations)),
        temperature=0.1,
        num_workers=1,
        devices=[str(resolved_device)],
        concurrent_games=max(1, int(args.eval_games)),
        opening_random_moves=0,
        sample_moves=False,
        search_backend="portable",
    )
    eval_elapsed = time.perf_counter() - eval_started

    cpu_model = _load_model(checkpoint_path, torch.device("cpu"))
    device_model = _load_model(checkpoint_path, resolved_device)
    config = PortableMCTSConfig(num_simulations=1, add_dirichlet_noise=False)
    states = [GameState()]
    cpu_eval = PortableMCTS(cpu_model, config, "cpu").evaluate_states(states)
    device_eval = PortableMCTS(device_model, config, resolved_device).evaluate_states(states)
    legal_equal = bool(torch.equal(cpu_eval.legal_masks, device_eval.legal_masks))
    policy_max_abs = float(torch.max(torch.abs(cpu_eval.priors - device_eval.priors)).item())
    value_max_abs = float(torch.max(torch.abs(cpu_eval.values - device_eval.values)).item())

    report: Dict[str, Any] = {
        "environment": {
            "model_name": _model_name(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "device_resolution": {
            "requested": str(args.device),
            "resolved": str(resolved_device),
            "fallback_count": int(resolution.fallback_count),
            "fallback_reasons": list(resolution.fallback_reasons),
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "conditions": {
            "seed": int(args.seed),
            "self_play_games": max(1, int(args.self_play_games)),
            "mcts_simulations": max(1, int(args.mcts_simulations)),
            "max_game_plies": max(1, int(args.max_game_plies)),
            "eval_games": max(2, int(args.eval_games)),
        },
        "artifacts": {
            "self_play_batch": str(batch_path),
            "self_play_stats": str(self_play_stats_path),
            "checkpoint": str(checkpoint_path),
            "training_metrics": str(metrics_path),
        },
        "self_play": self_play_stats,
        "training": metrics,
        "pipeline_fallback": {
            "count": int(pipeline_fallback_count),
            "reasons": pipeline_fallback_reasons,
        },
        "evaluation_vs_random_health_probe": {
            "wins": int(eval_stats.wins),
            "losses": int(eval_stats.losses),
            "draws": int(eval_stats.draws),
            "total_games": int(eval_stats.total_games),
            "elapsed_sec": float(eval_elapsed),
        },
        "checkpoint_reload": {
            "cpu_and_device_load_succeeded": True,
            "cpu_device_legal_mask_equal": legal_equal,
            "policy_max_abs_difference": policy_max_abs,
            "value_max_abs_difference": value_max_abs,
            "cpu_outputs_finite": bool(
                torch.isfinite(cpu_eval.priors).all().item()
                and torch.isfinite(cpu_eval.values).all().item()
            ),
            "device_outputs_finite": bool(
                torch.isfinite(device_eval.priors).all().item()
                and torch.isfinite(device_eval.values).all().item()
            ),
        },
        "elapsed_sec": float(time.perf_counter() - started),
        "train_pipeline_elapsed_sec": float(train_elapsed),
    }
    report["all_numeric_metrics_finite"] = _all_finite(report)
    report["passed"] = bool(
        report["all_numeric_metrics_finite"]
        and legal_equal
        and report["checkpoint_reload"]["cpu_outputs_finite"]
        and report["checkpoint_reload"]["device_outputs_finite"]
        and int(resolution.fallback_count) == 0
        and int(pipeline_fallback_count) == 0
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[smoke_v1_portable] report saved: {output_path}")
    if not report["passed"]:
        raise RuntimeError("Portable V1 smoke checks did not all pass; inspect the JSON report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
