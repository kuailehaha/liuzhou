#!/usr/bin/env python
"""Evaluate a checkpoint against RandomAgent and/or a previous checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

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

from src.evaluate import (  # noqa: E402
    EvaluationStats,
    evaluate_against_agent_parallel,
    evaluate_against_agent_parallel_v0,
)


def _parse_devices(device: str, eval_devices: Optional[str]) -> List[str]:
    raw = str(eval_devices).strip() if eval_devices is not None else ""
    items = [x.strip() for x in raw.split(",") if x.strip()] if raw else [str(device).strip()]
    return items or [str(device).strip()]


def _stats_to_payload(name: str, stats: EvaluationStats) -> Dict[str, Any]:
    return {
        "name": str(name),
        "wins": int(stats.wins),
        "losses": int(stats.losses),
        "draws": int(stats.draws),
        "total_games": int(stats.total_games),
        "win_rate": float(stats.win_rate),
        "loss_rate": float(stats.loss_rate),
        "draw_rate": float(stats.draw_rate),
    }


def _print_stats_line(title: str, payload: Dict[str, Any]) -> None:
    print(
        f"[eval] {title}: "
        f"W-L-D={int(payload['wins'])}-{int(payload['losses'])}-{int(payload['draws'])} "
        f"({int(payload['total_games'])} games), "
        f"win={float(payload['win_rate']) * 100.0:.2f}% "
        f"loss={float(payload['loss_rate']) * 100.0:.2f}% "
        f"draw={float(payload['draw_rate']) * 100.0:.2f}%"
    )


def _run_eval_one(
    *,
    backend: str,
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    num_games: int,
    device: str,
    devices: List[str],
    workers: int,
    mcts_simulations: int,
    temperature: float,
    sample_moves: bool,
    batch_leaves: int,
    inference_backend: str,
    inference_batch_size: int,
    inference_warmup_iters: int,
) -> EvaluationStats:
    if backend == "v0":
        try:
            return evaluate_against_agent_parallel_v0(
                challenger_checkpoint=challenger_checkpoint,
                opponent_checkpoint=opponent_checkpoint,
                num_games=int(num_games),
                device=str(device),
                mcts_simulations=int(mcts_simulations),
                temperature=float(temperature),
                add_dirichlet_noise=False,
                num_workers=int(workers),
                devices=devices,
                verbose=False,
                game_verbose=False,
                mcts_verbose=False,
                batch_leaves=int(batch_leaves),
                inference_backend=str(inference_backend),
                torchscript_path=None,
                torchscript_dtype=None,
                inference_batch_size=int(inference_batch_size),
                inference_warmup_iters=int(inference_warmup_iters),
                sample_moves=bool(sample_moves),
            )
        except Exception as exc:
            print(
                "[eval] warning: backend=v0 failed, fallback to legacy evaluate path. "
                f"reason={exc}"
            )
    return evaluate_against_agent_parallel(
        challenger_checkpoint=challenger_checkpoint,
        opponent_checkpoint=opponent_checkpoint,
        num_games=int(num_games),
        device=str(device),
        mcts_simulations=int(mcts_simulations),
        temperature=float(temperature),
        add_dirichlet_noise=False,
        num_workers=int(workers),
        devices=devices,
        verbose=False,
        game_verbose=False,
        mcts_verbose=False,
        sample_moves=bool(sample_moves),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint against random/previous.")
    parser.add_argument("--challenger_checkpoint", type=str, required=True)
    parser.add_argument("--previous_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_devices", type=str, default=None)
    parser.add_argument("--eval_workers", type=int, default=1)
    parser.add_argument("--backend", type=str, choices=["v0", "legacy"], default="v0")
    parser.add_argument("--mcts_simulations", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--sample_moves", action="store_true")
    parser.add_argument("--eval_games_vs_random", type=int, default=0)
    parser.add_argument("--eval_games_vs_previous", type=int, default=0)
    parser.add_argument("--batch_leaves", type=int, default=256)
    parser.add_argument("--inference_backend", type=str, default="graph")
    parser.add_argument("--inference_batch_size", type=int, default=512)
    parser.add_argument("--inference_warmup_iters", type=int, default=5)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    challenger = str(args.challenger_checkpoint)
    if not os.path.exists(challenger):
        raise FileNotFoundError(f"challenger checkpoint not found: {challenger}")

    prev_path = str(args.previous_checkpoint).strip() if args.previous_checkpoint is not None else ""
    previous_checkpoint = prev_path if prev_path and os.path.exists(prev_path) else None
    if prev_path and previous_checkpoint is None:
        print(f"[eval] warning: previous checkpoint not found, skip vs_previous: {prev_path}")

    eval_devices = _parse_devices(device=str(args.device), eval_devices=args.eval_devices)
    workers = max(1, int(args.eval_workers))
    backend = str(args.backend).strip().lower()
    started = time.perf_counter()

    result_rows: List[Dict[str, Any]] = []
    games_vs_random = max(0, int(args.eval_games_vs_random))
    if games_vs_random > 0:
        stats = _run_eval_one(
            backend=backend,
            challenger_checkpoint=challenger,
            opponent_checkpoint=None,
            num_games=games_vs_random,
            device=str(args.device),
            devices=eval_devices,
            workers=workers,
            mcts_simulations=int(args.mcts_simulations),
            temperature=float(args.temperature),
            sample_moves=bool(args.sample_moves),
            batch_leaves=int(args.batch_leaves),
            inference_backend=str(args.inference_backend),
            inference_batch_size=int(args.inference_batch_size),
            inference_warmup_iters=int(args.inference_warmup_iters),
        )
        payload = _stats_to_payload("vs_random", stats)
        result_rows.append(payload)
        _print_stats_line("vs_random", payload)

    games_vs_previous = max(0, int(args.eval_games_vs_previous))
    if games_vs_previous > 0 and previous_checkpoint:
        stats = _run_eval_one(
            backend=backend,
            challenger_checkpoint=challenger,
            opponent_checkpoint=previous_checkpoint,
            num_games=games_vs_previous,
            device=str(args.device),
            devices=eval_devices,
            workers=workers,
            mcts_simulations=int(args.mcts_simulations),
            temperature=float(args.temperature),
            sample_moves=bool(args.sample_moves),
            batch_leaves=int(args.batch_leaves),
            inference_backend=str(args.inference_backend),
            inference_batch_size=int(args.inference_batch_size),
            inference_warmup_iters=int(args.inference_warmup_iters),
        )
        payload = _stats_to_payload("vs_previous", stats)
        result_rows.append(payload)
        _print_stats_line("vs_previous", payload)
    elif games_vs_previous > 0 and not previous_checkpoint:
        print("[eval] skip vs_previous: previous checkpoint unavailable.")

    elapsed = max(1e-9, time.perf_counter() - started)
    report = {
        "challenger_checkpoint": challenger,
        "previous_checkpoint": previous_checkpoint,
        "backend": backend,
        "device": str(args.device),
        "eval_devices": eval_devices,
        "eval_workers": workers,
        "mcts_simulations": int(args.mcts_simulations),
        "temperature": float(args.temperature),
        "sample_moves": bool(args.sample_moves),
        "elapsed_sec": float(elapsed),
        "results": result_rows,
    }

    output_json = str(args.output_json).strip() if args.output_json is not None else ""
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[eval] report saved: {output_json}")

    if not result_rows:
        print("[eval] no evaluation configured (all eval_games are zero).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
