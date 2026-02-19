#!/usr/bin/env python
"""A/B validator for v1 finalize-graph capture with deterministic sampling off.

This script compares:
- A: --v1-finalize-graph off
- B: --v1-finalize-graph on

Both runs force --v1-sample-moves false so graph-capturable deterministic
finalize path can be triggered and compared directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _run_case(args: argparse.Namespace, mode: str, output_json: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "tools/run_selfplay_workload.py",
        "--mode",
        "v1",
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--duration-sec",
        str(args.duration_sec),
        "--num-games-per-iter",
        str(args.num_games_per_iter),
        "--mcts-simulations",
        str(args.mcts_simulations),
        "--v1-threads",
        str(args.v1_threads),
        "--v1-concurrent-games",
        str(args.v1_concurrent_games),
        "--v1-child-eval-mode",
        str(args.v1_child_eval_mode),
        "--v1-inference-backend",
        str(args.v1_inference_backend),
        "--v1-inference-batch-size",
        str(args.v1_inference_batch_size),
        "--v1-inference-warmup-iters",
        str(args.v1_inference_warmup_iters),
        "--v1-sample-moves",
        "false",
        "--v1-finalize-graph",
        str(mode),
        "--collect-step-timing",
        "--output-json",
        str(output_json),
    ]
    subprocess.run(cmd, check=True)
    with output_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return float("nan")
    return float(a / b)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A/B test v1 finalize-graph capture (sample_moves=false).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--duration-sec", type=float, default=60.0)
    p.add_argument("--num-games-per-iter", type=int, default=64)
    p.add_argument("--mcts-simulations", type=int, default=128)
    p.add_argument("--v1-threads", type=int, default=1)
    p.add_argument("--v1-concurrent-games", type=int, default=64)
    p.add_argument("--v1-child-eval-mode", type=str, default="value_only", choices=["value_only", "full"])
    p.add_argument("--v1-inference-backend", type=str, default="py", choices=["py", "graph"])
    p.add_argument("--v1-inference-batch-size", type=int, default=512)
    p.add_argument("--v1-inference-warmup-iters", type=int, default=5)
    p.add_argument(
        "--output-json",
        type=str,
        default=str(Path("results") / f"v1_finalize_graph_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    off_json = out_path.with_name(out_path.stem + "_off.json")
    on_json = out_path.with_name(out_path.stem + "_on.json")

    result_off = _run_case(args, mode="off", output_json=off_json)
    result_on = _run_case(args, mode="on", output_json=on_json)

    off_summary = result_off.get("summary", {})
    on_summary = result_on.get("summary", {})
    off_counters = off_summary.get("mcts_counters", {}) or {}
    on_counters = on_summary.get("mcts_counters", {}) or {}

    off_gps = float(off_summary.get("games_per_sec", 0.0))
    on_gps = float(on_summary.get("games_per_sec", 0.0))
    off_pps = float(off_summary.get("positions_per_sec", 0.0))
    on_pps = float(on_summary.get("positions_per_sec", 0.0))

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "device": args.device,
            "seed": int(args.seed),
            "duration_sec": float(args.duration_sec),
            "num_games_per_iter": int(args.num_games_per_iter),
            "mcts_simulations": int(args.mcts_simulations),
            "v1_threads": int(args.v1_threads),
            "v1_concurrent_games": int(args.v1_concurrent_games),
            "v1_child_eval_mode": str(args.v1_child_eval_mode),
            "v1_inference_backend": str(args.v1_inference_backend),
            "v1_inference_batch_size": int(args.v1_inference_batch_size),
            "forced_v1_sample_moves": False,
        },
        "artifacts": {
            "off_json": str(off_json),
            "on_json": str(on_json),
        },
        "off": {
            "games_per_sec": off_gps,
            "positions_per_sec": off_pps,
            "mcts_counters": {k: int(v) for k, v in off_counters.items()},
        },
        "on": {
            "games_per_sec": on_gps,
            "positions_per_sec": on_pps,
            "mcts_counters": {k: int(v) for k, v in on_counters.items()},
        },
        "summary": {
            "games_per_sec_ratio_on_over_off": _safe_div(on_gps, off_gps),
            "positions_per_sec_ratio_on_over_off": _safe_div(on_pps, off_pps),
            "on_capture_count": int(on_counters.get("finalize_graph_capture_count", 0)),
            "on_replay_count": int(on_counters.get("finalize_graph_replay_count", 0)),
            "on_fallback_count": int(on_counters.get("finalize_graph_fallback_count", 0)),
            "off_capture_count": int(off_counters.get("finalize_graph_capture_count", 0)),
            "off_replay_count": int(off_counters.get("finalize_graph_replay_count", 0)),
            "off_fallback_count": int(off_counters.get("finalize_graph_fallback_count", 0)),
        },
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report_saved={out_path}")
    print(
        "on/off ratio: "
        f"games_per_sec={payload['summary']['games_per_sec_ratio_on_over_off']:.3f}, "
        f"positions_per_sec={payload['summary']['positions_per_sec_ratio_on_over_off']:.3f}"
    )
    print(
        "capture counters(on): "
        f"capture={payload['summary']['on_capture_count']}, "
        f"replay={payload['summary']['on_replay_count']}, "
        f"fallback={payload['summary']['on_fallback_count']}"
    )


if __name__ == "__main__":
    main()

