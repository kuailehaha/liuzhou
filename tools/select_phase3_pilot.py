#!/usr/bin/env python
"""Evaluate equal-wall-clock Phase-3 pilots and apply the frozen selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(row: dict[str, Any]) -> float:
    total = int(row["total_games"])
    return (
        int(row["wins"]) + 0.5 * int(row["draws"])
    ) / max(1, total)


def choose_pilot(
    *,
    direct_score_16: float,
    worst_anchor_16: float,
    worst_anchor_32: float,
    average_anchor_16: float,
    average_anchor_32: float,
    healthy_16: bool = True,
    healthy_32: bool = True,
) -> tuple[int, str]:
    if not healthy_16 and not healthy_32:
        raise RuntimeError("Both Phase-3 pilots failed health audit.")
    if healthy_16 and not healthy_32:
        return 16, "32-sim pilot failed health audit"
    if healthy_32 and not healthy_16:
        return 32, "16-sim pilot failed health audit"

    direct_score_32 = 1.0 - float(direct_score_16)
    if (
        float(direct_score_16) >= 0.55
        and float(worst_anchor_16) >= float(worst_anchor_32) - 0.02
    ):
        return 16, "16-sim won direct match without >2pp worst-anchor regression"
    if (
        direct_score_32 >= 0.55
        and float(worst_anchor_32) >= float(worst_anchor_16) - 0.02
    ):
        return 32, "32-sim won direct match without >2pp worst-anchor regression"

    worst_delta = float(worst_anchor_16) - float(worst_anchor_32)
    if abs(worst_delta) >= 0.02:
        winner = 16 if worst_delta > 0.0 else 32
        return winner, "selected by worst-anchor score"

    average_delta = float(average_anchor_16) - float(average_anchor_32)
    if abs(average_delta) >= 0.02:
        winner = 16 if average_delta > 0.0 else 32
        return winner, "selected by league average score"
    return 16, "scores within 2pp; selected higher-throughput 16-sim"


def _parse_anchor(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--anchor must use LABEL=PATH")
    label, path_raw = raw.split("=", 1)
    path = Path(path_raw).expanduser().resolve()
    if not label.strip() or not path.is_file():
        raise ValueError(f"Invalid anchor: {raw}")
    return label.strip(), path


def _run_match(
    *,
    challenger: Path,
    opponent: Path,
    games: int,
    seed: int,
    name: str,
    output: Path,
    simulations: int,
    concurrency: int,
    cpp_threads: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/eval_checkpoint.py",
        "--challenger_checkpoint",
        str(challenger),
        "--previous_checkpoint",
        str(opponent),
        "--backend",
        "portable",
        "--device",
        "mps",
        "--eval_workers",
        "1",
        "--mcts_simulations",
        str(int(simulations)),
        "--temperature",
        "1",
        "--sample_moves",
        "--v1_concurrent_games",
        str(int(concurrency)),
        "--v1_opening_random_moves",
        "0",
        "--seed",
        str(int(seed)),
        "--match_name",
        name,
        "--output_json",
        str(output),
        "--eval_games_vs_previous",
        str(int(games)),
        "--portable_mcts_backend",
        "cpp",
        "--portable_cpp_threads",
        str(int(cpp_threads)),
    ]
    output.unlink(missing_ok=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    audit = report.get("portable_mcts_audit")
    if not isinstance(audit, dict) or any(
        int(audit.get(key, -1) or 0) != 0
        for key in ("fallback_count", "illegal_action_count", "non_finite_count")
    ):
        raise RuntimeError(f"Pilot match health audit failed: {output}")
    row = next(
        dict(item)
        for item in report.get("results", [])
        if isinstance(item, dict) and item.get("name") == name
    )
    if int(row.get("total_games", 0)) != int(games):
        raise RuntimeError(f"Pilot match game count mismatch: {output}")
    color_games = [
        int(value.get("games", 0) or 0)
        for value in row.get("color_breakdown", {}).values()
        if isinstance(value, dict)
    ]
    if (
        len(color_games) != 2
        or any(count != int(games) // 2 for count in color_games)
    ):
        raise RuntimeError(f"Pilot match color split is not exact: {output}")
    row["score"] = _score(row)
    row["report"] = str(output)
    row["audit"] = audit
    return row


def _pilot_health(run_dir: Path) -> dict[str, Any]:
    positions_per_second: list[float] = []
    health_failures: list[str] = []
    events = run_dir / "events.jsonl"
    if not events.is_file():
        return {
            "healthy": False,
            "failures": ["events.jsonl missing"],
            "median_positions_per_sec": 0.0,
        }
    for line in events.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") != "iteration_complete":
            continue
        stats = row.get("selfplay_stats", {})
        train = row.get("train_metrics", {})
        if isinstance(stats, dict):
            positions_per_second.append(
                float(stats.get("positions_per_sec", 0.0) or 0.0)
            )
            counters = stats.get("mcts_counters", {})
            for key in (
                "portable_fallback_count",
                "portable_cpp_illegal_actions",
                "portable_cpp_non_finite",
            ):
                if isinstance(counters, dict) and int(counters.get(key, 0) or 0):
                    health_failures.append(f"{key}={counters.get(key)}")
        if isinstance(train, dict):
            for key in ("filtered_non_finite_samples", "device_fallback_count"):
                if int(train.get(key, 0) or 0):
                    health_failures.append(f"{key}={train.get(key)}")
            if not bool(train.get("optimizer_loaded", False)):
                health_failures.append("optimizer_loaded=false")
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        health_failures.append("state.json missing")
    return {
        "healthy": not health_failures and bool(positions_per_second),
        "failures": sorted(set(health_failures)),
        "iteration_count": len(positions_per_second),
        "median_positions_per_sec": (
            statistics.median(positions_per_second)
            if positions_per_second
            else 0.0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot16-run", required=True)
    parser.add_argument("--pilot32-run", required=True)
    parser.add_argument("--anchor", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--direct-games", type=int, default=500)
    parser.add_argument("--anchor-games", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--cpp-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026072610)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pilot_runs = {
        16: Path(args.pilot16_run).expanduser().resolve(),
        32: Path(args.pilot32_run).expanduser().resolve(),
    }
    pilot_models = {
        simulations: run_dir / "checkpoints" / "best_model.pt"
        for simulations, run_dir in pilot_runs.items()
    }
    for path in pilot_models.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    anchors = [_parse_anchor(raw) for raw in args.anchor]
    output = Path(args.output_json).expanduser().resolve()
    match_dir = output.parent / f"{output.stem}_matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    direct = _run_match(
        challenger=pilot_models[16],
        opponent=pilot_models[32],
        games=int(args.direct_games),
        seed=int(args.seed),
        name="pilot16_vs_pilot32",
        output=match_dir / "direct_16_vs_32.json",
        simulations=int(args.simulations),
        concurrency=int(args.concurrency),
        cpp_threads=int(args.cpp_threads),
    )
    anchor_results: dict[str, list[dict[str, Any]]] = {"16": [], "32": []}
    for pilot_offset, simulations in enumerate((16, 32)):
        for anchor_offset, (label, anchor) in enumerate(anchors):
            row = _run_match(
                challenger=pilot_models[simulations],
                opponent=anchor,
                games=int(args.anchor_games),
                seed=int(args.seed) + 100 + pilot_offset * 10 + anchor_offset,
                name=f"pilot{simulations}_vs_{label}",
                output=(
                    match_dir
                    / f"pilot{simulations}_vs_anchor_{anchor_offset:02d}.json"
                ),
                simulations=int(args.simulations),
                concurrency=int(args.concurrency),
                cpp_threads=int(args.cpp_threads),
            )
            anchor_results[str(simulations)].append(
                {
                    "anchor": label,
                    "anchor_path": str(anchor),
                    "anchor_sha256": _sha256(anchor),
                    "result": row,
                }
            )

    health = {
        str(simulations): _pilot_health(run_dir)
        for simulations, run_dir in pilot_runs.items()
    }
    scores = {
        key: [float(row["result"]["score"]) for row in rows]
        for key, rows in anchor_results.items()
    }
    selected, reason = choose_pilot(
        direct_score_16=float(direct["score"]),
        worst_anchor_16=min(scores["16"]),
        worst_anchor_32=min(scores["32"]),
        average_anchor_16=sum(scores["16"]) / len(scores["16"]),
        average_anchor_32=sum(scores["32"]) / len(scores["32"]),
        healthy_16=bool(health["16"]["healthy"]),
        healthy_32=bool(health["32"]["healthy"]),
    )
    result = {
        "selected_simulations": selected,
        "selection_reason": reason,
        "conditions": {
            "simulations": int(args.simulations),
            "temperature": 1.0,
            "sample_moves": True,
            "direct_games": int(args.direct_games),
            "anchor_games": int(args.anchor_games),
            "seed": int(args.seed),
        },
        "pilots": {
            str(simulations): {
                "run_dir": str(pilot_runs[simulations]),
                "checkpoint": str(pilot_models[simulations]),
                "checkpoint_sha256": _sha256(pilot_models[simulations]),
                "health": health[str(simulations)],
                "worst_anchor_score": min(scores[str(simulations)]),
                "average_anchor_score": (
                    sum(scores[str(simulations)]) / len(scores[str(simulations)])
                ),
            }
            for simulations in (16, 32)
        },
        "direct": direct,
        "anchors": anchor_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"selected_simulations={selected} report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
