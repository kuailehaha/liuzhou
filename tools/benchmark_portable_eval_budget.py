#!/usr/bin/env python3
"""Run and summarize repeatable portable incumbent-gate evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_even_csv(raw: str) -> List[int]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values or any(value <= 0 or value % 2 for value in values):
        raise ValueError("--games must contain positive even integers")
    return list(dict.fromkeys(values))


def summarize_reports(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    scores: List[float] = []
    elapsed: List[float] = []
    rows: List[Dict[str, Any]] = []
    expected_games = None
    for report in reports:
        resolution = report.get("device_resolution", {})
        if int(resolution.get("fallback_count", 0) or 0) != 0:
            raise ValueError("evaluation report contains a device fallback")
        result_rows = report.get("results")
        if not isinstance(result_rows, list) or len(result_rows) != 1:
            raise ValueError("evaluation report must contain exactly one result")
        row = result_rows[0]
        games = int(row.get("total_games", 0) or 0)
        if games <= 0 or games % 2:
            raise ValueError("evaluation result must contain a positive even game count")
        if expected_games is None:
            expected_games = games
        elif games != expected_games:
            raise ValueError("cannot summarize reports with different game counts")
        colors = row.get("color_breakdown", {})
        black_games = int(colors.get("challenger_black", {}).get("games", 0) or 0)
        white_games = int(colors.get("challenger_white", {}).get("games", 0) or 0)
        if black_games != games // 2 or white_games != games // 2:
            raise ValueError("evaluation result is not evenly split by color")
        wins = int(row.get("wins", 0) or 0)
        losses = int(row.get("losses", 0) or 0)
        draws = int(row.get("draws", 0) or 0)
        if wins + losses + draws != games:
            raise ValueError("W/L/D counts do not sum to total_games")
        score = (wins + (0.5 * draws)) / games
        scores.append(score)
        elapsed.append(float(report.get("elapsed_sec", 0.0) or 0.0))
        rows.append(
            {
                "seed": int(row.get("seed")),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score": score,
                "elapsed_sec": elapsed[-1],
            }
        )
    mean_score = statistics.fmean(scores)
    between_seed_sd = statistics.stdev(scores) if len(scores) > 1 else None
    mean_ci_halfwidth = (
        1.959963984540054 * between_seed_sd / math.sqrt(len(scores))
        if between_seed_sd is not None
        else None
    )
    return {
        "games_per_repeat": expected_games,
        "repeats": len(rows),
        "score_mean": mean_score,
        "score_min": min(scores),
        "score_max": max(scores),
        "score_between_seed_sd": between_seed_sd,
        "score_mean_normal_95_halfwidth": mean_ci_halfwidth,
        "elapsed_total_sec": sum(elapsed),
        "elapsed_mean_sec": statistics.fmean(elapsed),
        "elapsed_median_sec": statistics.median(elapsed),
        "rows": rows,
    }


def _load_valid_existing(
    path: Path, *, challenger_sha: str, opponent_sha: str, games: int, seed: int
) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = payload.get("benchmark_audit", {})
    expected = {
        "challenger_sha256": challenger_sha,
        "opponent_sha256": opponent_sha,
        "games": games,
        "seed": seed,
    }
    if audit != expected:
        raise RuntimeError(f"existing report audit differs: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", default="500,1000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    games_values = _parse_positive_even_csv(args.games)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to estimate between-seed variance")
    challenger = Path(args.challenger).resolve()
    opponent = Path(args.opponent).resolve()
    if not challenger.is_file() or not opponent.is_file():
        parser.error("challenger and opponent checkpoints must exist")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    challenger_sha = _sha256(challenger)
    opponent_sha = _sha256(opponent)
    summaries: List[Dict[str, Any]] = []
    for games in games_values:
        reports: List[Dict[str, Any]] = []
        for repeat in range(args.repeats):
            seed = int(args.seed) + (games * 1000) + repeat
            report_path = output_dir / f"gate_{games:05d}_seed_{seed}.json"
            report = _load_valid_existing(
                report_path,
                challenger_sha=challenger_sha,
                opponent_sha=opponent_sha,
                games=games,
                seed=seed,
            )
            if report is None:
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
                    args.device,
                    "--eval_workers",
                    str(args.workers),
                    "--mcts_simulations",
                    str(args.simulations),
                    "--temperature",
                    "1.0",
                    "--sample_moves",
                    "--v1_concurrent_games",
                    str(args.concurrency),
                    "--v1_opening_random_moves",
                    "0",
                    "--seed",
                    str(seed),
                    "--match_name",
                    "vs_best",
                    "--eval_games_vs_previous",
                    str(games),
                    "--output_json",
                    str(report_path),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["benchmark_audit"] = {
                    "challenger_sha256": challenger_sha,
                    "opponent_sha256": opponent_sha,
                    "games": games,
                    "seed": seed,
                }
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            reports.append(report)
        summary = summarize_reports(reports)
        summary["challenger_sha256"] = challenger_sha
        summary["opponent_sha256"] = opponent_sha
        summaries.append(summary)
    output = {"schema_version": 1, "summaries": summaries}
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
