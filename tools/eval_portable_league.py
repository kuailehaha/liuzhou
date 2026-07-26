#!/usr/bin/env python
"""Run an audited portable checkpoint league and summarize robustness/cycles."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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


def _wilson(wins: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def analyze_league_results(
    labels: list[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    score_matrix = {
        label: {other: (0.5 if label == other else None) for other in labels}
        for label in labels
    }
    games_matrix = {
        label: {other: (0 if label == other else None) for other in labels}
        for label in labels
    }
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for match in matches:
        left = str(match["challenger"])
        right = str(match["opponent"])
        row = dict(match["result"])
        score = _score(row)
        score_matrix[left][right] = score
        score_matrix[right][left] = 1.0 - score
        games_matrix[left][right] = int(row["total_games"])
        games_matrix[right][left] = int(row["total_games"])
        pair_lookup[(left, right)] = row

    strengths = {label: 1.0 for label in labels}
    for _ in range(200):
        updated: dict[str, float] = {}
        for label in labels:
            points = 0.0
            denominator = 0.0
            for other in labels:
                if other == label or score_matrix[label][other] is None:
                    continue
                games = int(games_matrix[label][other] or 0)
                points += float(score_matrix[label][other]) * games
                denominator += games / max(
                    1e-12, strengths[label] + strengths[other]
                )
            updated[label] = (
                max(1e-12, points / denominator)
                if denominator > 0.0
                else strengths[label]
            )
        geometric_mean = math.exp(
            sum(math.log(max(1e-12, value)) for value in updated.values())
            / max(1, len(updated))
        )
        strengths = {
            label: max(1e-12, value / geometric_mean)
            for label, value in updated.items()
        }

    ratings = {
        label: 400.0 * math.log10(strengths[label])
        for label in labels
    }
    worst_case = {
        label: min(
            float(value)
            for other, value in score_matrix[label].items()
            if other != label and value is not None
        )
        for label in labels
    }

    cycles: list[list[str]] = []
    for a, b, c in itertools.combinations(labels, 3):
        if (
            float(score_matrix[a][b]) > 0.5
            and float(score_matrix[b][c]) > 0.5
            and float(score_matrix[c][a]) > 0.5
        ):
            cycles.append([a, b, c])
        elif (
            float(score_matrix[a][c]) > 0.5
            and float(score_matrix[c][b]) > 0.5
            and float(score_matrix[b][a]) > 0.5
        ):
            cycles.append([a, c, b])

    return {
        "score_matrix": score_matrix,
        "games_matrix": games_matrix,
        "bradley_terry_strength": strengths,
        "elo_centered": ratings,
        "worst_case_score": worst_case,
        "directed_three_cycles": cycles,
        "directed_three_cycle_count": len(cycles),
    }


def _parse_checkpoint(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--checkpoint must use LABEL=PATH")
    label, path_raw = raw.split("=", 1)
    label = label.strip()
    path = Path(path_raw).expanduser().resolve()
    if not label:
        raise ValueError("checkpoint label must not be empty")
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sample-moves",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--cpp-threads", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoints = [_parse_checkpoint(raw) for raw in args.checkpoint]
    labels = [label for label, _path in checkpoints]
    if len(checkpoints) < 2 or len(set(labels)) != len(labels):
        raise ValueError("At least two uniquely labeled checkpoints are required.")
    if int(args.games) <= 0 or int(args.games) % 2 != 0:
        raise ValueError("--games must be a positive even number.")
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    match_dir = output.parent / f"{output.stem}_matches"
    match_dir.mkdir(parents=True, exist_ok=True)

    matches: list[dict[str, Any]] = []
    for pair_index, ((left, left_path), (right, right_path)) in enumerate(
        itertools.combinations(checkpoints, 2)
    ):
        match_name = f"{left}_vs_{right}"
        report_path = match_dir / f"{pair_index:03d}_{match_name}.json"
        command = [
            sys.executable,
            "scripts/eval_checkpoint.py",
            "--challenger_checkpoint",
            str(left_path),
            "--previous_checkpoint",
            str(right_path),
            "--backend",
            "portable",
            "--device",
            str(args.device),
            "--eval_workers",
            "1",
            "--mcts_simulations",
            str(int(args.simulations)),
            "--temperature",
            str(float(args.temperature)),
            "--v1_concurrent_games",
            str(int(args.concurrency)),
            "--v1_opening_random_moves",
            "0",
            "--seed",
            str(int(args.seed) + pair_index),
            "--match_name",
            match_name,
            "--output_json",
            str(report_path),
            "--eval_games_vs_previous",
            str(int(args.games)),
            "--portable_mcts_backend",
            "cpp",
            "--portable_cpp_threads",
            str(int(args.cpp_threads)),
        ]
        if bool(args.sample_moves):
            command.append("--sample_moves")
        subprocess.run(command, cwd=ROOT_DIR, check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audit = report.get("portable_mcts_audit")
        if not isinstance(audit, dict) or any(
            int(audit.get(key, -1) or 0) != 0
            for key in ("fallback_count", "illegal_action_count", "non_finite_count")
        ):
            raise RuntimeError(f"League health audit failed: {report_path}")
        rows = report.get("results")
        row = next(
            item
            for item in rows
            if isinstance(item, dict) and item.get("name") == match_name
        )
        row = dict(row)
        color_games = [
            int(value.get("games", 0) or 0)
            for value in row.get("color_breakdown", {}).values()
            if isinstance(value, dict)
        ]
        if (
            len(color_games) != 2
            or any(count != int(args.games) // 2 for count in color_games)
        ):
            raise RuntimeError(f"League color split is not exact: {report_path}")
        row["score"] = _score(row)
        row["win_wilson_95"] = _wilson(
            int(row["wins"]), int(row["total_games"])
        )
        matches.append(
            {
                "challenger": left,
                "opponent": right,
                "challenger_sha256": _sha256(left_path),
                "opponent_sha256": _sha256(right_path),
                "result": row,
                "report": str(report_path),
                "audit": audit,
            }
        )

    result = {
        "checkpoints": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in checkpoints
        },
        "conditions": {
            "games_per_pair": int(args.games),
            "simulations": int(args.simulations),
            "temperature": float(args.temperature),
            "sample_moves": bool(args.sample_moves),
            "seed": int(args.seed),
            "device": str(args.device),
            "concurrency": int(args.concurrency),
            "portable_cpp_threads": int(args.cpp_threads),
        },
        "matches": matches,
        "analysis": analyze_league_results(labels, matches),
    }
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"league_report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
