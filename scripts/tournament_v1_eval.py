#!/usr/bin/env python3
"""Run a staged v1 checkpoint tournament and select the strongest model."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


MODEL_NAME_RE = re.compile(r"^model_iter_0\d\d\.pt$")
GAMES_PER_MATCH = 1000
TEMPERATURE = 1.0
SAMPLE_MOVES = True
MATCH_POINTS_WIN = 3
MATCH_POINTS_DRAW = 1
MATCH_POINTS_LOSS = 0


@dataclass(frozen=True)
class StageConfig:
    name: str
    group_size: int
    advance_count: int
    expected_count: int


@dataclass(frozen=True)
class ModelRef:
    name: str
    path: str
    seed_order: int


STAGE_PLAN: Tuple[StageConfig, ...] = (
    StageConfig(name="stage_1_80_to_32", group_size=5, advance_count=2, expected_count=80),
    StageConfig(name="stage_2_32_to_16", group_size=4, advance_count=2, expected_count=32),
    StageConfig(name="stage_3_16_to_8", group_size=4, advance_count=2, expected_count=16),
    StageConfig(name="stage_4_8_to_4", group_size=4, advance_count=2, expected_count=8),
    StageConfig(name="final_4_to_1", group_size=4, advance_count=1, expected_count=4),
)

_EVAL_FN = None


def _get_eval_v1_fn():
    global _EVAL_FN
    if _EVAL_FN is None:
        from eval_checkpoint import evaluate_against_agent_parallel_v1  # noqa: WPS433

        _EVAL_FN = evaluate_against_agent_parallel_v1
    return _EVAL_FN


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _read_int_env(key: str, default: int) -> int:
    raw = str(os.environ.get(key, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _default_eval_devices() -> str:
    raw = str(os.environ.get("EVAL_DEVICES", "")).strip()
    return raw if raw else "cuda:0,cuda:1,cuda:2,cuda:3"


def _parse_devices(raw: str) -> List[str]:
    devices = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(devices) < 4:
        raise ValueError(
            f"At least 4 eval devices are required, got {len(devices)} from: {raw!r}"
        )
    if len(devices) > 4:
        devices = devices[:4]
    return devices


def _discover_models(checkpoint_dir: Path) -> List[Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint_dir is not a directory: {checkpoint_dir}")
    candidates = sorted(
        p for p in checkpoint_dir.iterdir() if p.is_file() and MODEL_NAME_RE.match(p.name)
    )
    if len(candidates) != 80:
        raise RuntimeError(
            f"Expected exactly 80 checkpoints matching model_iter_0**.pt, got {len(candidates)} "
            f"under {checkpoint_dir}"
        )
    return candidates


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _log_line(msg: str, log_file: Path) -> None:
    print(msg, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _new_standing(model: ModelRef) -> Dict[str, Any]:
    return {
        "model_name": model.name,
        "model_path": model.path,
        "seed_order": int(model.seed_order),
        "match_points": 0,
        "match_wins": 0,
        "match_losses": 0,
        "match_draws": 0,
        "game_wins": 0,
        "game_losses": 0,
        "game_draws": 0,
        "game_total": 0,
        "game_win_rate": 0.0,
        "game_wins_minus_losses": 0,
    }


def _finalize_standing(row: Dict[str, Any]) -> None:
    total = int(row["game_total"])
    wins = int(row["game_wins"])
    losses = int(row["game_losses"])
    row["game_win_rate"] = 0.0 if total <= 0 else float(wins / total)
    row["game_wins_minus_losses"] = int(wins - losses)


def _ranking_key(row: Dict[str, Any]) -> Tuple[int, float, int, int]:
    return (
        int(row["match_points"]),
        float(row["game_win_rate"]),
        int(row["game_wins_minus_losses"]),
        -int(row["seed_order"]),
    )


def _play_match(
    challenger: ModelRef,
    opponent: ModelRef,
    *,
    devices: Sequence[str],
    eval_workers: int,
    mcts_simulations: int,
    v1_concurrent_games: int,
    v1_opening_random_moves: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    eval_v1_fn = _get_eval_v1_fn()
    stats = eval_v1_fn(
        challenger_checkpoint=challenger.path,
        opponent_checkpoint=opponent.path,
        num_games=int(GAMES_PER_MATCH),
        device=str(devices[0]),
        mcts_simulations=int(mcts_simulations),
        temperature=float(TEMPERATURE),
        num_workers=int(eval_workers),
        devices=list(devices),
        concurrent_games=int(v1_concurrent_games),
        opening_random_moves=int(v1_opening_random_moves),
        sample_moves=bool(SAMPLE_MOVES),
    )
    elapsed = float(max(0.0, time.perf_counter() - started))

    challenger_points = MATCH_POINTS_LOSS
    opponent_points = MATCH_POINTS_LOSS
    if int(stats.wins) > int(stats.losses):
        result = "challenger_win"
        winner = challenger.name
        challenger_points = MATCH_POINTS_WIN
        opponent_points = MATCH_POINTS_LOSS
    elif int(stats.wins) < int(stats.losses):
        result = "opponent_win"
        winner = opponent.name
        challenger_points = MATCH_POINTS_LOSS
        opponent_points = MATCH_POINTS_WIN
    else:
        result = "draw"
        winner = None
        challenger_points = MATCH_POINTS_DRAW
        opponent_points = MATCH_POINTS_DRAW

    return {
        "challenger_name": challenger.name,
        "challenger_path": challenger.path,
        "opponent_name": opponent.name,
        "opponent_path": opponent.path,
        "wins": int(stats.wins),
        "losses": int(stats.losses),
        "draws": int(stats.draws),
        "total_games": int(stats.total_games),
        "challenger_black_games": int(GAMES_PER_MATCH // 2),
        "opponent_black_games": int(GAMES_PER_MATCH // 2),
        "temperature": float(TEMPERATURE),
        "sample_moves": bool(SAMPLE_MOVES),
        "result": str(result),
        "winner": winner,
        "challenger_match_points": int(challenger_points),
        "opponent_match_points": int(opponent_points),
        "elapsed_sec": elapsed,
    }


def _update_standings_from_match(
    standings: Dict[str, Dict[str, Any]],
    match_row: Dict[str, Any],
) -> None:
    a = standings[str(match_row["challenger_name"])]
    b = standings[str(match_row["opponent_name"])]

    a["game_wins"] += int(match_row["wins"])
    a["game_losses"] += int(match_row["losses"])
    a["game_draws"] += int(match_row["draws"])
    a["game_total"] += int(match_row["total_games"])

    b["game_wins"] += int(match_row["losses"])
    b["game_losses"] += int(match_row["wins"])
    b["game_draws"] += int(match_row["draws"])
    b["game_total"] += int(match_row["total_games"])

    a["match_points"] += int(match_row["challenger_match_points"])
    b["match_points"] += int(match_row["opponent_match_points"])

    result = str(match_row["result"])
    if result == "challenger_win":
        a["match_wins"] += 1
        b["match_losses"] += 1
    elif result == "opponent_win":
        a["match_losses"] += 1
        b["match_wins"] += 1
    else:
        a["match_draws"] += 1
        b["match_draws"] += 1


def _run_stage(
    stage_cfg: StageConfig,
    participants: List[ModelRef],
    *,
    devices: Sequence[str],
    eval_workers: int,
    mcts_simulations: int,
    v1_concurrent_games: int,
    v1_opening_random_moves: int,
    report: Dict[str, Any],
    output_json: Path,
    log_file: Path,
) -> List[ModelRef]:
    if len(participants) != int(stage_cfg.expected_count):
        raise RuntimeError(
            f"{stage_cfg.name}: expected {stage_cfg.expected_count} models, got {len(participants)}"
        )
    if len(participants) % int(stage_cfg.group_size) != 0:
        raise RuntimeError(
            f"{stage_cfg.name}: participant count {len(participants)} is not divisible by "
            f"group_size={stage_cfg.group_size}"
        )

    group_count = len(participants) // int(stage_cfg.group_size)
    stage_payload: Dict[str, Any] = {
        "name": stage_cfg.name,
        "input_count": len(participants),
        "group_size": int(stage_cfg.group_size),
        "group_count": int(group_count),
        "advance_count_per_group": int(stage_cfg.advance_count),
        "groups": [],
        "qualified": [],
    }
    report["stages"].append(stage_payload)
    _log_line(
        f"[tournament] {stage_cfg.name} start: models={len(participants)} groups={group_count} "
        f"group_size={stage_cfg.group_size} advance={stage_cfg.advance_count}",
        log_file,
    )

    qualified: List[ModelRef] = []
    for group_idx in range(group_count):
        start = group_idx * int(stage_cfg.group_size)
        end = start + int(stage_cfg.group_size)
        group_models = participants[start:end]
        group_names = [m.name for m in group_models]
        _log_line(
            f"[tournament] {stage_cfg.name} group={group_idx + 1}/{group_count} members={group_names}",
            log_file,
        )

        standings: Dict[str, Dict[str, Any]] = {m.name: _new_standing(m) for m in group_models}
        matches: List[Dict[str, Any]] = []
        pairings = list(itertools.combinations(group_models, 2))
        for pair_idx, (challenger, opponent) in enumerate(pairings, start=1):
            _log_line(
                f"[match] {stage_cfg.name} g{group_idx + 1} {pair_idx}/{len(pairings)} "
                f"{challenger.name} vs {opponent.name} (games={GAMES_PER_MATCH}, temp={TEMPERATURE})",
                log_file,
            )
            match_row = _play_match(
                challenger,
                opponent,
                devices=devices,
                eval_workers=int(eval_workers),
                mcts_simulations=int(mcts_simulations),
                v1_concurrent_games=int(v1_concurrent_games),
                v1_opening_random_moves=int(v1_opening_random_moves),
            )
            matches.append(match_row)
            _update_standings_from_match(standings, match_row)
            _log_line(
                f"[match] result {challenger.name} vs {opponent.name}: "
                f"W-L-D={match_row['wins']}-{match_row['losses']}-{match_row['draws']} "
                f"winner={match_row['winner'] or 'draw'} elapsed={match_row['elapsed_sec']:.2f}s",
                log_file,
            )
            _write_json(output_json, report)

        table = list(standings.values())
        for row in table:
            _finalize_standing(row)
        table_sorted = sorted(table, key=_ranking_key, reverse=True)
        for rank, row in enumerate(table_sorted, start=1):
            row["rank"] = int(rank)

        qualified_rows = table_sorted[: int(stage_cfg.advance_count)]
        qualified_group = [
            next(m for m in group_models if m.name == str(row["model_name"])) for row in qualified_rows
        ]
        qualified.extend(qualified_group)

        group_payload = {
            "group_index": int(group_idx + 1),
            "members": [
                {"name": m.name, "path": m.path, "seed_order": int(m.seed_order)} for m in group_models
            ],
            "matches": matches,
            "table": table_sorted,
            "qualified": [
                {"name": m.name, "path": m.path, "seed_order": int(m.seed_order)}
                for m in qualified_group
            ],
        }
        stage_payload["groups"].append(group_payload)
        _log_line(
            f"[tournament] {stage_cfg.name} g{group_idx + 1} qualified="
            f"{[m.name for m in qualified_group]}",
            log_file,
        )
        _write_json(output_json, report)

    stage_payload["qualified"] = [
        {"name": m.name, "path": m.path, "seed_order": int(m.seed_order)} for m in qualified
    ]
    _write_json(output_json, report)
    _log_line(
        f"[tournament] {stage_cfg.name} done: qualified_count={len(qualified)}",
        log_file,
    )
    return qualified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged v1 tournament over model_iter_0** checkpoints.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models",
        help="Directory containing model_iter_0**.pt checkpoints (expects exactly 80 models).",
    )
    parser.add_argument(
        "--eval_devices",
        type=str,
        default=_default_eval_devices(),
        help="Comma-separated CUDA devices; first 4 devices are used.",
    )
    parser.add_argument(
        "--mcts_simulations",
        type=int,
        default=_read_int_env("EVAL_MCTS_SIMULATIONS", 1024),
        help="MCTS simulations per move (default follows scripts/big_train_v1.sh).",
    )
    parser.add_argument(
        "--v1_concurrent_games",
        type=int,
        default=_read_int_env("EVAL_V1_CONCURRENT_GAMES", 8192),
        help="Concurrent games hint for v1 eval (default follows scripts/big_train_v1.sh).",
    )
    parser.add_argument(
        "--v1_opening_random_moves",
        type=int,
        default=_read_int_env("EVAL_V1_OPENING_RANDOM_MOVES", 0),
        help="Opening random moves for eval (default follows scripts/big_train_v1.sh).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260226,
        help="Shuffle seed used for initial model order and final tie-break fallback.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output JSON path (default: results/v1_tournament_<timestamp>.json).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_dir = Path(str(args.checkpoint_dir)).expanduser().resolve()
    devices = _parse_devices(str(args.eval_devices))
    eval_workers = 4
    mcts_simulations = max(1, int(args.mcts_simulations))
    v1_concurrent_games = max(1, int(args.v1_concurrent_games))
    v1_opening_random_moves = max(0, int(args.v1_opening_random_moves))
    seed = int(args.seed)

    started_iso = _now_iso()
    if str(args.output_json).strip():
        output_json = Path(str(args.output_json)).expanduser().resolve()
    else:
        output_json = Path("results").resolve() / f"v1_tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file = output_json.with_suffix(".log")
    _ensure_parent(log_file)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"[tournament] start={started_iso}\n")

    model_paths = _discover_models(checkpoint_dir)
    rng = random.Random(seed)
    rng.shuffle(model_paths)
    participants: List[ModelRef] = [
        ModelRef(name=p.name, path=str(p), seed_order=idx) for idx, p in enumerate(model_paths)
    ]

    report: Dict[str, Any] = {
        "created_at": started_iso,
        "finished_at": None,
        "checkpoint_dir": str(checkpoint_dir),
        "scan_regex": str(MODEL_NAME_RE.pattern),
        "games_per_match": int(GAMES_PER_MATCH),
        "temperature": float(TEMPERATURE),
        "sample_moves": bool(SAMPLE_MOVES),
        "seed": int(seed),
        "eval_config": {
            "eval_devices": list(devices),
            "eval_workers": int(eval_workers),
            "mcts_simulations": int(mcts_simulations),
            "v1_concurrent_games": int(v1_concurrent_games),
            "v1_opening_random_moves": int(v1_opening_random_moves),
        },
        "initial_order": [
            {"name": m.name, "path": m.path, "seed_order": int(m.seed_order)} for m in participants
        ],
        "stages": [],
        "champion": None,
    }
    _write_json(output_json, report)

    _log_line(
        f"[tournament] discovered {len(participants)} checkpoints from {checkpoint_dir}; "
        f"shuffled with seed={seed}",
        log_file,
    )
    _log_line(
        f"[tournament] eval_devices={devices} eval_workers={eval_workers} "
        f"mcts_simulations={mcts_simulations} v1_concurrent_games={v1_concurrent_games} "
        f"v1_opening_random_moves={v1_opening_random_moves}",
        log_file,
    )

    current = participants
    for stage_cfg in STAGE_PLAN:
        current = _run_stage(
            stage_cfg,
            current,
            devices=devices,
            eval_workers=int(eval_workers),
            mcts_simulations=int(mcts_simulations),
            v1_concurrent_games=int(v1_concurrent_games),
            v1_opening_random_moves=int(v1_opening_random_moves),
            report=report,
            output_json=output_json,
            log_file=log_file,
        )

    if len(current) != 1:
        raise RuntimeError(f"Final stage must output exactly one champion, got {len(current)}")

    champion = current[0]
    report["champion"] = {
        "name": champion.name,
        "path": champion.path,
        "seed_order": int(champion.seed_order),
    }
    report["finished_at"] = _now_iso()
    _write_json(output_json, report)

    _log_line(f"[tournament] champion={champion.name} path={champion.path}", log_file)
    _log_line(f"[tournament] report_json={output_json}", log_file)
    _log_line(f"[tournament] log_file={log_file}", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
