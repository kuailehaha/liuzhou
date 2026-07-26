#!/usr/bin/env python
"""Run Phase-3 16/32-simulation pilots, select one, then start formal training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path}")
    return payload


def _require_file(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_command(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    simulations: int,
    hours: float,
    formal: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/long_train_portable_mps.py",
        "--run-dir",
        str(run_dir),
        "--hours",
        str(float(hours)),
        "--initial-checkpoint",
        str(args.initial_checkpoint),
        "--initial-optimizer-state",
        str(args.initial_optimizer),
        "--lineage-parent-run",
        str(args.phase2_parent_run),
        "--lineage-parent-iteration",
        "2478",
        "--lineage-parent-best-iteration",
        "2100",
        "--state-encoding-schema",
        "markov_v2",
        "--symmetry-augmentation",
        "random_d4",
        "--optimizer-name",
        "adamw",
        "--strict-optimizer-restore",
        "--device",
        "mps",
        "--self-play-games",
        "128",
        "--self-play-concurrency",
        "128",
        "--portable-self-play-workers",
        "1",
        "--portable-mcts-backend",
        "cpp",
        "--portable-cpp-threads",
        "1",
        "--mcts-simulations",
        str(int(simulations)),
        "--temperature-init",
        "1",
        "--temperature-final",
        "1",
        "--policy-target-temperature",
        "1",
        "--policy-target-prior-pseudocount",
        "1",
        "--no-sample-moves",
        "--opening-random-start",
        "3",
        "--opening-random-final",
        "0",
        "--opening-random-anneal-fraction",
        "0.25",
        "--soft-label-alpha",
        "0",
        "--replay-window",
        "16",
        "--batch-size",
        "256",
        "--epochs",
        "3",
        "--lr-start",
        "0.0001",
        "--lr-final",
        "0.00002",
        "--eval-every",
        "10",
        "--eval-games-best",
        "500",
        "--eval-simulations",
        "64",
        "--eval-concurrency",
        "64",
        "--league-gate",
        "--rollback-on-gate-fail",
        "--league-screen-incumbent-games",
        "128",
        "--league-screen-anchor-games",
        "64",
        "--league-full-anchor-games",
        "200",
        "--league-cycle-screen-games",
        "64",
        "--league-cycle-screen-simulations",
        "16",
        "--no-eval-at-start",
        "--no-final-eval",
        "--seed",
        str(int(args.seed)),
    ]
    for anchor in args.anchors:
        command.extend(["--promotion-anchor", str(anchor)])
    if formal:
        command.extend(
            [
                "--quality-strength-eval",
                "--strength-eval-games",
                "500",
                "--strength-eval-simulations",
                "64",
                "--strength-diagnostic-games",
                "64",
                "--strength-diagnostic-simulations",
                "64",
                "--strength-screen-games",
                "64",
                "--strength-screen-simulations",
                "256",
                "--strength-screen-temperature",
                "0",
                "--no-strength-screen-sample-moves",
            ]
        )
    if (run_dir / "state.json").is_file():
        command.append("--resume")
    return command


def _run_training(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    simulations: int,
    hours: float,
    formal: bool,
    workflow_log: Path,
) -> None:
    summary = run_dir / "final_summary.json"
    if summary.is_file():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    command = _training_command(
        args=args,
        run_dir=run_dir,
        simulations=simulations,
        hours=hours,
        formal=formal,
    )
    resource_prefix = run_dir / "resources"
    with workflow_log.open("a", encoding="utf-8") as output:
        output.write(
            f"[{_utc_now()}] start simulations={simulations} "
            f"hours={hours} formal={formal} command={command!r}\n"
        )
        output.flush()
        environment = os.environ.copy()
        environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=environment,
            stdout=output,
            stderr=output,
            text=True,
        )
        monitor = subprocess.Popen(
            [
                sys.executable,
                "tools/monitor_portable_mps_resources.py",
                "--root-pid",
                str(process.pid),
                "--output-prefix",
                str(resource_prefix),
                "--interval",
                "2",
            ],
            cwd=ROOT_DIR,
            env=environment,
            stdout=output,
            stderr=output,
            text=True,
        )
        return_code = process.wait()
        if monitor.poll() is None:
            monitor.send_signal(signal.SIGTERM)
        monitor.wait(timeout=30)
        output.write(
            f"[{_utc_now()}] end simulations={simulations} "
            f"formal={formal} exit_code={return_code}\n"
        )
        output.flush()
    if return_code != 0:
        raise RuntimeError(
            f"Phase-3 training failed for {simulations} simulations."
        )
    if not summary.is_file():
        raise RuntimeError(f"Training exited without final summary: {summary}")


def _run_checked(
    command: list[str],
    *,
    log_path: Path,
) -> None:
    with log_path.open("a", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            stdout=output,
            stderr=output,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--initial-optimizer", required=True)
    parser.add_argument("--phase2-parent-run", required=True)
    parser.add_argument("--anchor", action="append", required=True)
    parser.add_argument("--pilot-hours", type=float, default=3.0)
    parser.add_argument("--formal-hours", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=2026072605)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(ROOT_DIR)
    args.initial_checkpoint = _require_file(args.initial_checkpoint)
    args.initial_optimizer = _require_file(args.initial_optimizer)
    args.phase2_parent_run = Path(args.phase2_parent_run).expanduser().resolve()
    if not args.phase2_parent_run.is_dir():
        raise FileNotFoundError(args.phase2_parent_run)
    parsed_anchors: list[tuple[str, Path]] = []
    for raw in args.anchor:
        if "=" not in raw:
            raise ValueError("--anchor must use LABEL=PATH")
        label, path_raw = raw.split("=", 1)
        if not label.strip():
            raise ValueError("Anchor label must not be empty.")
        parsed_anchors.append((label.strip(), _require_file(path_raw)))
    if len(parsed_anchors) != 3:
        raise ValueError("Phase-3 workflow requires exactly three fixed anchors.")
    args.anchors = [path for _label, path in parsed_anchors]

    root = Path(args.root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "workflow_state.json"
    workflow_log = root / "workflow.log"
    state = _load_json(state_path)
    if not state:
        state = {
            "created_utc": _utc_now(),
            "phase2_parent_iteration": 2478,
            "phase2_best_iteration": 2100,
            "phase2_parent_run": str(args.phase2_parent_run),
            "initial_checkpoint": str(args.initial_checkpoint),
            "initial_checkpoint_sha256": _sha256(args.initial_checkpoint),
            "initial_optimizer": str(args.initial_optimizer),
            "initial_optimizer_sha256": _sha256(args.initial_optimizer),
            "anchors": {
                label: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for label, path in parsed_anchors
            },
            "pilot_hours_each": float(args.pilot_hours),
            "formal_hours": float(args.formal_hours),
            "seed": int(args.seed),
            "stage": "initialized",
        }
        _atomic_json(state_path, state)

    pilot16 = root / "pilot_16sim_3h"
    pilot32 = root / "pilot_32sim_3h"
    _run_training(
        args=args,
        run_dir=pilot16,
        simulations=16,
        hours=float(args.pilot_hours),
        formal=False,
        workflow_log=workflow_log,
    )
    state["stage"] = "pilot16_complete"
    state["pilot16_completed_utc"] = _utc_now()
    _atomic_json(state_path, state)

    _run_training(
        args=args,
        run_dir=pilot32,
        simulations=32,
        hours=float(args.pilot_hours),
        formal=False,
        workflow_log=workflow_log,
    )
    state["stage"] = "pilots_complete"
    state["pilot32_completed_utc"] = _utc_now()
    _atomic_json(state_path, state)

    selection_report = root / "pilot_selection.json"
    if not selection_report.is_file():
        selection_command = [
            sys.executable,
            "tools/select_phase3_pilot.py",
            "--pilot16-run",
            str(pilot16),
            "--pilot32-run",
            str(pilot32),
            "--output-json",
            str(selection_report),
            "--simulations",
            "64",
            "--direct-games",
            "500",
            "--anchor-games",
            "200",
            "--concurrency",
            "64",
            "--cpp-threads",
            "1",
            "--seed",
            str(int(args.seed) + 1_000_000),
        ]
        for label, path in parsed_anchors:
            selection_command.extend(["--anchor", f"{label}={path}"])
        _run_checked(selection_command, log_path=workflow_log)
    selection = _load_json(selection_report)
    selected_simulations = int(selection["selected_simulations"])
    if selected_simulations not in {16, 32}:
        raise RuntimeError("Pilot selection report has an invalid winner.")
    state["stage"] = "pilot_selected"
    state["selected_simulations"] = selected_simulations
    state["selection_report"] = str(selection_report)
    _atomic_json(state_path, state)

    formal = root / f"formal_40h_{selected_simulations}sim"
    if not (formal / "state.json").exists():
        state["formal_started_utc"] = _utc_now()
        state["stage"] = "formal_starting"
        _atomic_json(state_path, state)
    _run_training(
        args=args,
        run_dir=formal,
        simulations=selected_simulations,
        hours=float(args.formal_hours),
        formal=True,
        workflow_log=workflow_log,
    )
    state["formal_completed_utc"] = _utc_now()
    state["stage"] = "formal_complete"
    state["formal_run"] = str(formal)
    _atomic_json(state_path, state)

    final_league = root / "final_incumbent_league.json"
    if not final_league.is_file():
        league_command = [
            sys.executable,
            "tools/eval_portable_league.py",
            "--checkpoint",
            f"phase3_final={formal / 'checkpoints' / 'best_model.pt'}",
            "--games",
            "200",
            "--simulations",
            "64",
            "--temperature",
            "1",
            "--sample-moves",
            "--seed",
            str(int(args.seed) + 2_000_000),
            "--device",
            "mps",
            "--concurrency",
            "64",
            "--cpp-threads",
            "1",
            "--output-json",
            str(final_league),
        ]
        for label, path in parsed_anchors:
            league_command.extend(["--checkpoint", f"{label}={path}"])
        _run_checked(league_command, log_path=workflow_log)
    state["stage"] = "complete"
    state["final_league_report"] = str(final_league)
    state["completed_utc"] = _utc_now()
    _atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
