#!/usr/bin/env python3
"""Resumable wall-clock V1 portable training for Apple Silicon.

The orchestrator keeps immutable per-iteration model checkpoints together with
rolling recovery state and two best-model aliases:

* ``model_iter_XXXXXX.pt`` retains every successfully committed iteration;
* ``best_model.pt`` is the incumbent selected by a direct 500-game match;
* ``best_vs_random.pt`` has the strongest observed fixed-seed random score.

All rule/search/training work stays in the existing V1 entry points.  This
module owns only scheduling, persistence, validation and promotion policy.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] [portable_long] {message}", flush=True)


def _clamp_progress(progress: float) -> float:
    return max(0.0, min(1.0, float(progress)))


def cosine_schedule(start: float, final: float, progress: float) -> float:
    p = _clamp_progress(progress)
    weight = 0.5 * (1.0 + math.cos(math.pi * p))
    return float(final) + (float(start) - float(final)) * weight


def linear_int_schedule(start: int, final: int, progress: float) -> int:
    p = _clamp_progress(progress)
    return max(0, int(round(float(start) + (float(final) - float(start)) * p)))


def wilson_interval(wins: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    n = int(total)
    if n <= 0:
        return 0.0, 0.0
    p = int(wins) / float(n)
    denominator = 1.0 + (z * z / n)
    center = (p + (z * z / (2.0 * n))) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def random_result_rank(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row.get("wins", 0) or 0), -int(row.get("losses", 0) or 0)


def candidate_beats_incumbent(row: Dict[str, Any], promotion_score: float) -> bool:
    total_games = int(row.get("total_games", 0) or 0)
    if total_games <= 0:
        return False
    wins = int(row.get("wins", 0) or 0)
    draws = int(row.get("draws", 0) or 0)
    score = (wins + (0.5 * draws)) / float(total_games)
    return score >= float(promotion_score)


def target_requires_confirmation(state: Dict[str, Any], target_win_rate: float) -> bool:
    if bool(state.get("target_reached", False)):
        return False
    best_result = state.get("best_random_result")
    if not isinstance(best_result, dict):
        return False
    total_games = int(best_result.get("total_games", 0) or 0)
    if total_games <= 0:
        return False
    return int(best_result.get("wins", 0) or 0) / total_games >= float(target_win_rate)


def select_replay_inputs(
    files: Sequence[Path], *, primary: Path, window: int
) -> List[Path]:
    history = sorted(path for path in files if path != primary)
    count = max(0, int(window))
    return history[-count:] if count > 0 else []


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def atomic_copy(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    if destination.exists() and source_resolved == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source_resolved, temporary)
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_train_metrics(path: Path, *, require_optimizer_loaded: bool) -> Dict[str, Any]:
    payload = _load_json(path)
    if isinstance(payload, list) and payload:
        row = payload[-1]
    elif isinstance(payload, dict):
        row = payload
    else:
        raise RuntimeError(f"train metrics have no row: {path}")
    if not isinstance(row, dict):
        raise RuntimeError(f"train metrics row is not an object: {path}")

    for key in ("train_avg_loss", "train_avg_policy_loss", "train_avg_value_loss"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite required train metric {key}={value!r}: {path}")
    for key in (
        "filtered_non_finite_samples",
        "self_play_value_nonfinite_local",
        "self_play_soft_value_nonfinite_local",
        "device_fallback_count",
    ):
        if int(row.get(key, 0) or 0) != 0:
            raise RuntimeError(f"train health check failed {key}={row.get(key)!r}: {path}")
    if require_optimizer_loaded and row.get("optimizer_loaded") is not True:
        raise RuntimeError(
            "optimizer continuity was lost: "
            f"loaded={row.get('optimizer_loaded')!r} "
            f"error={row.get('optimizer_load_error')!r} path={path}"
        )
    return row


def audit_selfplay_stats(path: Path, *, expected_games: int) -> Dict[str, Any]:
    row = _load_json(path)
    if not isinstance(row, dict):
        raise RuntimeError(f"self-play stats are not an object: {path}")
    if int(row.get("num_games", 0) or 0) != int(expected_games):
        raise RuntimeError(
            f"self-play game count mismatch: got={row.get('num_games')} "
            f"expected={expected_games} path={path}"
        )
    for key in ("fallback_count",):
        if int(row.get(key, 0) or 0) != 0:
            raise RuntimeError(f"self-play fallback detected {key}={row.get(key)!r}: {path}")
    for summary_key in (
        "value_target_summary",
        "soft_value_target_summary",
        "mixed_value_target_summary",
    ):
        summary = row.get(summary_key)
        if isinstance(summary, dict) and int(summary.get("nonfinite_count", 0) or 0) != 0:
            raise RuntimeError(f"non-finite {summary_key} in {path}")
    return row


def audit_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint has no model state: {path}")
    bad = [name for name, tensor in state.items() if not bool(torch.isfinite(tensor).all().item())]
    if bad:
        raise RuntimeError(f"checkpoint contains non-finite tensors: {bad[:5]} path={path}")
    return {
        "sha256": sha256_file(path),
        "tensor_count": len(state),
        "parameter_elements": sum(int(tensor.numel()) for tensor in state.values()),
    }


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _command_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_command(
    command: Sequence[str],
    *,
    label: str,
    retries: int,
    dry_run: bool,
    prepare_attempt: Optional[Callable[[], None]] = None,
) -> None:
    log(f"{label} command={shlex.join(list(command))}")
    if dry_run:
        return
    for attempt in range(1, max(0, int(retries)) + 2):
        if prepare_attempt is not None:
            prepare_attempt()
        started = time.perf_counter()
        completed = subprocess.run(list(command), cwd=ROOT_DIR, env=_command_env(), check=False)
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            log(f"{label} complete elapsed_sec={elapsed:.2f}")
            return
        log(
            f"{label} failed exit_code={completed.returncode} "
            f"attempt={attempt}/{max(0, int(retries)) + 1} elapsed_sec={elapsed:.2f}"
        )
        if attempt <= max(0, int(retries)):
            time.sleep(float(attempt * 5))
    raise RuntimeError(f"{label} exhausted retries")


def _git_metadata() -> Dict[str, Any]:
    def output(command: Sequence[str]) -> str:
        result = subprocess.run(
            list(command), cwd=ROOT_DIR, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": output(["git", "rev-parse", "HEAD"]),
        "branch": output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(output(["git", "status", "--porcelain"])),
    }


def _on_ac_power() -> bool:
    if platform.system() != "Darwin":
        return True
    result = subprocess.run(
        ["pmset", "-g", "batt"], text=True, capture_output=True, check=False
    )
    return result.returncode == 0 and "AC Power" in result.stdout


def _external_display_connected() -> bool:
    if platform.system() != "Darwin":
        return True
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    connection_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if "Connection Type:" in line
    ]
    return any(not line.endswith("Internal") for line in connection_lines)


def _preflight(args: argparse.Namespace) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip() == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is forbidden for verified portable runs.")
    if str(args.device) == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
    if bool(args.require_ac) and not _on_ac_power():
        raise RuntimeError("Long training requires AC power; connect the power adapter first.")
    if bool(args.require_external_display) and not _external_display_connected():
        raise RuntimeError(
            "Closed-display preflight failed: no external display was detected. "
            "Apple-supported clamshell operation also requires AC power and an external keyboard/mouse."
        )
    if float(args.hours) <= 0.0:
        raise ValueError("--hours must be positive.")
    if str(args.portable_mcts_backend).strip().lower() not in {"python", "cpp"}:
        raise ValueError("--portable-mcts-backend must be python or cpp.")
    if str(args.portable_mcts_backend).strip().lower() == "cpp":
        from v1.python.portable_cpp_loader import load_portable_cpp

        load_portable_cpp(required=True)
    for name in (
        "self_play_games",
        "self_play_concurrency",
        "portable_self_play_workers",
        "portable_cpp_threads",
        "eval_concurrency",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("eval_games_random", "eval_games_best", "final_eval_games"):
        value = int(getattr(args, name))
        if value <= 0 or value % 2 != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive even number.")
    if int(args.eval_every) <= 0:
        raise ValueError("--eval-every must be positive.")
    if int(args.batch_size) <= 0 or int(args.epochs) <= 0:
        raise ValueError("--batch-size and --epochs must be positive.")
    if int(args.replay_window) < 0:
        raise ValueError("--replay-window cannot be negative.")
    if not 0.0 < float(args.target_win_rate) <= 1.0:
        raise ValueError("--target-win-rate must be in (0, 1].")
    if not 0.5 < float(args.incumbent_promotion_score) <= 1.0:
        raise ValueError("--incumbent-promotion-score must be in (0.5, 1].")


def _config_signature(args: argparse.Namespace) -> Dict[str, Any]:
    keys = (
        "hours",
        "device",
        "self_play_games",
        "self_play_concurrency",
        "portable_self_play_workers",
        "portable_mcts_backend",
        "portable_cpp_threads",
        "mcts_simulations",
        "temperature_init",
        "temperature_final",
        "temperature_threshold",
        "opening_random_start",
        "opening_random_final",
        "max_game_plies",
        "soft_label_alpha",
        "policy_draw_weight",
        "replay_window",
        "batch_size",
        "epochs",
        "lr_start",
        "lr_final",
        "weight_decay",
        "eval_every",
        "eval_games_random",
        "eval_games_best",
        "eval_simulations",
        "eval_concurrency",
        "incumbent_promotion_score",
        "target_win_rate",
        "seed",
    )
    return {key: getattr(args, key) for key in keys}


def _eval_result(path: Path, name: str) -> Dict[str, Any]:
    payload = _load_json(path)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"evaluation report has no results: {path}")
    for row in rows:
        if isinstance(row, dict) and str(row.get("name")) == str(name):
            return row
    raise RuntimeError(f"evaluation report has no row name={name!r}: {path}")


class PortableLongTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = str(Path(sys.executable).resolve())
        self.run_dir = _absolute_path(args.run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.replay_dir = self.run_dir / "replay"
        self.metrics_dir = self.run_dir / "metrics"
        self.eval_dir = self.run_dir / "eval"
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "final_summary.json"
        self.current_checkpoint = self.checkpoint_dir / "current.pt"
        self.best_checkpoint = self.checkpoint_dir / "best_model.pt"
        self.best_random_checkpoint = self.checkpoint_dir / "best_vs_random.pt"
        self.optimizer_checkpoint = self.checkpoint_dir / "optimizer.pt"
        self.optimizer_work = self.checkpoint_dir / "optimizer_work.pt"
        self.candidate_checkpoint = self.checkpoint_dir / "candidate.pt"
        self.rollback_current = self.checkpoint_dir / ".rollback_current.pt"
        self.rollback_optimizer = self.checkpoint_dir / ".rollback_optimizer.pt"
        self.state: Dict[str, Any] = {}

    def _save_state(self) -> None:
        self.state["updated_utc"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def _iteration_checkpoint_path(self, iteration: int) -> Path:
        return self.checkpoint_dir / f"model_iter_{int(iteration):06d}.pt"

    def _preserve_iteration_checkpoint(self, iteration: int) -> Path:
        if not self.current_checkpoint.is_file():
            raise RuntimeError(
                f"cannot retain iteration {iteration}: current checkpoint is missing"
            )
        destination = self._iteration_checkpoint_path(iteration)
        current_sha256 = sha256_file(self.current_checkpoint)
        if destination.exists():
            retained_sha256 = sha256_file(destination)
            if retained_sha256 != current_sha256:
                raise RuntimeError(
                    "refusing to overwrite retained checkpoint: "
                    f"iteration={iteration} path={destination} "
                    f"retained_sha256={retained_sha256} current_sha256={current_sha256}"
                )
        else:
            atomic_copy(self.current_checkpoint, destination)
        self.state["latest_iteration_checkpoint"] = str(destination)
        self.state["latest_iteration_checkpoint_sha256"] = current_sha256
        return destination

    def _event(self, event: str, **payload: Any) -> None:
        row = {"utc": utc_now(), "event": str(event), **payload}
        append_jsonl(self.events_path, row)

    def _set_stop_reason(self, reason: str) -> None:
        self.state["stop_reason"] = str(reason)
        self._save_state()
        self._event(
            "run_stopping",
            reason=str(reason),
            iteration=int(self.state.get("iteration", 0)),
        )

    def _clear_commit_rollback(self) -> None:
        self.rollback_current.unlink(missing_ok=True)
        self.rollback_optimizer.unlink(missing_ok=True)

    def _prepare_commit_rollback(self) -> None:
        self._clear_commit_rollback()
        for source, rollback in (
            (self.current_checkpoint, self.rollback_current),
            (self.optimizer_checkpoint, self.rollback_optimizer),
        ):
            if not source.is_file():
                continue
            try:
                os.link(source, rollback)
            except OSError:
                atomic_copy(source, rollback)

    @staticmethod
    def _artifact_matches(path: Path, expected_sha256: Optional[str]) -> bool:
        if expected_sha256 is None:
            return not path.exists()
        return path.is_file() and sha256_file(path) == str(expected_sha256)

    def _recover_interrupted_commit(self) -> bool:
        changed = False
        for key, path in (
            ("latest_checkpoint_sha256", self.current_checkpoint),
            ("latest_optimizer_sha256", self.optimizer_checkpoint),
        ):
            if key not in self.state:
                self.state[key] = sha256_file(path) if path.is_file() else None
                changed = True

        expected_checkpoint = self.state.get("latest_checkpoint_sha256")
        expected_optimizer = self.state.get("latest_optimizer_sha256")
        checkpoint_matches = self._artifact_matches(
            self.current_checkpoint, expected_checkpoint
        )
        optimizer_matches = self._artifact_matches(
            self.optimizer_checkpoint, expected_optimizer
        )
        if checkpoint_matches and optimizer_matches:
            self._clear_commit_rollback()
            return changed

        for path, rollback, expected in (
            (self.current_checkpoint, self.rollback_current, expected_checkpoint),
            (self.optimizer_checkpoint, self.rollback_optimizer, expected_optimizer),
        ):
            if self._artifact_matches(path, expected):
                continue
            if expected is None:
                path.unlink(missing_ok=True)
                continue
            if not rollback.is_file() or sha256_file(rollback) != str(expected):
                raise RuntimeError(
                    "interrupted model/optimizer commit cannot be recovered: "
                    f"artifact={path} expected_sha256={expected} rollback={rollback}"
                )
            atomic_copy(rollback, path)

        if not self._artifact_matches(
            self.current_checkpoint, expected_checkpoint
        ) or not self._artifact_matches(self.optimizer_checkpoint, expected_optimizer):
            raise RuntimeError(
                "interrupted model/optimizer commit recovery did not restore the committed pair"
            )
        self._clear_commit_rollback()
        return True

    def initialize(self) -> None:
        for directory in (
            self.checkpoint_dir,
            self.replay_dir,
            self.metrics_dir,
            self.eval_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        signature = _config_signature(self.args)
        if self.state_path.exists():
            if not bool(self.args.resume):
                raise RuntimeError(
                    f"run state already exists: {self.state_path}; pass --resume to continue"
                )
            state = _load_json(self.state_path)
            if not isinstance(state, dict):
                raise RuntimeError(f"invalid run state: {self.state_path}")
            if int(state.get("schema_version", 0)) != STATE_SCHEMA_VERSION:
                raise RuntimeError(f"unsupported state schema: {state.get('schema_version')}")
            previous_signature = state.get("config")
            migrated_config: Dict[str, Any] = {}
            additive_defaults = {
                "incumbent_promotion_score": float(self.args.incumbent_promotion_score),
                "portable_self_play_workers": int(self.args.portable_self_play_workers),
                "portable_mcts_backend": str(self.args.portable_mcts_backend),
                "portable_cpp_threads": int(self.args.portable_cpp_threads),
            }
            if isinstance(previous_signature, dict):
                previous_signature = dict(previous_signature)
                for field_name, default_value in additive_defaults.items():
                    if field_name not in previous_signature:
                        previous_signature[field_name] = default_value
                        migrated_config[field_name] = default_value
                state["config"] = previous_signature
            if previous_signature != signature:
                raise RuntimeError(
                    "resume configuration differs from state.json; "
                    "resume with the original frozen training/evaluation parameters"
                )
            previous_stop_reason = state.get("stop_reason")
            resume_after_target = (
                previous_stop_reason in {"target_confirmed", "target_confirmed_at_start"}
                and not bool(self.args.stop_on_target)
            )
            resume_stopped_run = previous_stop_reason in {"max_iterations", "dry_run"}
            resumed = bool(
                (resume_stopped_run or resume_after_target)
                and int(time.time()) < int(state.get("deadline_epoch", 0))
            )
            if resumed:
                state["stop_reason"] = None
                state.pop("ended_utc", None)
                state.pop("elapsed_sec", None)
                state["resume_count"] = int(state.get("resume_count", 0)) + 1
                state["last_resume_utc"] = utc_now()
                state["last_resume_stop_reason"] = previous_stop_reason
            self.state = state
            for field_name, field_value in migrated_config.items():
                self._event(
                    "config_migrated",
                    field=field_name,
                    value=field_value,
                )
            reconciled = self._recover_interrupted_commit()
            retained_checkpoint: Optional[Path] = None
            retained_checkpoint_created = False
            if self.current_checkpoint.is_file():
                retained_checkpoint = self._iteration_checkpoint_path(
                    int(state.get("iteration", 0))
                )
                retained_checkpoint_created = not retained_checkpoint.exists()
                retained_checkpoint = self._preserve_iteration_checkpoint(
                    int(state.get("iteration", 0))
                )
            if reconciled:
                self._event(
                    "commit_state_reconciled",
                    iteration=int(state.get("iteration", 0)),
                    checkpoint_sha256=state.get("latest_checkpoint_sha256"),
                    optimizer_sha256=state.get("latest_optimizer_sha256"),
                )
            if retained_checkpoint_created and retained_checkpoint is not None:
                self._event(
                    "iteration_checkpoint_retained",
                    iteration=int(state.get("iteration", 0)),
                    checkpoint=str(retained_checkpoint),
                    checkpoint_sha256=state.get(
                        "latest_iteration_checkpoint_sha256"
                    ),
                    source="resume",
                )
            if reconciled or resumed or retained_checkpoint is not None:
                self._save_state()
            if resumed:
                self._event(
                    "run_resumed",
                    iteration=int(state.get("iteration", 0)),
                    previous_stop_reason=previous_stop_reason,
                    target_reached=bool(state.get("target_reached", False)),
                )
            if not self.current_checkpoint.exists() and int(state.get("iteration", 0)) > 0:
                raise RuntimeError(f"resume checkpoint is missing: {self.current_checkpoint}")
            log(
                f"resume run_dir={self.run_dir} iteration={int(state.get('iteration', 0))} "
                f"deadline_epoch={int(state.get('deadline_epoch', 0))}"
            )
            return

        start_epoch = int(time.time())
        self.state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "start_epoch": start_epoch,
            "deadline_epoch": start_epoch + int(round(float(self.args.hours) * 3600.0)),
            "iteration": 0,
            "last_eval_iteration": None,
            "best_random_rank": None,
            "best_random_result": None,
            "target_reached": False,
            "target_confirmed_result": None,
            "latest_checkpoint_sha256": None,
            "latest_optimizer_sha256": None,
            "latest_iteration_checkpoint": None,
            "latest_iteration_checkpoint_sha256": None,
            "stop_reason": None,
            "config": signature,
            "git": _git_metadata(),
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "platform": platform.platform(),
                "device": str(self.args.device),
            },
        }
        initial = str(self.args.initial_checkpoint or "").strip()
        initial_optimizer = str(self.args.initial_optimizer_state or "").strip()
        if initial_optimizer and not initial:
            raise ValueError(
                "--initial-optimizer-state requires its matching --initial-checkpoint"
            )
        if initial:
            initial_path = _absolute_path(initial)
            if not initial_path.is_file():
                raise FileNotFoundError(f"initial checkpoint not found: {initial_path}")
            atomic_copy(initial_path, self.current_checkpoint)
            audit = audit_checkpoint(self.current_checkpoint)
            self.state["initial_checkpoint"] = str(initial_path)
            self.state["initial_checkpoint_audit"] = audit
            self.state["latest_checkpoint_sha256"] = audit["sha256"]
        if initial_optimizer:
            optimizer_path = _absolute_path(initial_optimizer)
            if not optimizer_path.is_file():
                raise FileNotFoundError(f"initial optimizer state not found: {optimizer_path}")
            atomic_copy(optimizer_path, self.optimizer_checkpoint)
            self.state["initial_optimizer_state"] = str(optimizer_path)
            self.state["latest_optimizer_sha256"] = sha256_file(
                self.optimizer_checkpoint
            )
        if self.current_checkpoint.is_file():
            retained_checkpoint = self._preserve_iteration_checkpoint(0)
            self._event(
                "iteration_checkpoint_retained",
                iteration=0,
                checkpoint=str(retained_checkpoint),
                checkpoint_sha256=self.state.get(
                    "latest_iteration_checkpoint_sha256"
                ),
                source="initial",
            )
        self._save_state()
        self._event("run_initialized", state=dict(self.state))
        log(
            f"new run_dir={self.run_dir} deadline_epoch={self.state['deadline_epoch']} "
            f"initial_checkpoint={initial or 'none'}"
        )

    def _progress(self) -> float:
        duration = max(1, int(self.state["deadline_epoch"]) - int(self.state["start_epoch"]))
        return _clamp_progress((time.time() - int(self.state["start_epoch"])) / duration)

    def _replay_files(self) -> List[Path]:
        return sorted(
            path
            for path in self.replay_dir.glob("selfplay_iter_*.pt")
            if re.fullmatch(r"selfplay_iter_\d{6}\.pt", path.name)
        )

    def _prune_replay(self) -> None:
        keep = max(0, int(self.args.replay_window)) + 1
        files = self._replay_files()
        for path in files[:-keep] if keep > 0 else files:
            path.unlink(missing_ok=True)
            for chunk in self.replay_dir.glob(f"{path.stem}.w*.chunk*.pt"):
                chunk.unlink(missing_ok=True)

    def _run_selfplay(self, iteration: int, progress: float) -> Tuple[Path, Dict[str, Any]]:
        tag = f"{iteration:06d}"
        output = self.replay_dir / f"selfplay_iter_{tag}.pt"
        stats_path = self.metrics_dir / f"selfplay_iter_{tag}.json"
        opening_moves = linear_int_schedule(
            int(self.args.opening_random_start),
            int(self.args.opening_random_final),
            progress,
        )
        command = [
            self.python,
            "scripts/train_entry.py",
            "--pipeline",
            "v1",
            "--stage",
            "selfplay",
            "--search_backend",
            "portable",
            "--device",
            str(self.args.device),
            "--devices",
            str(self.args.device),
            "--train_devices",
            str(self.args.device),
            "--train_strategy",
            "none",
            "--self_play_games",
            str(int(self.args.self_play_games)),
            "--self_play_concurrent_games",
            str(int(self.args.self_play_concurrency)),
            "--portable_self_play_workers",
            str(int(self.args.portable_self_play_workers)),
            "--portable_mcts_backend",
            str(self.args.portable_mcts_backend),
            "--portable_cpp_threads",
            str(int(self.args.portable_cpp_threads)),
            "--self_play_opening_random_moves",
            str(opening_moves),
            "--mcts_simulations",
            str(int(self.args.mcts_simulations)),
            "--temperature_init",
            str(float(self.args.temperature_init)),
            "--temperature_final",
            str(float(self.args.temperature_final)),
            "--temperature_threshold",
            str(int(self.args.temperature_threshold)),
            "--max_game_plies",
            str(int(self.args.max_game_plies)),
            "--model_init_seed",
            str(int(self.args.seed)),
            "--self_play_iteration_seed",
            str(int(self.args.seed) + iteration),
            "--soft_label_alpha",
            str(float(self.args.soft_label_alpha)),
            "--checkpoint_dir",
            str(self.checkpoint_dir),
            "--self_play_output",
            str(output),
            "--self_play_stats_json",
            str(stats_path),
        ]
        if self.current_checkpoint.exists():
            command.extend(["--load_checkpoint", str(self.current_checkpoint)])

        def prepare_selfplay_attempt() -> None:
            output.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)

        run_command(
            command,
            label=f"selfplay(iter={iteration},opening={opening_moves})",
            retries=int(self.args.stage_retries),
            dry_run=bool(self.args.dry_run),
            prepare_attempt=prepare_selfplay_attempt,
        )
        if bool(self.args.dry_run):
            return output, {}
        if not output.is_file() or not stats_path.is_file():
            raise RuntimeError(f"self-play artifacts are missing for iteration {iteration}")
        stats = audit_selfplay_stats(stats_path, expected_games=int(self.args.self_play_games))
        return output, stats

    def _run_train(
        self, iteration: int, progress: float, primary: Path
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        tag = f"{iteration:06d}"
        metrics_path = self.metrics_dir / f"train_iter_{tag}.json"
        had_optimizer = self.optimizer_checkpoint.is_file()

        def prepare_train_attempt() -> None:
            metrics_path.unlink(missing_ok=True)
            self.candidate_checkpoint.unlink(missing_ok=True)
            self.optimizer_work.unlink(missing_ok=True)
            if had_optimizer:
                atomic_copy(self.optimizer_checkpoint, self.optimizer_work)

        replay_inputs = select_replay_inputs(
            self._replay_files(), primary=primary, window=int(self.args.replay_window)
        )
        learning_rate = cosine_schedule(
            float(self.args.lr_start), float(self.args.lr_final), progress
        )
        command = [
            self.python,
            "scripts/train_entry.py",
            "--pipeline",
            "v1",
            "--stage",
            "train",
            "--search_backend",
            "portable",
            "--device",
            str(self.args.device),
            "--devices",
            str(self.args.device),
            "--train_devices",
            str(self.args.device),
            "--train_strategy",
            "none",
            "--batch_size",
            str(int(self.args.batch_size)),
            "--epochs",
            str(int(self.args.epochs)),
            "--lr",
            f"{learning_rate:.12g}",
            "--weight_decay",
            str(float(self.args.weight_decay)),
            "--soft_label_alpha",
            str(float(self.args.soft_label_alpha)),
            "--policy_draw_weight",
            str(float(self.args.policy_draw_weight)),
            "--checkpoint_dir",
            str(self.checkpoint_dir),
            "--self_play_input",
            str(primary),
            "--optimizer_state_path",
            str(self.optimizer_work),
            "--checkpoint_name",
            self.candidate_checkpoint.name,
            "--metrics_output",
            str(metrics_path),
            "--model_init_seed",
            str(int(self.args.seed)),
        ]
        if replay_inputs:
            command.extend(
                ["--self_play_replay_inputs", ",".join(str(path) for path in replay_inputs)]
            )
        if self.current_checkpoint.exists():
            command.extend(["--load_checkpoint", str(self.current_checkpoint)])
        run_command(
            command,
            label=f"train(iter={iteration},lr={learning_rate:.8g},replay={len(replay_inputs)})",
            retries=int(self.args.stage_retries),
            dry_run=bool(self.args.dry_run),
            prepare_attempt=prepare_train_attempt,
        )
        if bool(self.args.dry_run):
            return {}, {}, ""
        if not self.candidate_checkpoint.is_file() or not self.optimizer_work.is_file():
            raise RuntimeError(f"train artifacts are missing for iteration {iteration}")
        row = dict(audit_train_metrics(metrics_path, require_optimizer_loaded=had_optimizer))
        row["orchestrator_iteration"] = int(iteration)
        row["orchestrator_primary_input"] = str(primary)
        row["orchestrator_replay_inputs"] = [str(path) for path in replay_inputs]
        row["orchestrator_optimizer_continuity_required"] = bool(had_optimizer)
        checkpoint_audit = audit_checkpoint(self.candidate_checkpoint)
        optimizer_sha256 = sha256_file(self.optimizer_work)
        self._prepare_commit_rollback()
        try:
            os.replace(self.candidate_checkpoint, self.current_checkpoint)
            os.replace(self.optimizer_work, self.optimizer_checkpoint)
        except Exception:
            self._recover_interrupted_commit()
            raise
        return row, checkpoint_audit, optimizer_sha256

    def _run_eval(
        self,
        *,
        challenger: Path,
        opponent: Optional[Path],
        games: int,
        seed: int,
        name: str,
        output: Path,
        temperature: float,
        sample_moves: bool,
    ) -> Dict[str, Any]:
        command = [
            self.python,
            "scripts/eval_checkpoint.py",
            "--challenger_checkpoint",
            str(challenger),
            "--backend",
            "portable",
            "--device",
            str(self.args.device),
            "--eval_workers",
            "1",
            "--mcts_simulations",
            str(int(self.args.eval_simulations)),
            "--temperature",
            str(float(temperature)),
            "--v1_concurrent_games",
            str(int(self.args.eval_concurrency)),
            "--v1_opening_random_moves",
            "0",
            "--seed",
            str(int(seed)),
            "--match_name",
            str(name),
            "--output_json",
            str(output),
        ]
        if opponent is None:
            command.extend(["--eval_games_vs_random", str(int(games))])
        else:
            command.extend(
                [
                    "--previous_checkpoint",
                    str(opponent),
                    "--eval_games_vs_previous",
                    str(int(games)),
                ]
            )
        if sample_moves:
            command.append("--sample_moves")
        run_command(
            command,
            label=f"eval({name},games={games},seed={seed})",
            retries=int(self.args.stage_retries),
            dry_run=bool(self.args.dry_run),
            prepare_attempt=lambda: output.unlink(missing_ok=True),
        )
        if bool(self.args.dry_run):
            return {}
        row = _eval_result(output, name)
        if int(row.get("total_games", 0) or 0) != int(games):
            raise RuntimeError(f"evaluation game count mismatch in {output}")
        if int(row.get("seed", -1)) != int(seed):
            raise RuntimeError(f"evaluation seed mismatch in {output}")
        colors = row.get("color_breakdown")
        if not isinstance(colors, dict) or sum(
            int(value.get("games", 0) or 0)
            for value in colors.values()
            if isinstance(value, dict)
        ) != int(games):
            raise RuntimeError(f"evaluation color breakdown is missing/inconsistent: {output}")
        return row

    def evaluate_current(self, iteration: int, *, label: str = "periodic") -> Dict[str, Any]:
        if not self.current_checkpoint.is_file():
            raise RuntimeError("cannot evaluate without current.pt")
        random_seed = int(self.args.seed) + 1_000_000 + (iteration * 2)
        random_output = self.eval_dir / f"{label}_random_iter_{iteration:06d}.json"
        random_row = self._run_eval(
            challenger=self.current_checkpoint,
            opponent=None,
            games=int(self.args.eval_games_random),
            seed=random_seed,
            name="vs_random",
            output=random_output,
            temperature=0.0,
            sample_moves=False,
        )
        if bool(self.args.dry_run):
            return {}

        low, high = wilson_interval(
            int(random_row["wins"]), int(random_row["total_games"])
        )
        random_row = dict(random_row)
        random_row["wilson_95"] = [low, high]
        new_rank = random_result_rank(random_row)
        previous_rank_value = self.state.get("best_random_rank")
        previous_rank = (
            tuple(int(x) for x in previous_rank_value)
            if isinstance(previous_rank_value, list)
            else None
        )
        if previous_rank is None or new_rank > previous_rank:
            atomic_copy(self.current_checkpoint, self.best_random_checkpoint)
            self.state["best_random_rank"] = list(new_rank)
            self.state["best_random_result"] = random_row
            self.state["best_random_iteration"] = iteration
            self.state["best_random_sha256"] = sha256_file(self.best_random_checkpoint)
            self._event("best_vs_random_updated", iteration=iteration, result=random_row)
            log(
                f"best_vs_random updated iter={iteration} "
                f"W-L-D={random_row['wins']}-{random_row['losses']}-{random_row['draws']}"
            )

        best_row: Optional[Dict[str, Any]] = None
        if not self.best_checkpoint.is_file():
            atomic_copy(self.current_checkpoint, self.best_checkpoint)
            self.state["best_iteration"] = iteration
            self.state["best_sha256"] = sha256_file(self.best_checkpoint)
            self._event("best_model_bootstrapped", iteration=iteration)
            log(f"best_model bootstrapped iter={iteration}")
        else:
            best_seed = random_seed + 1
            best_output = self.eval_dir / f"{label}_best_iter_{iteration:06d}.json"
            best_row = self._run_eval(
                challenger=self.current_checkpoint,
                opponent=self.best_checkpoint,
                games=int(self.args.eval_games_best),
                seed=best_seed,
                name="vs_best",
                output=best_output,
                temperature=1.0,
                sample_moves=True,
            )
            if candidate_beats_incumbent(
                best_row, float(self.args.incumbent_promotion_score)
            ):
                atomic_copy(self.current_checkpoint, self.best_checkpoint)
                self.state["best_iteration"] = iteration
                self.state["best_sha256"] = sha256_file(self.best_checkpoint)
                self.state["best_gate_result"] = best_row
                self._event("best_model_promoted", iteration=iteration, result=best_row)
                log(
                    f"best_model promoted iter={iteration} "
                    f"W-L-D={best_row['wins']}-{best_row['losses']}-{best_row['draws']} "
                    f"score={(int(best_row['wins']) + 0.5 * int(best_row['draws'])) / int(best_row['total_games']):.4f} "
                    f"threshold={float(self.args.incumbent_promotion_score):.4f}"
                )
            else:
                self._event("best_model_retained", iteration=iteration, result=best_row)
                log(
                    f"best_model retained iter={iteration} "
                    f"candidate_W-L-D={best_row['wins']}-{best_row['losses']}-{best_row['draws']} "
                    f"score={(int(best_row['wins']) + 0.5 * int(best_row['draws'])) / int(best_row['total_games']):.4f} "
                    f"threshold={float(self.args.incumbent_promotion_score):.4f}"
                )

        self.state["last_eval_iteration"] = iteration
        self.state["last_random_result"] = random_row
        self.state["last_best_result"] = best_row
        self._save_state()
        return random_row

    def confirm_target(self, iteration: int) -> bool:
        if bool(self.state.get("target_reached", False)):
            return True
        if not target_requires_confirmation(self.state, float(self.args.target_win_rate)):
            return False
        seed = int(self.args.seed) + 10_000_000 + iteration
        output = self.eval_dir / f"target_confirm_iter_{iteration:06d}.json"
        row = self._run_eval(
            challenger=self.best_random_checkpoint,
            opponent=None,
            games=int(self.args.eval_games_random),
            seed=seed,
            name="target_confirm_vs_random",
            output=output,
            temperature=0.0,
            sample_moves=False,
        )
        if bool(self.args.dry_run):
            return False
        low, high = wilson_interval(int(row["wins"]), int(row["total_games"]))
        row = dict(row)
        row["wilson_95"] = [low, high]
        confirmed_rate = int(row["wins"]) / max(1, int(row["total_games"]))
        confirmed = confirmed_rate >= float(self.args.target_win_rate)
        self.state["target_reached"] = confirmed
        self.state["target_confirmed_result"] = row
        self.state["target_confirmed_iteration"] = iteration if confirmed else None
        self._event("target_confirmation", iteration=iteration, confirmed=confirmed, result=row)
        self._save_state()
        log(
            f"target confirmation iter={iteration} confirmed={confirmed} "
            f"W-L-D={row['wins']}-{row['losses']}-{row['draws']}"
        )
        return confirmed

    def _final_evaluation(self) -> Optional[Dict[str, Any]]:
        if not bool(self.args.final_eval) or not self.best_random_checkpoint.is_file():
            return None
        iteration = int(self.state.get("iteration", 0))
        seed = int(self.args.seed) + 20_000_000 + iteration
        output = self.eval_dir / f"final_best_random_iter_{iteration:06d}.json"
        row = self._run_eval(
            challenger=self.best_random_checkpoint,
            opponent=None,
            games=int(self.args.final_eval_games),
            seed=seed,
            name="final_best_vs_random",
            output=output,
            temperature=0.0,
            sample_moves=False,
        )
        if bool(self.args.dry_run):
            return None
        low, high = wilson_interval(int(row["wins"]), int(row["total_games"]))
        row = dict(row)
        row["wilson_95"] = [low, high]
        self.state["final_eval_result"] = row
        self._event("final_evaluation", iteration=iteration, result=row)
        self._save_state()
        return row

    def run(self) -> int:
        if bool(self.args.preflight_only):
            log("preflight-only complete")
            return 0
        self.initialize()

        if (
            bool(self.args.eval_at_start)
            and self.current_checkpoint.is_file()
            and self.state.get("last_eval_iteration") is None
        ):
            self.evaluate_current(0, label="initial")
            if self.confirm_target(0) and bool(self.args.stop_on_target):
                self._set_stop_reason("target_confirmed_at_start")

        while not self.state.get("stop_reason"):
            now = int(time.time())
            iteration = int(self.state.get("iteration", 0))
            if now >= int(self.state["deadline_epoch"]):
                self._set_stop_reason("wall_clock_deadline")
                break
            if int(self.args.max_iterations) > 0 and iteration >= int(self.args.max_iterations):
                self._set_stop_reason("max_iterations")
                break

            next_iteration = iteration + 1
            progress = self._progress()
            log(
                f"iteration={next_iteration} progress={progress:.4f} "
                f"remaining_sec={max(0, int(self.state['deadline_epoch']) - now)}"
            )
            primary, selfplay_stats = self._run_selfplay(next_iteration, progress)
            train_row, checkpoint_audit, optimizer_sha256 = self._run_train(
                next_iteration, progress, primary
            )
            if bool(self.args.dry_run):
                self._set_stop_reason("dry_run")
                break

            self.state["iteration"] = next_iteration
            self.state["latest_checkpoint_sha256"] = checkpoint_audit["sha256"]
            self.state["latest_optimizer_sha256"] = optimizer_sha256
            retained_checkpoint = self._preserve_iteration_checkpoint(next_iteration)
            self.state["last_selfplay_stats"] = selfplay_stats
            self.state["last_train_metrics"] = train_row
            self._save_state()
            self._clear_commit_rollback()
            self._prune_replay()
            self._event(
                "iteration_complete",
                iteration=next_iteration,
                selfplay_stats=selfplay_stats,
                train_metrics=train_row,
                checkpoint_audit=checkpoint_audit,
                retained_checkpoint=str(retained_checkpoint),
            )

            should_eval = (
                next_iteration % max(1, int(self.args.eval_every)) == 0
                or int(time.time()) >= int(self.state["deadline_epoch"])
            )
            if should_eval:
                self.evaluate_current(next_iteration)
                if self.confirm_target(next_iteration) and bool(self.args.stop_on_target):
                    self._set_stop_reason("target_confirmed")
                    break

        if not bool(self.args.dry_run):
            final_row = self._final_evaluation()
            self.state["ended_utc"] = utc_now()
            self.state["elapsed_sec"] = int(time.time()) - int(self.state["start_epoch"])
            self.state["final_eval_result"] = final_row
            self._save_state()
            summary = {
                "run_dir": str(self.run_dir),
                "stop_reason": self.state.get("stop_reason"),
                "iteration": int(self.state.get("iteration", 0)),
                "elapsed_sec": int(self.state.get("elapsed_sec", 0)),
                "target_win_rate": float(self.args.target_win_rate),
                "target_reached": bool(self.state.get("target_reached", False)),
                "best_model": str(self.best_checkpoint) if self.best_checkpoint.exists() else None,
                "best_vs_random": (
                    str(self.best_random_checkpoint)
                    if self.best_random_checkpoint.exists()
                    else None
                ),
                "best_random_result": self.state.get("best_random_result"),
                "latest_iteration_checkpoint": self.state.get(
                    "latest_iteration_checkpoint"
                ),
                "retained_iteration_checkpoints": len(
                    list(self.checkpoint_dir.glob("model_iter_*.pt"))
                ),
                "target_confirmed_result": self.state.get("target_confirmed_result"),
                "final_eval_result": final_row,
                "state": str(self.state_path),
                "events": str(self.events_path),
            }
            atomic_write_json(self.summary_path, summary)
            log(
                f"run ended stop_reason={summary['stop_reason']} "
                f"iterations={summary['iteration']} elapsed_sec={summary['elapsed_sec']} "
                f"target_reached={summary['target_reached']} summary={self.summary_path}"
            )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable wall-clock portable MPS training with random and incumbent gating."
    )
    parser.add_argument("--run-dir", default="tmp/v1_portable_long_20h")
    parser.add_argument("--hours", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--initial-optimizer-state", default=None)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--require-ac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-external-display", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=0)

    parser.add_argument("--self-play-games", type=int, default=128)
    parser.add_argument("--self-play-concurrency", type=int, default=128)
    parser.add_argument("--portable-self-play-workers", type=int, default=1)
    parser.add_argument(
        "--portable-mcts-backend",
        choices=["python", "cpp"],
        default="python",
    )
    parser.add_argument("--portable-cpp-threads", type=int, default=1)
    parser.add_argument("--mcts-simulations", type=int, default=8)
    parser.add_argument("--temperature-init", type=float, default=1.0)
    parser.add_argument("--temperature-final", type=float, default=0.1)
    parser.add_argument("--temperature-threshold", type=int, default=10)
    parser.add_argument("--opening-random-start", type=int, default=6)
    parser.add_argument("--opening-random-final", type=int, default=0)
    parser.add_argument("--max-game-plies", type=int, default=512)
    parser.add_argument("--soft-label-alpha", type=float, default=0.5)
    parser.add_argument("--policy-draw-weight", type=float, default=1.0)
    parser.add_argument("--replay-window", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-start", type=float, default=3e-4)
    parser.add_argument("--lr-final", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games-random", type=int, default=500)
    parser.add_argument("--eval-games-best", type=int, default=500)
    parser.add_argument("--eval-simulations", type=int, default=8)
    parser.add_argument("--eval-concurrency", type=int, default=64)
    parser.add_argument("--eval-at-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--incumbent-promotion-score", type=float, default=0.55)
    parser.add_argument("--target-win-rate", type=float, default=0.99)
    parser.add_argument("--stop-on-target", action="store_true")
    parser.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-eval-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(ROOT_DIR)
    _preflight(args)
    run_dir = _absolute_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "run.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process already owns {lock_path}") from exc
        return PortableLongTrainer(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
