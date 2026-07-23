from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from scripts.long_train_portable_mps import (
    PortableLongTrainer,
    _config_signature,
    audit_train_metrics,
    build_parser,
    candidate_beats_incumbent,
    cosine_schedule,
    linear_int_schedule,
    random_result_rank,
    run_command,
    select_replay_inputs,
    sha256_file,
    target_requires_confirmation,
    wilson_interval,
)


def test_curriculum_schedules_are_bounded() -> None:
    assert cosine_schedule(3e-4, 5e-5, -1.0) == pytest.approx(3e-4)
    assert cosine_schedule(3e-4, 5e-5, 0.5) == pytest.approx(1.75e-4)
    assert cosine_schedule(3e-4, 5e-5, 2.0) == pytest.approx(5e-5)
    assert linear_int_schedule(6, 0, 0.0) == 6
    assert linear_int_schedule(6, 0, 0.5) == 3
    assert linear_int_schedule(6, 0, 1.0) == 0


def test_m5_long_run_defaults_are_frozen() -> None:
    args = build_parser().parse_args([])
    assert args.hours == 20.0
    assert (args.self_play_games, args.self_play_concurrency) == (128, 128)
    assert args.portable_self_play_workers == 1
    assert (args.batch_size, args.epochs, args.replay_window) == (256, 3, 4)
    assert (args.eval_games_random, args.eval_games_best) == (500, 500)
    assert args.final_eval_games == 500
    assert args.eval_concurrency == 64
    assert args.incumbent_promotion_score == pytest.approx(0.55)
    assert args.target_win_rate == pytest.approx(0.99)


def test_replay_window_zero_selects_no_history(tmp_path) -> None:
    primary = tmp_path / "selfplay_iter_000003.pt"
    files = [
        tmp_path / "selfplay_iter_000001.pt",
        tmp_path / "selfplay_iter_000002.pt",
        primary,
    ]
    assert select_replay_inputs(files, primary=primary, window=0) == []
    assert select_replay_inputs(files, primary=primary, window=1) == [files[1]]
    assert select_replay_inputs(files, primary=primary, window=2) == files[:2]


def test_replay_discovery_and_pruning_handle_worker_chunks(tmp_path) -> None:
    trainer = PortableLongTrainer.__new__(PortableLongTrainer)
    trainer.replay_dir = tmp_path
    trainer.args = SimpleNamespace(replay_window=1)
    manifests = [
        tmp_path / "selfplay_iter_000001.pt",
        tmp_path / "selfplay_iter_000002.pt",
        tmp_path / "selfplay_iter_000003.pt",
    ]
    chunks = [
        tmp_path / "selfplay_iter_000001.w00.chunk00000.pt",
        tmp_path / "selfplay_iter_000002.w00.chunk00000.pt",
        tmp_path / "selfplay_iter_000003.w00.chunk00000.pt",
    ]
    for path in manifests + chunks:
        path.write_bytes(b"payload")

    assert trainer._replay_files() == manifests
    trainer._prune_replay()

    assert not manifests[0].exists()
    assert not chunks[0].exists()
    assert all(path.exists() for path in manifests[1:] + chunks[1:])


def test_random_rank_and_incumbent_gate() -> None:
    assert random_result_rank({"wins": 496, "losses": 1, "draws": 3}) > random_result_rank(
        {"wins": 495, "losses": 0, "draws": 5}
    )
    assert random_result_rank({"wins": 495, "losses": 0, "draws": 5}) > random_result_rank(
        {"wins": 495, "losses": 1, "draws": 4}
    )
    assert candidate_beats_incumbent(
        {"wins": 200, "losses": 150, "draws": 150, "total_games": 500},
        0.55,
    )
    assert not candidate_beats_incumbent(
        {"wins": 199, "losses": 150, "draws": 151, "total_games": 500},
        0.55,
    )
    assert not candidate_beats_incumbent(
        {"wins": 1, "losses": 0, "draws": 0, "total_games": 0},
        0.55,
    )


def test_target_confirmation_runs_once_after_observed_threshold() -> None:
    state = {
        "target_reached": False,
        "best_random_result": {"wins": 495, "losses": 1, "draws": 4, "total_games": 500},
    }
    assert target_requires_confirmation(state, 0.99)

    state["target_reached"] = True
    assert not target_requires_confirmation(state, 0.99)

    state["target_reached"] = False
    state["best_random_result"]["wins"] = 494
    assert not target_requires_confirmation(state, 0.99)


def test_confirm_target_reuses_existing_confirmation_without_eval() -> None:
    trainer = PortableLongTrainer.__new__(PortableLongTrainer)
    trainer.args = SimpleNamespace(target_win_rate=0.99)
    trainer.state = {"target_reached": True}
    assert trainer.confirm_target(12)


def test_resume_can_continue_after_confirmed_target_when_stop_flag_is_removed(tmp_path) -> None:
    args = build_parser().parse_args(["--run-dir", str(tmp_path), "--resume"])
    trainer = PortableLongTrainer(args)
    trainer.checkpoint_dir.mkdir(parents=True)
    trainer.current_checkpoint.write_bytes(b"model")
    trainer.optimizer_checkpoint.write_bytes(b"optimizer")
    trainer.state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deadline_epoch": int(time.time()) + 3600,
                "iteration": 50,
                "target_reached": True,
                "stop_reason": "target_confirmed",
                "ended_utc": "2026-07-22T09:40:01Z",
                "elapsed_sec": 7502,
                "config": _config_signature(args),
                "latest_checkpoint_sha256": sha256_file(trainer.current_checkpoint),
                "latest_optimizer_sha256": sha256_file(trainer.optimizer_checkpoint),
            }
        ),
        encoding="utf-8",
    )

    trainer.initialize()

    assert trainer.state["stop_reason"] is None
    assert trainer.state["iteration"] == 50
    assert trainer.state["target_reached"] is True
    assert trainer.state["resume_count"] == 1
    assert trainer.state["last_resume_stop_reason"] == "target_confirmed"
    assert "ended_utc" not in trainer.state
    assert "elapsed_sec" not in trainer.state
    retained = trainer.checkpoint_dir / "model_iter_000050.pt"
    assert retained.read_bytes() == b"model"
    assert trainer.state["latest_iteration_checkpoint"] == str(retained)
    assert trainer.state["latest_iteration_checkpoint_sha256"] == sha256_file(retained)


def test_resume_migrates_legacy_incumbent_gate_to_explicit_default(tmp_path) -> None:
    args = build_parser().parse_args(["--run-dir", str(tmp_path), "--resume"])
    trainer = PortableLongTrainer(args)
    trainer.checkpoint_dir.mkdir(parents=True)
    trainer.current_checkpoint.write_bytes(b"model")
    trainer.optimizer_checkpoint.write_bytes(b"optimizer")
    legacy_config = _config_signature(args)
    legacy_config.pop("incumbent_promotion_score")
    legacy_config.pop("portable_self_play_workers")
    trainer.state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deadline_epoch": int(time.time()) + 3600,
                "iteration": 50,
                "stop_reason": None,
                "config": legacy_config,
                "latest_checkpoint_sha256": sha256_file(trainer.current_checkpoint),
                "latest_optimizer_sha256": sha256_file(trainer.optimizer_checkpoint),
            }
        ),
        encoding="utf-8",
    )

    trainer.initialize()

    assert trainer.state["config"]["incumbent_promotion_score"] == pytest.approx(0.55)
    assert trainer.state["config"]["portable_self_play_workers"] == 1


def test_interrupted_model_optimizer_commit_restores_previous_pair(tmp_path) -> None:
    trainer = PortableLongTrainer.__new__(PortableLongTrainer)
    trainer.current_checkpoint = tmp_path / "current.pt"
    trainer.optimizer_checkpoint = tmp_path / "optimizer.pt"
    trainer.rollback_current = tmp_path / ".rollback_current.pt"
    trainer.rollback_optimizer = tmp_path / ".rollback_optimizer.pt"
    trainer.current_checkpoint.write_bytes(b"old-model")
    trainer.optimizer_checkpoint.write_bytes(b"old-optimizer")
    trainer.state = {
        "latest_checkpoint_sha256": sha256_file(trainer.current_checkpoint),
        "latest_optimizer_sha256": sha256_file(trainer.optimizer_checkpoint),
    }

    trainer._prepare_commit_rollback()
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"new-model")
    os.replace(candidate, trainer.current_checkpoint)

    assert trainer._recover_interrupted_commit()
    assert trainer.current_checkpoint.read_bytes() == b"old-model"
    assert trainer.optimizer_checkpoint.read_bytes() == b"old-optimizer"
    assert not trainer.rollback_current.exists()
    assert not trainer.rollback_optimizer.exists()


def test_iteration_checkpoints_are_retained_without_overwrite(tmp_path) -> None:
    trainer = PortableLongTrainer.__new__(PortableLongTrainer)
    trainer.checkpoint_dir = tmp_path
    trainer.current_checkpoint = tmp_path / "current.pt"
    trainer.state = {}

    trainer.current_checkpoint.write_bytes(b"iteration-1")
    first = trainer._preserve_iteration_checkpoint(1)
    trainer.current_checkpoint.write_bytes(b"iteration-2")
    second = trainer._preserve_iteration_checkpoint(2)

    assert first == tmp_path / "model_iter_000001.pt"
    assert second == tmp_path / "model_iter_000002.pt"
    assert first.read_bytes() == b"iteration-1"
    assert second.read_bytes() == b"iteration-2"
    assert trainer.state["latest_iteration_checkpoint"] == str(second)
    assert trainer.state["latest_iteration_checkpoint_sha256"] == sha256_file(second)

    with pytest.raises(RuntimeError, match="refusing to overwrite retained checkpoint"):
        trainer._preserve_iteration_checkpoint(1)
    assert first.read_bytes() == b"iteration-1"


def test_stage_retry_prepares_every_attempt(monkeypatch) -> None:
    return_codes = iter([1, 0])
    attempts = []

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=next(return_codes))

    monkeypatch.setattr("scripts.long_train_portable_mps.subprocess.run", fake_run)
    monkeypatch.setattr("scripts.long_train_portable_mps.time.sleep", lambda _seconds: None)
    run_command(
        ["fake-stage"],
        label="retry-test",
        retries=1,
        dry_run=False,
        prepare_attempt=lambda: attempts.append(len(attempts) + 1),
    )
    assert attempts == [1, 2]


def test_wilson_interval_for_known_result() -> None:
    low, high = wilson_interval(859, 1000)
    assert low == pytest.approx(0.836054, abs=1e-6)
    assert high == pytest.approx(0.879199, abs=1e-6)


def test_audit_train_metrics_accepts_finite_continuation(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            [
                {
                    "train_avg_loss": 1.25,
                    "train_avg_policy_loss": 0.5,
                    "train_avg_value_loss": 0.75,
                    "optimizer_loaded": True,
                    "optimizer_load_error": None,
                    "filtered_non_finite_samples": 0,
                    "self_play_value_nonfinite_local": 0,
                    "self_play_soft_value_nonfinite_local": 0,
                    "device_fallback_count": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    row = audit_train_metrics(path, require_optimizer_loaded=True)
    assert row["train_avg_loss"] == 1.25


def test_audit_train_metrics_rejects_optimizer_reset(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            [
                {
                    "train_avg_loss": 1.0,
                    "train_avg_policy_loss": 0.5,
                    "train_avg_value_loss": 0.5,
                    "optimizer_loaded": False,
                    "optimizer_load_error": "bad state",
                    "filtered_non_finite_samples": 0,
                    "self_play_value_nonfinite_local": 0,
                    "self_play_soft_value_nonfinite_local": 0,
                    "device_fallback_count": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="optimizer continuity"):
        audit_train_metrics(path, require_optimizer_loaded=True)
