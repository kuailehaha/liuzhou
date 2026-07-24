from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.long_train_portable_mps import (
    PortableLongTrainer,
    _config_signature,
    _launchd_spawn_type,
    _preflight,
    audit_train_metrics,
    build_parser,
    candidate_beats_incumbent,
    cosine_schedule,
    curriculum_progress,
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
    assert curriculum_progress(0.0, 0.25) == 0.0
    assert curriculum_progress(0.125, 0.25) == pytest.approx(0.5)
    assert curriculum_progress(0.25, 0.25) == 1.0
    assert curriculum_progress(1.0, 0.25) == 1.0


def test_m5_long_run_defaults_are_frozen() -> None:
    args = build_parser().parse_args([])
    assert args.hours == 20.0
    assert (args.self_play_games, args.self_play_concurrency) == (128, 128)
    assert args.portable_self_play_workers == 1
    assert args.portable_mcts_backend == "python"
    assert args.portable_cpp_threads == 1
    assert args.checkpoint_retain_every == 10
    assert args.policy_target_temperature is None
    assert args.policy_target_prior_pseudocount == 0.0
    assert args.opening_random_anneal_fraction == 1.0
    assert args.initial_iteration == 0
    assert (args.batch_size, args.epochs, args.replay_window) == (256, 3, 4)
    assert (args.eval_games_random, args.eval_games_best) == (500, 500)
    assert args.final_eval_games == 500
    assert args.eval_concurrency == 64
    assert args.incumbent_promotion_score == pytest.approx(0.55)
    assert args.target_win_rate == pytest.approx(0.99)


def test_direct_script_execution_adds_repository_root_to_sys_path(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "long_train_portable_mps.py"
    filtered = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() not in {root, script.parent}
    ]
    monkeypatch.setattr(sys, "path", filtered)

    runpy.run_path(str(script), run_name="_long_train_import_test")

    assert str(root) in sys.path


def test_launchd_spawn_type_is_detected_from_active_service(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.liuzhou.test")
    monkeypatch.setattr(
        "scripts.long_train_portable_mps.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="spawn type = background (5)\n",
        ),
    )

    assert _launchd_spawn_type() == "background"


def test_preflight_rejects_background_launch_agent(monkeypatch) -> None:
    args = build_parser().parse_args(["--device", "cpu", "--no-require-ac"])
    monkeypatch.setattr(
        "scripts.long_train_portable_mps.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        "scripts.long_train_portable_mps._launchd_spawn_type",
        lambda: "background",
    )

    with pytest.raises(RuntimeError, match="ProcessType=Background"):
        _preflight(args)


def test_long_run_cpp_evaluation_is_explicit_and_audited(
    monkeypatch,
    tmp_path,
) -> None:
    trainer = PortableLongTrainer.__new__(PortableLongTrainer)
    trainer.python = sys.executable
    trainer.args = SimpleNamespace(
        device="mps",
        eval_simulations=8,
        eval_concurrency=64,
        portable_mcts_backend="cpp",
        portable_cpp_threads=4,
        stage_retries=0,
        dry_run=False,
    )
    output = tmp_path / "eval.json"
    commands = []

    def fake_run_command(command, **_kwargs) -> None:
        commands.append(command)
        output.write_text(
            json.dumps(
                {
                    "portable_mcts_backend": "cpp",
                    "portable_cpp_threads": 4,
                    "portable_mcts_audit": {
                        "fallback_count": 0,
                        "illegal_action_count": 0,
                        "non_finite_count": 0,
                    },
                    "results": [
                        {
                            "name": "vs_random",
                            "wins": 2,
                            "losses": 0,
                            "draws": 0,
                            "total_games": 2,
                            "seed": 123,
                            "color_breakdown": {
                                "challenger_black": {"games": 1},
                                "challenger_white": {"games": 1},
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scripts.long_train_portable_mps.run_command",
        fake_run_command,
    )

    result = trainer._run_eval(
        challenger=tmp_path / "candidate.pt",
        opponent=None,
        games=2,
        seed=123,
        name="vs_random",
        output=output,
        temperature=0.0,
        sample_moves=False,
    )

    assert result["wins"] == 2
    assert commands
    assert commands[0][commands[0].index("--portable_mcts_backend") + 1] == "cpp"
    assert commands[0][commands[0].index("--portable_cpp_threads") + 1] == "4"


def test_iteration_checkpoint_retention_is_periodic() -> None:
    trainer = PortableLongTrainer.__new__(PortableLongTrainer)
    trainer.args = SimpleNamespace(checkpoint_retain_every=10)

    assert trainer._should_retain_iteration_checkpoint(0)
    assert not trainer._should_retain_iteration_checkpoint(1)
    assert not trainer._should_retain_iteration_checkpoint(9)
    assert trainer._should_retain_iteration_checkpoint(10)
    assert trainer._should_retain_iteration_checkpoint(20)


def test_fork_run_preserves_deadline_optimizer_replay_and_incumbent(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "phase2"
    (source / "checkpoints").mkdir(parents=True)
    (source / "replay").mkdir()
    current = source / "checkpoints" / "current.pt"
    optimizer = source / "checkpoints" / "optimizer.pt"
    best = source / "checkpoints" / "best_model.pt"
    best_random = source / "checkpoints" / "best_vs_random.pt"
    current.write_bytes(b"current-model")
    best.write_bytes(b"incumbent-model")
    best_random.write_bytes(b"random-model")
    torch_optimizer = {
        "state": {},
        "param_groups": [{"params": [], "lr": 1.8e-4, "initial_lr": 1.8e-4}],
    }
    import torch

    torch.save(torch_optimizer, optimizer)
    for iteration in range(847, 851):
        replay = source / "replay" / f"selfplay_iter_{iteration:06d}.pt"
        torch.save(
            {
                "metadata": {
                    "self_play_opening_random_moves": 3,
                }
            },
            replay,
        )
        (source / "replay" / f"selfplay_iter_{iteration:06d}.w00.chunk00000.pt").write_bytes(
            f"chunk-{iteration}".encode()
        )
    deadline = int(time.time()) + 3600
    source_state = {
        "schema_version": 1,
        "start_epoch": int(time.time()) - 3600,
        "deadline_epoch": deadline,
        "iteration": 850,
        "last_eval_iteration": 850,
        "stop_reason": "max_iterations",
        "latest_checkpoint_sha256": sha256_file(current),
        "latest_optimizer_sha256": sha256_file(optimizer),
        "best_iteration": 600,
        "best_sha256": sha256_file(best),
        "best_random_iteration": 840,
        "best_random_sha256": sha256_file(best_random),
        "best_random_rank": [450, 0],
        "best_random_result": {"wins": 450, "losses": 0, "draws": 50},
        "target_reached": False,
        "config": _config_signature(build_parser().parse_args([])),
    }
    (source / "state.json").write_text(
        json.dumps(source_state),
        encoding="utf-8",
    )
    (source / "run.lock").touch()

    args = build_parser().parse_args(
        [
            "--run-dir",
            str(destination),
            "--fork-from-run",
            str(source),
            "--device",
            "cpu",
            "--no-require-ac",
            "--mcts-simulations",
            "16",
            "--policy-target-temperature",
            "1.0",
            "--policy-target-prior-pseudocount",
            "1.0",
            "--opening-random-anneal-fraction",
            "0.25",
        ]
    )
    trainer = PortableLongTrainer(args)
    trainer.initialize()

    assert args.initial_iteration == 850
    assert args.lr_start == pytest.approx(1.8e-4)
    assert args.opening_random_start == 3
    assert trainer.state["deadline_epoch"] == deadline
    assert trainer.state["iteration"] == 850
    assert trainer.state["fork"]["parent_iteration"] == 850
    assert trainer.state["fork"]["parent_checkpoint_sha256"] == sha256_file(current)
    assert trainer.state["best_iteration"] == 600
    assert trainer.current_checkpoint.read_bytes() == b"current-model"
    assert trainer.best_checkpoint.read_bytes() == b"incumbent-model"
    assert (trainer.checkpoint_dir / "phase_start_model.pt").read_bytes() == b"current-model"
    assert len(trainer._replay_files()) == 4
    assert len(list(trainer.replay_dir.glob("*.chunk*.pt"))) == 4


def test_fork_refuses_elapsed_parent_deadline(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "phase2"
    source.mkdir()
    (source / "run.lock").touch()
    (source / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 850,
                "last_eval_iteration": 850,
                "deadline_epoch": int(time.time()) - 1,
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(destination),
            "--fork-from-run",
            str(source),
            "--device",
            "cpu",
            "--no-require-ac",
        ]
    )

    with pytest.raises(RuntimeError, match="deadline has already elapsed"):
        PortableLongTrainer(args).initialize()


def test_fork_can_reset_elapsed_deadline_with_explicit_authorization(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "phase2"
    (source / "checkpoints").mkdir(parents=True)
    (source / "replay").mkdir()
    current = source / "checkpoints" / "current.pt"
    optimizer = source / "checkpoints" / "optimizer.pt"
    current.write_bytes(b"current-model")
    import torch

    torch.save(
        {
            "state": {},
            "param_groups": [{"params": [], "lr": 1.8e-4, "initial_lr": 1.8e-4}],
        },
        optimizer,
    )
    for iteration in range(847, 851):
        torch.save(
            {"metadata": {"self_play_opening_random_moves": 3}},
            source / "replay" / f"selfplay_iter_{iteration:06d}.pt",
        )
    parent_deadline = int(time.time()) - 60
    source_state = {
        "schema_version": 1,
        "deadline_epoch": parent_deadline,
        "iteration": 850,
        "last_eval_iteration": 850,
        "stop_reason": "max_iterations",
        "latest_checkpoint_sha256": sha256_file(current),
        "latest_optimizer_sha256": sha256_file(optimizer),
        "config": _config_signature(build_parser().parse_args([])),
    }
    (source / "state.json").write_text(json.dumps(source_state), encoding="utf-8")
    (source / "run.lock").touch()
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(destination),
            "--fork-from-run",
            str(source),
            "--reset-fork-deadline",
            "--hours",
            "40",
            "--device",
            "cpu",
            "--no-require-ac",
        ]
    )

    before = int(time.time())
    trainer = PortableLongTrainer(args)
    trainer.initialize()

    assert trainer.state["deadline_epoch"] == pytest.approx(
        before + 40 * 3600,
        abs=2,
    )
    assert trainer.state["fork"]["original_deadline_epoch"] == parent_deadline
    assert trainer.state["fork"]["deadline_reset_authorized"] is True
    assert trainer.state["fork"]["phase_hours"] == 40.0


def test_best_promotion_saves_matching_optimizer(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--no-require-ac",
        ]
    )
    trainer = PortableLongTrainer(args)
    trainer.checkpoint_dir.mkdir(parents=True)
    trainer.current_checkpoint.write_bytes(b"candidate")
    trainer.optimizer_checkpoint.write_bytes(b"candidate-optimizer")
    trainer.state = {"schema_version": 1}

    gate = {"wins": 3, "losses": 1, "draws": 0}
    trainer._promote_best_pair(iteration=12, gate_result=gate)

    assert trainer.best_checkpoint.read_bytes() == b"candidate"
    assert trainer.best_optimizer_checkpoint.read_bytes() == b"candidate-optimizer"
    assert trainer.state["best_iteration"] == 12
    assert trainer.state["best_gate_result"] == gate
    assert trainer.state["best_sha256"] == sha256_file(trainer.best_checkpoint)
    assert trainer.state["best_optimizer_sha256"] == sha256_file(
        trainer.best_optimizer_checkpoint
    )
    assert not trainer.best_pair_transaction.exists()
    assert not trainer.rollback_best_model.exists()
    assert not trainer.rollback_best_optimizer.exists()


def test_partial_best_pair_commit_rolls_back_both_artifacts(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--no-require-ac",
        ]
    )
    trainer = PortableLongTrainer(args)
    trainer.checkpoint_dir.mkdir(parents=True)
    trainer.current_checkpoint.write_bytes(b"new-model")
    trainer.optimizer_checkpoint.write_bytes(b"new-optimizer")
    trainer.best_checkpoint.write_bytes(b"old-model")
    trainer.best_optimizer_checkpoint.write_bytes(b"old-optimizer")
    trainer.state = {
        "schema_version": 1,
        "best_iteration": 10,
        "best_sha256": sha256_file(trainer.best_checkpoint),
        "best_optimizer_sha256": sha256_file(
            trainer.best_optimizer_checkpoint
        ),
    }
    trainer._prepare_best_pair_transaction(
        iteration=20,
        gate_result={"wins": 3, "losses": 1, "draws": 0},
    )
    os.replace(trainer.best_model_next, trainer.best_checkpoint)

    assert trainer._recover_best_pair_transaction()
    assert trainer.best_checkpoint.read_bytes() == b"old-model"
    assert trainer.best_optimizer_checkpoint.read_bytes() == b"old-optimizer"
    assert trainer.state["best_iteration"] == 10
    assert not trainer.best_pair_transaction.exists()


def test_complete_best_pair_commit_reconciles_state_after_crash(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--no-require-ac",
        ]
    )
    trainer = PortableLongTrainer(args)
    trainer.checkpoint_dir.mkdir(parents=True)
    trainer.current_checkpoint.write_bytes(b"new-model")
    trainer.optimizer_checkpoint.write_bytes(b"new-optimizer")
    trainer.best_checkpoint.write_bytes(b"old-model")
    trainer.best_optimizer_checkpoint.write_bytes(b"old-optimizer")
    trainer.state = {
        "schema_version": 1,
        "best_iteration": 10,
        "best_sha256": sha256_file(trainer.best_checkpoint),
        "best_optimizer_sha256": sha256_file(
            trainer.best_optimizer_checkpoint
        ),
    }
    gate = {"wins": 3, "losses": 1, "draws": 0}
    transaction = trainer._prepare_best_pair_transaction(
        iteration=20,
        gate_result=gate,
    )
    os.replace(trainer.best_model_next, trainer.best_checkpoint)
    os.replace(
        trainer.best_optimizer_next,
        trainer.best_optimizer_checkpoint,
    )

    assert trainer._recover_best_pair_transaction()
    assert trainer.best_checkpoint.read_bytes() == b"new-model"
    assert trainer.best_optimizer_checkpoint.read_bytes() == b"new-optimizer"
    assert trainer.state["best_iteration"] == 20
    assert trainer.state["best_gate_result"] == gate
    assert trainer.state["best_sha256"] == transaction["new_model_sha256"]
    assert (
        trainer.state["best_optimizer_sha256"]
        == transaction["new_optimizer_sha256"]
    )
    assert not trainer.best_pair_transaction.exists()


def test_nonzero_initial_iteration_requires_initial_checkpoint(tmp_path) -> None:
    args = build_parser().parse_args(
        ["--run-dir", str(tmp_path), "--initial-iteration", "473"]
    )
    trainer = PortableLongTrainer(args)

    with pytest.raises(
        ValueError,
        match="--initial-iteration requires --initial-checkpoint",
    ):
        trainer.initialize()


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
    legacy_config.pop("portable_mcts_backend")
    legacy_config.pop("portable_cpp_threads")
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
    assert trainer.state["config"]["portable_mcts_backend"] == "python"
    assert trainer.state["config"]["portable_cpp_threads"] == 1


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
