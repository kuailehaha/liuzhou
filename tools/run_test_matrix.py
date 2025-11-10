"""
Batch runner for the v1 test matrix (accuracy, performance, smoke suites).

Usage examples:
  python -m tools.run_test_matrix                # run all groups
  python -m tools.run_test_matrix --group accuracy --group performance
  python -m tools.run_test_matrix --dry-run      # print commands only
  python -m tools.run_test_matrix --pytest-args -k fast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TestJob:
    """Configuration for a single test invocation."""

    name: str
    log_path: Path
    # kind: "pytest" or "cli"
    kind: str = "pytest"
    # For pytest jobs
    test_path: Path | None = None
    pytest_args: Sequence[str] = ()
    # For CLI jobs (python -m <module> ...)
    module: str | None = None
    cli_args: Sequence[str] = ()


TEST_GROUPS = {
    "accuracy": [
        TestJob(
            name="fast_apply_moves_accuracy",
            log_path=Path("tests/result/apply_moves_accuracy.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_fast_apply_moves.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="policy_projection_fast_accuracy",
            log_path=Path("tests/result/policy_proj_accuracy.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_policy_projection_fast_accuracy.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="policy_projection_regression",
            log_path=Path("tests/result/policy_projection_regression.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_policy_projection.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="move_encoder_accuracy",
            log_path=Path("tests/result/move_encoder_accuracy.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_move_encoder.py"),
            pytest_args=("-q",),
        ),
    ],
    "performance": [
        TestJob(
            name="fast_apply_moves_performance",
            log_path=Path("tools/result/apply_moves_perf.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_fast_apply_moves_performance.py"),
            pytest_args=("-s",),
        ),
        TestJob(
            name="policy_projection_fast_performance",
            log_path=Path("tools/result/policy_proj_perf.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_policy_projection_fast_performance.py"),
            pytest_args=("-s",),
        ),
        TestJob(
            name="legal_mask_benchmark_cli",
            log_path=Path("tools/result/legal_mask_bench.txt"),
            kind="cli",
            module="tools.benchmark_legal_mask",
            # defaults align with TEST_README.md; can be overridden via --pytest-args passthrough not applicable to CLI
            cli_args=(
                "--states", "1000",
                "--batch-size", "64",
                "--runs", "5",
                "--max-random-moves", "80",
                "--device", "cpu",
                "--seed", "0",
            ),
        ),
    ],
    "smoke": [
        TestJob(
            name="encoding_compat_smoke",
            log_path=Path("tests/result/encoding_compat_smoke.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_encoding_compat.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="state_batch_smoke",
            log_path=Path("tests/result/state_batch_smoke.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_state_batch.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="vectorized_mcts_smoke",
            log_path=Path("tests/result/vectorized_mcts_smoke.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_vectorized_mcts.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="train_pipeline_smoke",
            log_path=Path("tests/result/train_pipeline_smoke.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_train_pipeline.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="self_play_runner_smoke",
            log_path=Path("tests/result/self_play_runner_smoke.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_self_play_runner.py"),
            pytest_args=("-q",),
        ),
        TestJob(
            name="scaffold_smoke",
            log_path=Path("tests/result/scaffold_smoke.txt"),
            kind="pytest",
            test_path=Path("tests/v1/test_scaffold.py"),
            pytest_args=("-q",),
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run curated pytest suites and capture logs.")
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(TEST_GROUPS.keys()),
        help="Test group(s) to run. Defaults to all groups when omitted.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failing test invocation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest commands without executing them.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed through to pytest (use after '--').",
    )
    return parser.parse_args()


def iter_selected_groups(selected: Iterable[str] | None) -> List[str]:
    if selected:
        return list(dict.fromkeys(selected))  # maintain order, drop duplicates
    return list(TEST_GROUPS.keys())


def run_job(job: TestJob, extra_pytest_args: Sequence[str], dry_run: bool) -> int:
    if job.kind == "cli":
        assert job.module is not None
        cmd = [sys.executable, "-m", job.module, *job.cli_args]
    else:
        assert job.test_path is not None
        cmd = [sys.executable, "-m", "pytest", str(job.test_path), *job.pytest_args, *extra_pytest_args]

    printable = " ".join(cmd)
    print(f"[run] {job.name}: {printable}")

    if dry_run:
        return 0

    process = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    log_path = ROOT / job.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")

    if process.returncode == 0:
        print(f"[pass] {job.name} (log saved to {job.log_path.as_posix()})")
    else:
        print(f"[fail] {job.name} (log saved to {job.log_path.as_posix()})", file=sys.stderr)

    return process.returncode


def main() -> int:
    args = parse_args()
    groups = iter_selected_groups(args.group)
    extra_pytest_args = list(args.pytest_args)
    failures: List[str] = []

    for group in groups:
        print(f"\n== Running group: {group} ==")
        for job in TEST_GROUPS[group]:
            code = run_job(job, extra_pytest_args, args.dry_run)
            if code != 0:
                failures.append(job.name)
                if args.fail_fast:
                    print("Fail-fast enabled; aborting remaining jobs.", file=sys.stderr)
                    break
        if args.fail_fast and failures:
            break

    if failures:
        print(
            f"\nCompleted with failures ({len(failures)}): {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1

    print("\nAll requested test groups completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

