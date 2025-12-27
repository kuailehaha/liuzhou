"""
Batch runner for the test matrix (legacy, v0, integration, benchmark suites).

Usage examples:
  python -m tools.run_test_matrix                # run all groups
  python -m tools.run_test_matrix --group legacy --group v0
  python -m tools.run_test_matrix --dry-run      # print commands only
  python -m tools.run_test_matrix --pytest-args -k mcts
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
    "legacy": [
        TestJob(
            name="legacy_mcts",
            log_path=Path("tests/result/legacy_mcts.txt"),
            kind="pytest",
            test_path=Path("tests/legacy/test_mcts.py"),
            pytest_args=("-v",),
        ),
        TestJob(
            name="legacy_self_play",
            log_path=Path("tests/result/legacy_self_play.txt"),
            kind="pytest",
            test_path=Path("tests/legacy/test_self_play.py"),
            pytest_args=("-v",),
        ),
    ],
    "v0": [
        TestJob(
            name="v0_actions",
            log_path=Path("tests/result/v0_actions.txt"),
            kind="pytest",
            test_path=Path("tests/v0/test_actions.py"),
            pytest_args=("-v",),
        ),
        TestJob(
            name="v0_state_batch",
            log_path=Path("tests/result/v0_state_batch.txt"),
            kind="pytest",
            test_path=Path("tests/v0/test_state_batch.py"),
            pytest_args=("-v",),
        ),
        TestJob(
            name="v0_mcts",
            log_path=Path("tests/result/v0_mcts.txt"),
            kind="pytest",
            test_path=Path("tests/v0/test_mcts.py"),
            pytest_args=("-v",),
        ),
    ],
    "v0_cuda": [
        TestJob(
            name="v0_cuda_apply_moves",
            log_path=Path("tests/result/v0_cuda_apply_moves.txt"),
            kind="pytest",
            test_path=Path("tests/v0/cuda/test_fast_apply_moves_cuda.py"),
            pytest_args=("-v",),
        ),
        TestJob(
            name="v0_cuda_legal_mask",
            log_path=Path("tests/result/v0_cuda_legal_mask.txt"),
            kind="pytest",
            test_path=Path("tests/v0/cuda/test_fast_legal_mask_cuda.py"),
            pytest_args=("-v",),
        ),
    ],
    "integration": [
        TestJob(
            name="integration_self_play",
            log_path=Path("tests/result/integration_self_play.txt"),
            kind="pytest",
            test_path=Path("tests/integration/test_self_play.py"),
            pytest_args=("-v",),
        ),
    ],
    "random_agent": [
        TestJob(
            name="random_agent",
            log_path=Path("tests/result/random_agent.txt"),
            kind="cli",
            module="tests.random_agent.run_tests",
            cli_args=("basic", "-n", "10", "-s", "42"),
        ),
    ],
    "benchmark": [
        TestJob(
            name="benchmark_mcts",
            log_path=Path("tools/result/benchmark_mcts.txt"),
            kind="cli",
            module="tools.benchmark_mcts",
            cli_args=(
                "--samples", "3",
                "--sims", "32",
                "--device", "cpu",
                "--skip-legacy",
            ),
        ),
        TestJob(
            name="benchmark_self_play",
            log_path=Path("tools/result/benchmark_self_play.txt"),
            kind="cli",
            module="tools.benchmark_self_play",
            cli_args=(
                "--num-games", "1",
                "--mcts-simulations", "16",
                "--skip-legacy",
            ),
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
