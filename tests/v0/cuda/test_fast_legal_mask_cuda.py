"""
Accuracy + performance tests for fast legal mask (CPU vs CUDA).

This module validates that the C++/CUDA fast legal mask path (used inside
`v0_core.encode_actions_fast`) matches the CPU implementation and provides
a basic throughput comparison.

Seed / sampling conventions (from Test Matrix):
    - rng seed: 0xF00DCAFE
    - torch seed: 0xF00DCAFE
    - num_states (accuracy): 10_000
    - max_random_moves (conceptual budget): 80

Usage:
    pytest tests/v0/cuda/test_fast_legal_mask_cuda.py -q
    pytest tests/v0/cuda/test_fast_legal_mask_cuda.py -m slow -s  # include performance
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Tuple

import pytest
import torch

# ---------------------------------------------------------------------------
# Global conventions from the Test Matrix
# ---------------------------------------------------------------------------

SEED: int = 0xF00DCAFE
NUM_STATES: int = 10_000
MAX_RANDOM_MOVES: int = 80

random_rng = random.Random(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Import v0_core (C++/CUDA extension)
# ---------------------------------------------------------------------------

try:
    import v0_core
except ImportError:
    v0_core = None

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_RESULT_DIR = Path(__file__).resolve().parents[1] / "result"
_ACCURACY_LOG = _RESULT_DIR / "fast_legal_mask_cuda_accuracy.txt"
_PERF_LOG = _RESULT_DIR / "fast_legal_mask_cuda_perf.txt"


def _append_log(path: Path, line: str) -> None:
    """Append a single line to the given log file, creating dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# State sampling for encode_actions_fast / legal mask
# ---------------------------------------------------------------------------


def _random_cpu_batch(num_states: int) -> Tuple[torch.Tensor, ...]:
    """
    Generate a batch of synthetic game states on CPU.

    Shapes / dtypes match the encode_actions_fast binding:
        - board:                 (N, 6, 6), int8 in {-1, 0, 1}
        - marks_black/marks_white:(N, 6, 6), bool
        - phase:                 (N,), int64 in [1, 7]
        - current_player:        (N,), int64 in {-1, 1}
        - pending_* / forced_*:  (N,), int64 with small non-negative values
    """
    device = torch.device("cpu")

    board = torch.randint(-1, 2, (num_states, 6, 6), dtype=torch.int8, device=device)
    marks_black = torch.randint(0, 2, (num_states, 6, 6), dtype=torch.bool, device=device)
    marks_white = torch.randint(0, 2, (num_states, 6, 6), dtype=torch.bool, device=device)
    phase = torch.randint(1, 8, (num_states,), dtype=torch.int64, device=device)

    current_player = torch.randint(0, 2, (num_states,), dtype=torch.int64, device=device)
    current_player = current_player.mul(-2).add(1)  # map {0,1} -> {-1,1}

    pending_marks_required = torch.zeros(num_states, dtype=torch.int64, device=device)
    pending_marks_remaining = torch.randint(
        0, 3, (num_states,), dtype=torch.int64, device=device
    )
    pending_captures_required = torch.zeros(num_states, dtype=torch.int64, device=device)
    pending_captures_remaining = torch.randint(
        0, 3, (num_states,), dtype=torch.int64, device=device
    )
    forced_removals_done = torch.randint(
        0, 3, (num_states,), dtype=torch.int64, device=device
    )

    return (
        board,
        marks_black,
        marks_white,
        phase,
        current_player,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
    )


def _move_batch_to_device(
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in batch)


def _run_encode_actions_fast(
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around v0_core.encode_actions_fast for a given device.

    This is the entry point that ultimately exercises the fast legal mask
    kernels on CPU and CUDA.
    """
    (
        board,
        marks_black,
        marks_white,
        phase,
        current_player,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
    ) = _move_batch_to_device(batch, device)

    placement_dim = board.size(-1) * board.size(-2)
    movement_dim = placement_dim * 4
    selection_dim = placement_dim
    auxiliary_dim = 1

    return v0_core.encode_actions_fast(
        board,
        marks_black,
        marks_white,
        phase,
        current_player,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
        placement_dim,
        movement_dim,
        selection_dim,
        auxiliary_dim,
    )


# ---------------------------------------------------------------------------
# Accuracy: CPU vs CUDA parity for legal mask via encode_actions_fast
# ---------------------------------------------------------------------------


def test_fast_legal_mask_cuda_matches_cpu_accuracy() -> None:
    """Deterministic parity check between CPU and CUDA fast legal mask paths."""
    if v0_core is None:
        _append_log(
            _ACCURACY_LOG,
            "SKIP: v0_core extension could not be imported; accuracy test not run.",
        )
        pytest.skip("v0_core extension not importable")

    if not torch.cuda.is_available():
        _append_log(
            _ACCURACY_LOG,
            "SKIP: CUDA device not available; accuracy comparison limited to CPU.",
        )
        pytest.skip("CUDA not available")

    device_cpu = torch.device("cpu")
    device_cuda = torch.device("cuda")

    batch_cpu = _random_cpu_batch(NUM_STATES)

    # CPU path
    mask_cpu, meta_cpu = _run_encode_actions_fast(batch_cpu, device_cpu)

    # CUDA path (may throw if kernels weren't compiled)
    try:
        mask_cuda, meta_cuda = _run_encode_actions_fast(batch_cpu, device_cuda)
    except RuntimeError as exc:
        msg = str(exc)
        if "CUDA kernels were not built" in msg or "CUDA kernels were not compiled" in msg:
            _append_log(
                _ACCURACY_LOG,
                "SKIP: CUDA kernels for fast legal mask were not compiled; "
                "accuracy test fell back to CPU only.",
            )
            pytest.skip("CUDA kernels were not compiled for fast legal mask")
        raise

    # Parity checks
    assert mask_cpu.shape == mask_cuda.shape
    assert meta_cpu.shape == meta_cuda.shape

    assert torch.equal(mask_cpu, mask_cuda.cpu())
    assert torch.equal(meta_cpu, meta_cuda.cpu())

    _append_log(
        _ACCURACY_LOG,
        f"OK: fast_legal_mask CPU vs CUDA parity passed "
        f"(num_states={NUM_STATES}, seed=0x{SEED:X}).",
    )


# ---------------------------------------------------------------------------
# Performance: basic throughput comparison (CPU vs CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fast_legal_mask_cuda_performance_smoke() -> None:
    """
    Smoke performance benchmark for CPU vs CUDA fast legal mask.

    This is a *micro-benchmark* intended for local runs, not strict CI gating.
    """
    if v0_core is None:
        _append_log(
            _PERF_LOG,
            "SKIP: v0_core extension could not be imported; performance test not run.",
        )
        pytest.skip("v0_core extension not importable")

    if not torch.cuda.is_available():
        _append_log(
            _PERF_LOG,
            "SKIP: CUDA device not available; performance comparison limited to CPU.",
        )
        pytest.skip("CUDA not available")

    device_cpu = torch.device("cpu")
    device_cuda = torch.device("cuda")

    batch_cpu = _random_cpu_batch(NUM_STATES)

    runs = 100
    cpu_times_ms = []
    cuda_times_ms = []

    # Warmup
    _ = _run_encode_actions_fast(batch_cpu, device_cpu)
    _ = _run_encode_actions_fast(batch_cpu, device_cuda)
    torch.cuda.synchronize()

    # Benchmark CPU
    for _i in range(runs):
        start = time.perf_counter()
        _ = _run_encode_actions_fast(batch_cpu, device_cpu)
        end = time.perf_counter()
        cpu_times_ms.append((end - start) * 1e3)

    # Benchmark CUDA
    for _i in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _run_encode_actions_fast(batch_cpu, device_cuda)
        torch.cuda.synchronize()
        end = time.perf_counter()
        cuda_times_ms.append((end - start) * 1e3)

    avg_cpu = sum(cpu_times_ms) / len(cpu_times_ms)
    avg_cuda = sum(cuda_times_ms) / len(cuda_times_ms)

    line_header = (
        f"fast_legal_mask performance "
        f"(num_states={NUM_STATES}, runs={runs}, seed=0x{SEED:X}):"
    )
    line_summary = f"  cpu={avg_cpu:.3f} ms, cuda={avg_cuda:.3f} ms"

    print(line_header)
    for i, (c, g) in enumerate(zip(cpu_times_ms, cuda_times_ms), start=1):
        print(f"    run {i}: cpu={c:.3f} ms, cuda={g:.3f} ms")
    print(line_summary)

    _append_log(_PERF_LOG, line_header)
    for i, (c, g) in enumerate(zip(cpu_times_ms, cuda_times_ms), start=1):
        _append_log(_PERF_LOG, f"    run {i}: cpu={c:.3f} ms, cuda={g:.3f} ms")
    _append_log(_PERF_LOG, line_summary)

