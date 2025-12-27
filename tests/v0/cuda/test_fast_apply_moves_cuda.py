"""
Accuracy + performance tests for fast apply moves (CPU vs CUDA).

This suite mirrors the legal-mask tests: it generates synthetic parent states /
encoded actions, compares the CPU and CUDA `v0_core.batch_apply_moves` outputs,
and optionally records a simple throughput benchmark.

Seed / sampling conventions:
    - rng seed: 0xA11CEB0B
    - torch seed: 0xA11CEB0B
    - num_actions (accuracy): 10000
    - action kinds covered: placement, mark/capture, forced/counter removals,
      movement, no-move removal, and process-removal.

Usage:
    pytest tests/v0/cuda/test_fast_apply_moves_cuda.py -q
    pytest tests/v0/cuda/test_fast_apply_moves_cuda.py -m slow -s  # include performance
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

# ---------------------------------------------------------------------------
# Global conventions
# ---------------------------------------------------------------------------

SEED: int = 0xA11CEB0B
NUM_ACTIONS: int = 10_000

random_rng = random.Random(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Import v0_core
# ---------------------------------------------------------------------------

try:
    import v0_core
except ImportError:
    v0_core = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_RESULT_DIR = Path(__file__).resolve().parents[1] / "result"
_ACCURACY_LOG = _RESULT_DIR / "fast_apply_moves_cuda_accuracy.txt"
_PERF_LOG = _RESULT_DIR / "fast_apply_moves_cuda_perf.txt"


def _append_log(path: Path, line: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Constants shared with the C++ layer
# ---------------------------------------------------------------------------

K_ACTION_PLACEMENT = 1
K_ACTION_MOVEMENT = 2
K_ACTION_MARK_SELECTION = 3
K_ACTION_CAPTURE_SELECTION = 4
K_ACTION_FORCED_REMOVAL = 5
K_ACTION_COUNTER_REMOVAL = 6
K_ACTION_NO_MOVES_REMOVAL = 7
K_ACTION_PROCESS_REMOVAL = 8

K_PHASE_PLACEMENT = 1
K_PHASE_MARK_SELECTION = 2
K_PHASE_REMOVAL = 3
K_PHASE_MOVEMENT = 4
K_PHASE_CAPTURE_SELECTION = 5
K_PHASE_FORCED_REMOVAL = 6
K_PHASE_COUNTER_REMOVAL = 7

# 正交 4 方向（必须和 C++ / encode 里的一致）
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ---------------------------------------------------------------------------
# Synthetic state / action generator
# ---------------------------------------------------------------------------

def _state_tensors() -> Tuple[torch.Tensor, ...]:
    device = torch.device("cpu")
    board = torch.zeros((6, 6), dtype=torch.int8, device=device)
    marks_black = torch.zeros((6, 6), dtype=torch.bool, device=device)
    marks_white = torch.zeros((6, 6), dtype=torch.bool, device=device)
    phase = torch.tensor(K_PHASE_PLACEMENT, dtype=torch.int64, device=device)
    current = torch.tensor(random_rng.choice([-1, 1]), dtype=torch.int64, device=device)
    pending_marks_required = torch.tensor(0, dtype=torch.int64, device=device)
    pending_marks_remaining = torch.tensor(0, dtype=torch.int64, device=device)
    pending_captures_required = torch.tensor(0, dtype=torch.int64, device=device)
    pending_captures_remaining = torch.tensor(0, dtype=torch.int64, device=device)
    forced_removals_done = torch.tensor(0, dtype=torch.int64, device=device)
    move_count = torch.tensor(0, dtype=torch.int64, device=device)
    return (
        board,
        marks_black,
        marks_white,
        phase,
        current,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
        move_count,
    )


def _sample_action_state(kind: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, int, int, int]]:
    (
        board,
        marks_black,
        marks_white,
        phase,
        current,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
        move_count,
    ) = _state_tensors()

    def rand_cell() -> int:
        return random_rng.randrange(36)

    if kind == K_ACTION_PLACEMENT:
        phase.fill_(K_PHASE_PLACEMENT)
        cell = rand_cell()
        board.view(-1)[cell] = 0  # ensure empty
        action = (kind, cell, 0, 0)

    elif kind == K_ACTION_MARK_SELECTION:
        phase.fill_(K_PHASE_MARK_SELECTION)
        opponent = int(-current.item())
        cell = rand_cell()
        board.view(-1)[cell] = opponent
        pending_marks_required.fill_(random_rng.choice([1, 2]))
        pending_marks_remaining.copy_(pending_marks_required)
        action = (kind, cell, 0, 0)

    elif kind == K_ACTION_PROCESS_REMOVAL:
        phase.fill_(K_PHASE_REMOVAL)
        board[0, 0] = 1
        board[0, 1] = -1
        marks_black[0, 0] = True
        marks_white[0, 1] = True
        action = (kind, 0, 0, 0)

    elif kind == K_ACTION_FORCED_REMOVAL:
        phase.fill_(K_PHASE_FORCED_REMOVAL)
        current.fill_(-1)
        board[0, 0] = 1
        forced_removals_done.fill_(0)
        action = (kind, 0, 0, 0)

    elif kind == K_ACTION_MOVEMENT:
        phase.fill_(K_PHASE_MOVEMENT)

        while True:
            origin = rand_cell()
            dir_idx = random_rng.randrange(len(DIRS))
            r, c = divmod(origin, 6)
            dr, dc = DIRS[dir_idx]
            r_to, c_to = r + dr, c + dc
            if 0 <= r_to < 6 and 0 <= c_to < 6:
                break

        dest = r_to * 6 + c_to

        board.view(-1).fill_(0)
        board.view(-1)[origin] = int(current.item())
        board.view(-1)[dest] = 0

        action = (kind, origin, dir_idx, 0)

    elif kind == K_ACTION_NO_MOVES_REMOVAL:
        phase.fill_(K_PHASE_MOVEMENT)
        opponent = int(-current.item())
        cell = rand_cell()
        board.view(-1)[cell] = opponent
        action = (kind, cell, 0, 0)

    elif kind == K_ACTION_CAPTURE_SELECTION:
        phase.fill_(K_PHASE_CAPTURE_SELECTION)
        opponent = int(-current.item())
        cell = rand_cell()
        board.view(-1)[cell] = opponent
        if opponent == -1:
            marks_white.view(-1)[cell] = True
        else:
            marks_black.view(-1)[cell] = True
        pending_captures_required.fill_(random_rng.choice([1, 2]))
        pending_captures_remaining.copy_(pending_captures_required)
        action = (kind, cell, 0, 0)

    elif kind == K_ACTION_COUNTER_REMOVAL:
        phase.fill_(K_PHASE_COUNTER_REMOVAL)
        opponent = int(-current.item())
        cell = rand_cell()
        board.view(-1)[cell] = opponent
        action = (kind, cell, 0, 0)

    else:
        kind = K_ACTION_PLACEMENT
        phase.fill_(K_PHASE_PLACEMENT)
        cell = rand_cell()
        action = (kind, cell, 0, 0)

    tensors = (
        board,
        marks_black,
        marks_white,
        phase,
        current,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
        move_count,
    )
    return tensors, action


def _stack_tensor_list(values: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(values, dim=0)


def _random_apply_batch(num_actions: int) -> Tuple[torch.Tensor, ...]:
    boards: List[torch.Tensor] = []
    marks_black: List[torch.Tensor] = []
    marks_white: List[torch.Tensor] = []
    phase: List[torch.Tensor] = []
    current_player: List[torch.Tensor] = []
    pending_marks_required: List[torch.Tensor] = []
    pending_marks_remaining: List[torch.Tensor] = []
    pending_captures_required: List[torch.Tensor] = []
    pending_captures_remaining: List[torch.Tensor] = []
    forced_removals_done: List[torch.Tensor] = []
    move_count: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []

    kinds = [
        K_ACTION_PLACEMENT,
        K_ACTION_MOVEMENT,
        K_ACTION_MARK_SELECTION,
        K_ACTION_CAPTURE_SELECTION,
        K_ACTION_FORCED_REMOVAL,
        K_ACTION_COUNTER_REMOVAL,
        K_ACTION_NO_MOVES_REMOVAL,
        K_ACTION_PROCESS_REMOVAL,
    ]

    for _ in range(num_actions):
        tensors, action = _sample_action_state(random_rng.choice(kinds))
        (
            board,
            mark_b,
            mark_w,
            ph,
            cur,
            pm_req,
            pm_rem,
            pc_req,
            pc_rem,
            forced,
            moves,
        ) = tensors

        boards.append(board)
        marks_black.append(mark_b)
        marks_white.append(mark_w)
        phase.append(ph)
        current_player.append(cur)
        pending_marks_required.append(pm_req)
        pending_marks_remaining.append(pm_rem)
        pending_captures_required.append(pc_req)
        pending_captures_remaining.append(pc_rem)
        forced_removals_done.append(forced)
        move_count.append(moves)
        actions.append(torch.tensor(action, dtype=torch.int32))

    parent_indices = torch.arange(num_actions, dtype=torch.int64)

    return (
        _stack_tensor_list(boards),
        _stack_tensor_list(marks_black),
        _stack_tensor_list(marks_white),
        torch.stack(phase, dim=0),
        torch.stack(current_player, dim=0),
        torch.stack(pending_marks_required, dim=0),
        torch.stack(pending_marks_remaining, dim=0),
        torch.stack(pending_captures_required, dim=0),
        torch.stack(pending_captures_remaining, dim=0),
        torch.stack(forced_removals_done, dim=0),
        torch.stack(move_count, dim=0),
        torch.stack(actions, dim=0),
        parent_indices,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_batch(tensors: Tuple[torch.Tensor, ...], device: torch.device):
    moved = tuple(t.to(device) for t in tensors)
    return v0_core.batch_apply_moves(*moved)


def _assert_equal(cpu_tensors, cuda_tensors) -> None:
    for cpu_tensor, cuda_tensor in zip(cpu_tensors, cuda_tensors):
        assert torch.equal(cpu_tensor.cpu(), cuda_tensor.cpu())


# ---------------------------------------------------------------------------
# Accuracy test
# ---------------------------------------------------------------------------

def test_fast_apply_moves_cuda_accuracy() -> None:
    if v0_core is None:
        _append_log(_ACCURACY_LOG, "SKIP: v0_core import failed; accuracy test not run.")
        pytest.skip("v0_core not importable")

    if not torch.cuda.is_available():
        _append_log(_ACCURACY_LOG, "SKIP: CUDA not available; accuracy test skipped.")
        pytest.skip("CUDA not available")

    batch = _random_apply_batch(NUM_ACTIONS)

    cpu_output = _run_batch(batch, torch.device("cpu"))
    try:
        cuda_output = _run_batch(batch, torch.device("cuda"))
    except RuntimeError as exc:
        msg = str(exc)
        if "CUDA kernels were not built" in msg or "CUDA kernels were not compiled" in msg:
            _append_log(
                _ACCURACY_LOG,
                "SKIP: CUDA kernels for fast apply moves not compiled; accuracy skipped.",
            )
            pytest.skip("CUDA kernels were not compiled for fast apply moves")
        raise

    _assert_equal(cpu_output, cuda_output)
    _append_log(
        _ACCURACY_LOG,
        f"OK: fast_apply_moves CPU vs CUDA parity passed "
        f"(num_actions={NUM_ACTIONS}, seed=0x{SEED:X}).",
    )


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fast_apply_moves_cuda_performance_smoke() -> None:
    if v0_core is None:
        _append_log(_PERF_LOG, "SKIP: v0_core import failed; performance test not run.")
        pytest.skip("v0_core not importable")

    if not torch.cuda.is_available():
        _append_log(_PERF_LOG, "SKIP: CUDA not available; performance test skipped.")
        pytest.skip("CUDA not available")

    batch = _random_apply_batch(NUM_ACTIONS)

    runs = 100
    cpu_times: List[float] = []
    cuda_times: List[float] = []

    # warmup
    _run_batch(batch, torch.device("cpu"))
    _run_batch(batch, torch.device("cuda"))
    torch.cuda.synchronize()

    for _ in range(runs):
        start = time.perf_counter()
        _run_batch(batch, torch.device("cpu"))
        end = time.perf_counter()
        cpu_times.append((end - start) * 1e3)

    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _run_batch(batch, torch.device("cuda"))
        torch.cuda.synchronize()
        end = time.perf_counter()
        cuda_times.append((end - start) * 1e3)

    avg_cpu = sum(cpu_times) / len(cpu_times)
    avg_cuda = sum(cuda_times) / len(cuda_times)

    header = (
        f"fast_apply_moves performance "
        f"(num_actions={NUM_ACTIONS}, runs={runs}, seed=0x{SEED:X}):"
    )
    summary = f"  cpu={avg_cpu:.3f} ms, cuda={avg_cuda:.3f} ms"

    print(header)
    for i, (c, g) in enumerate(zip(cpu_times, cuda_times), start=1):
        print(f"    run {i}: cpu={c:.3f} ms, cuda={g:.3f} ms")
    print(summary)

    _append_log(_PERF_LOG, header)
    for i, (c, g) in enumerate(zip(cpu_times, cuda_times), start=1):
        _append_log(_PERF_LOG, f"    run {i}: cpu={c:.3f} ms, cuda={g:.3f} ms")
    _append_log(_PERF_LOG, summary)

