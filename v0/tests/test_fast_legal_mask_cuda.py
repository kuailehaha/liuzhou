import pytest
import torch

import v0_core


def _random_batch(batch_size: int, device: torch.device):
    board = torch.randint(-1, 2, (batch_size, 6, 6), dtype=torch.int8, device=device)
    marks_black = torch.randint(0, 2, (batch_size, 6, 6), dtype=torch.bool, device=device)
    marks_white = torch.randint(0, 2, (batch_size, 6, 6), dtype=torch.bool, device=device)
    phase = torch.randint(1, 8, (batch_size,), dtype=torch.int64, device=device)
    current_player = torch.randint(0, 2, (batch_size,), dtype=torch.int64, device=device)
    current_player = current_player.mul(-2).add(1)  # map {0,1} -> {-1,1}
    pending_marks_required = torch.zeros(batch_size, dtype=torch.int64, device=device)
    pending_marks_remaining = torch.randint(0, 3, (batch_size,), dtype=torch.int64, device=device)
    pending_captures_required = torch.zeros(batch_size, dtype=torch.int64, device=device)
    pending_captures_remaining = torch.randint(0, 3, (batch_size,), dtype=torch.int64, device=device)
    forced_removals_done = torch.randint(0, 3, (batch_size,), dtype=torch.int64, device=device)
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


def test_encode_actions_fast_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size = 8
    cpu_tensors = _random_batch(batch_size, torch.device("cpu"))
    placement_dim = cpu_tensors[0].size(-1) * cpu_tensors[0].size(-2)
    movement_dim = placement_dim * 4
    selection_dim = placement_dim
    auxiliary_dim = 1

    mask_cpu, meta_cpu = v0_core.encode_actions_fast(
        *cpu_tensors,
        placement_dim,
        movement_dim,
        selection_dim,
        auxiliary_dim,
    )

    cuda_tensors = tuple(t.to("cuda") for t in cpu_tensors)
    try:
        mask_cuda, meta_cuda = v0_core.encode_actions_fast(
            *cuda_tensors,
            placement_dim,
            movement_dim,
            selection_dim,
            auxiliary_dim,
        )
    except RuntimeError as exc:  # pragma: no cover - fallback path
        if "CUDA kernels were not built" in str(exc):
            pytest.skip("CUDA kernels were not compiled")
        raise

    assert torch.equal(mask_cpu, mask_cuda.cpu())
    assert torch.equal(meta_cpu, meta_cuda.cpu())
