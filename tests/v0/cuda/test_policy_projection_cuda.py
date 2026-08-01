from __future__ import annotations

import pytest
import torch

try:
    import v0_core
except ImportError:
    v0_core = None


pytestmark = pytest.mark.skipif(
    v0_core is None or not torch.cuda.is_available(),
    reason="CUDA v0_core is required",
)


def _reference(p1, p2, pmc, legal):
    batch = int(p1.size(0))
    combined = torch.empty((batch, 220), dtype=p1.dtype, device=p1.device)
    combined[:, :36] = p1
    cells = torch.arange(36, dtype=torch.int64, device=p1.device)
    rows, cols = cells // 6, cells % 6
    movement = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dest_rows, dest_cols = rows + dr, cols + dc
        valid = (
            dest_rows.ge(0)
            & dest_rows.lt(6)
            & dest_cols.ge(0)
            & dest_cols.lt(6)
        )
        dest = (dest_rows * 6 + dest_cols).clamp(0, 35)
        scores = p2 + p1.index_select(1, dest)
        movement.append(scores.masked_fill(~valid.view(1, -1), float("-inf")))
    combined[:, 36:180] = torch.stack(movement, dim=2).reshape(batch, 144)
    combined[:, 180:216] = pmc
    combined[:, 216:] = 0
    masked = combined.masked_fill(~legal, float("-inf"))
    has_legal = legal.any(dim=1)
    has_finite = masked.isfinite().any(dim=1)
    valid = has_legal & has_finite
    safe = torch.where(valid.view(-1, 1), masked, torch.zeros_like(masked))
    probabilities = torch.where(
        valid.view(-1, 1), torch.softmax(safe, dim=1), torch.zeros_like(safe)
    )
    fallback = torch.where(
        legal, torch.zeros_like(masked), torch.full_like(masked, float("-inf"))
    )
    masked = torch.where(
        (has_legal & ~has_finite).view(-1, 1), fallback, masked
    )
    return probabilities, masked


@pytest.mark.parametrize(
    ("dtype", "atol"),
    ((torch.float32, 1e-7), (torch.float16, 5e-4)),
)
def test_cuda_projection_matches_dense_reference(dtype, atol):
    generator = torch.Generator(device="cuda").manual_seed(20260801)
    heads = [
        torch.randn((128, 36), dtype=dtype, device="cuda", generator=generator)
        for _ in range(3)
    ]
    legal = torch.rand((128, 220), device="cuda", generator=generator).lt(0.2)
    actual = v0_core.project_policy_logits_fast(*heads, legal, 36, 144, 36, 4)
    expected = _reference(*heads, legal)
    assert torch.allclose(actual[0], expected[0], atol=atol, rtol=0)
    finite = expected[1].isfinite()
    assert torch.equal(actual[1].isinf(), expected[1].isinf())
    assert torch.equal(actual[1][finite], expected[1][finite])


def test_cuda_projection_preserves_non_finite_diagnostics():
    p1 = torch.zeros((3, 36), device="cuda")
    p2 = torch.zeros_like(p1)
    pmc = torch.zeros_like(p1)
    p1[:, 0] = float("nan")
    legal = torch.zeros((3, 220), dtype=torch.bool, device="cuda")
    legal[1, :2] = True
    legal[2, 0] = True
    actual = v0_core.project_policy_logits_fast(p1, p2, pmc, legal, 36, 144, 36, 4)
    expected = _reference(p1, p2, pmc, legal)
    assert torch.equal(torch.isnan(actual[0]), torch.isnan(expected[0]))
    assert torch.equal(torch.nan_to_num(actual[0]), torch.nan_to_num(expected[0]))
    assert torch.equal(torch.isnan(actual[1]), torch.isnan(expected[1]))
    assert torch.equal(torch.isinf(actual[1]), torch.isinf(expected[1]))
    assert torch.equal(torch.nan_to_num(actual[1]), torch.nan_to_num(expected[1]))
