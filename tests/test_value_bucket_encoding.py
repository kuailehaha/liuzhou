"""Unit tests for 101-bucket scalar value encoding/decoding."""

from __future__ import annotations

import math

import pytest
import torch

from src.neural_network import (
    bucket_logits_to_scalar,
    scalar_to_bucket_twohot,
)


def test_scalar_to_bucket_twohot_endpoints() -> None:
    values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    target = scalar_to_bucket_twohot(values, num_bins=101)
    assert tuple(target.shape) == (3, 101)
    assert float(target[0, 0].item()) == pytest.approx(1.0)
    assert float(target[1, 50].item()) == pytest.approx(1.0)
    assert float(target[2, 100].item()) == pytest.approx(1.0)


def test_scalar_twohot_roundtrip_error_bound() -> None:
    values = torch.linspace(-1.0, 1.0, steps=401, dtype=torch.float32)
    target = scalar_to_bucket_twohot(values, num_bins=101)
    centers = torch.linspace(-1.0, 1.0, steps=101, dtype=torch.float32)
    decoded = (target * centers.view(1, -1)).sum(dim=-1)
    max_err = float((decoded - values).abs().max().item())
    assert max_err <= 0.02 + 1e-6


def test_bucket_logits_to_scalar_onehot() -> None:
    logits = torch.full((3, 101), -20.0, dtype=torch.float32)
    logits[0, 0] = 20.0
    logits[1, 50] = 20.0
    logits[2, 100] = 20.0
    decoded = bucket_logits_to_scalar(logits, num_bins=101)
    assert float(decoded[0].item()) == pytest.approx(-1.0, abs=1e-4)
    assert float(decoded[1].item()) == pytest.approx(0.0, abs=1e-4)
    assert float(decoded[2].item()) == pytest.approx(1.0, abs=1e-4)


def test_mcts_to_scalar_value_from_bucket_logits() -> None:
    try:
        from v1.python.mcts_gpu import V1RootMCTS
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"mcts import unavailable: {exc}")

    obj = object.__new__(V1RootMCTS)
    logits = torch.full((1, 101), -20.0, dtype=torch.float32)
    logits[0, 75] = 20.0
    out = V1RootMCTS._to_scalar_value(obj, logits)
    expected = -1.0 + (2.0 * 75.0 / 100.0)
    assert float(out.item()) == pytest.approx(expected, abs=5e-3)
    assert math.isfinite(float(out.item()))
