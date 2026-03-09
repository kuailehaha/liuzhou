"""Shared storage helpers for v1 self-play payloads."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import torch

from .trajectory_buffer import TensorSelfPlayBatch


def _split_counts(total: int, parts: int) -> List[int]:
    total_i = max(0, int(total))
    parts_i = max(1, int(parts))
    if total_i <= 0:
        return []
    parts_i = min(parts_i, total_i)
    base = total_i // parts_i
    extra = total_i % parts_i
    return [base + (1 if idx < extra else 0) for idx in range(parts_i)]


def estimate_bytes_per_sample(samples: TensorSelfPlayBatch) -> int:
    """Estimate in-memory bytes per sample from actual tensor layouts."""
    n = max(1, int(samples.num_samples))
    total = 0
    for tensor in (
        samples.state_tensors,
        samples.legal_masks,
        samples.policy_targets,
        samples.value_targets,
        samples.soft_value_targets,
    ):
        if int(tensor.numel()) > 0:
            total += int(tensor.element_size()) * int(tensor.numel() // n)
    return max(1, total)


def plan_sample_ranges(
    *,
    total_samples: int,
    num_shards: int,
    target_samples_per_shard: int = 0,
    chunk_target_bytes: int = 0,
    bytes_per_sample: int = 0,
) -> List[Tuple[int, int]]:
    total_i = int(total_samples)
    if total_i <= 0:
        return []

    shard_count = max(1, min(int(num_shards), total_i))
    target_n = max(0, int(target_samples_per_shard))
    if int(chunk_target_bytes) > 0:
        bps = max(1, int(bytes_per_sample))
        target_n = max(1, int(chunk_target_bytes) // bps)

    if target_n > 0:
        shard_count = max(shard_count, int(math.ceil(total_i / float(target_n))))
        shard_count = min(shard_count, total_i)

    ranges: List[Tuple[int, int]] = []
    start = 0
    for size in _split_counts(total_i, shard_count):
        end = start + int(size)
        ranges.append((start, end))
        start = end
    return ranges


def slice_batch_cpu(
    samples: TensorSelfPlayBatch,
    *,
    start: int,
    end: int,
) -> TensorSelfPlayBatch:
    start_i = int(start)
    end_i = int(end)
    return TensorSelfPlayBatch(
        state_tensors=samples.state_tensors[start_i:end_i].to("cpu"),
        legal_masks=samples.legal_masks[start_i:end_i].to("cpu"),
        policy_targets=samples.policy_targets[start_i:end_i].to("cpu"),
        value_targets=samples.value_targets[start_i:end_i].to("cpu"),
        soft_value_targets=samples.soft_value_targets[start_i:end_i].to("cpu"),
    )


def save_self_play_payload(
    *,
    path: str,
    samples: TensorSelfPlayBatch,
    stats_payload: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "state_tensors": samples.state_tensors.detach().cpu(),
        "legal_masks": samples.legal_masks.detach().cpu(),
        "policy_targets": samples.policy_targets.detach().cpu(),
        "value_targets": samples.value_targets.detach().cpu(),
        "soft_value_targets": samples.soft_value_targets.detach().cpu(),
        "stats": dict(stats_payload),
        "metadata": dict(metadata),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
