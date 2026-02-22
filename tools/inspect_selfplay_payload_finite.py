#!/usr/bin/env python
"""Inspect v1 self-play payloads for non-finite tensors.

Supports both single-file payloads and sharded manifest payloads.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch


REQUIRED_KEYS = [
    "state_tensors",
    "legal_masks",
    "policy_targets",
    "value_targets",
    "soft_value_targets",
]


@dataclass
class TensorFiniteStats:
    total_elements: int
    nonfinite_elements: int


@dataclass
class SampleFiniteStats:
    total_samples: int
    nonfinite_samples: int


def _tensor_nonfinite_stats(t: torch.Tensor) -> TensorFiniteStats:
    flat = t.detach().to("cpu").view(-1)
    total = int(flat.numel())
    if total <= 0:
        return TensorFiniteStats(total_elements=0, nonfinite_elements=0)
    nonfinite = int(torch.count_nonzero(torch.isfinite(flat).logical_not()).item())
    return TensorFiniteStats(total_elements=total, nonfinite_elements=nonfinite)


def _sample_nonfinite_stats(
    value_targets: torch.Tensor,
    soft_value_targets: torch.Tensor,
    policy_targets: torch.Tensor,
    state_tensors: torch.Tensor,
) -> SampleFiniteStats:
    values_bad = torch.isfinite(value_targets.detach().to("cpu").view(-1)).logical_not()
    soft_bad = torch.isfinite(soft_value_targets.detach().to("cpu").view(-1)).logical_not()
    policy_bad = torch.isfinite(policy_targets.detach().to("cpu")).all(dim=1).logical_not()
    state_bad = (
        torch.isfinite(state_tensors.detach().to("cpu").view(int(state_tensors.shape[0]), -1))
        .all(dim=1)
        .logical_not()
    )
    bad = values_bad.logical_or(soft_bad).logical_or(policy_bad).logical_or(state_bad)
    return SampleFiniteStats(
        total_samples=int(bad.numel()),
        nonfinite_samples=int(torch.count_nonzero(bad).item()),
    )


def _inspect_single_payload(path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported payload format in {path}: {type(payload)!r}")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise RuntimeError(f"Missing keys in {path}: {missing}")

    state_tensors = payload["state_tensors"]
    policy_targets = payload["policy_targets"]
    value_targets = payload["value_targets"]
    soft_value_targets = payload["soft_value_targets"]

    value_stats = _tensor_nonfinite_stats(value_targets)
    soft_stats = _tensor_nonfinite_stats(soft_value_targets)
    policy_stats = _tensor_nonfinite_stats(policy_targets)
    state_stats = _tensor_nonfinite_stats(state_tensors)
    sample_stats = _sample_nonfinite_stats(
        value_targets=value_targets,
        soft_value_targets=soft_value_targets,
        policy_targets=policy_targets,
        state_tensors=state_tensors,
    )
    return {
        "path": path,
        "num_samples": int(value_targets.numel()),
        "value_nonfinite": value_stats.nonfinite_elements,
        "value_total": value_stats.total_elements,
        "soft_nonfinite": soft_stats.nonfinite_elements,
        "soft_total": soft_stats.total_elements,
        "policy_nonfinite": policy_stats.nonfinite_elements,
        "policy_total": policy_stats.total_elements,
        "state_nonfinite": state_stats.nonfinite_elements,
        "state_total": state_stats.total_elements,
        "sample_nonfinite": sample_stats.nonfinite_samples,
        "sample_total": sample_stats.total_samples,
    }


def _inspect_payload(path: str) -> List[Dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported payload format in {path}: {type(payload)!r}")

    payload_format = str(payload.get("payload_format", "")).strip().lower()
    if payload_format != "v1_sharded_manifest":
        return [_inspect_single_payload(path)]

    shard_files = payload.get("shard_files")
    if not isinstance(shard_files, list) or not shard_files:
        raise RuntimeError(f"Invalid sharded manifest {path}: shard_files missing or empty.")

    base_dir = os.path.dirname(path) or "."
    rows: List[Dict[str, object]] = []
    for idx, entry in enumerate(shard_files):
        shard_item = str(entry).strip()
        if not shard_item:
            continue
        shard_path = shard_item if os.path.isabs(shard_item) else os.path.join(base_dir, shard_item)
        row = _inspect_single_payload(shard_path)
        row["manifest"] = path
        row["shard_index"] = int(idx)
        rows.append(row)
    return rows


def _print_rows(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No payload rows found.")
        return

    total_samples = 0
    total_bad_samples = 0
    total_value_nonfinite = 0
    total_soft_nonfinite = 0
    total_policy_nonfinite = 0
    total_state_nonfinite = 0

    for row in rows:
        total_samples += int(row["sample_total"])
        total_bad_samples += int(row["sample_nonfinite"])
        total_value_nonfinite += int(row["value_nonfinite"])
        total_soft_nonfinite += int(row["soft_nonfinite"])
        total_policy_nonfinite += int(row["policy_nonfinite"])
        total_state_nonfinite += int(row["state_nonfinite"])

        prefix = f"[shard {row['shard_index']}]" if "shard_index" in row else "[payload]"
        print(
            f"{prefix} path={row['path']} "
            f"samples_bad={row['sample_nonfinite']}/{row['sample_total']} "
            f"value_nonfinite={row['value_nonfinite']}/{row['value_total']} "
            f"soft_nonfinite={row['soft_nonfinite']}/{row['soft_total']} "
            f"policy_nonfinite={row['policy_nonfinite']}/{row['policy_total']} "
            f"state_nonfinite={row['state_nonfinite']}/{row['state_total']}"
        )

    print(
        "[summary] "
        f"samples_bad={total_bad_samples}/{total_samples} "
        f"value_nonfinite={total_value_nonfinite} "
        f"soft_nonfinite={total_soft_nonfinite} "
        f"policy_nonfinite={total_policy_nonfinite} "
        f"state_nonfinite={total_state_nonfinite}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect non-finite values in v1 self-play payloads.")
    parser.add_argument("payload", type=str, help="Path to selfplay_iter_XXX.pt (single or sharded manifest).")
    args = parser.parse_args()

    path = str(args.payload).strip()
    if not path:
        print("payload path is required", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(f"payload not found: {path}", file=sys.stderr)
        return 2

    rows = _inspect_payload(path)
    _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
