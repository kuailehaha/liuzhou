"""Streaming IterableDataset for v1 training — loads shard files on-the-fly."""

from __future__ import annotations

import gc
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.utils.data

from .trajectory_buffer import TensorSelfPlayBatch


_LOAD_MAX_RETRIES = 3
_LOAD_RETRY_DELAY = 2.0


def _torch_load_retry(path: str) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(_LOAD_MAX_RETRIES):
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:
            last_exc = exc
            if attempt < _LOAD_MAX_RETRIES - 1:
                gc.collect()
                time.sleep(_LOAD_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"Failed to load {path} after {_LOAD_MAX_RETRIES} attempts") from last_exc


@dataclass
class ShardSpec:
    """Describes one shard file to load."""
    path: str
    num_samples: int
    sample_budget: int  # 0 = use all samples


def resolve_shard_specs(
    primary_input: str,
    replay_inputs: List[str],
    replay_budget_per_file: int,
    *,
    ddp_rank: int = 0,
    ddp_world: int = 1,
) -> Tuple[List[ShardSpec], int]:
    """Resolve all shard file paths and sample counts without loading tensor data.

    Returns (shard_specs, total_estimated_samples).
    """
    specs: List[ShardSpec] = []
    total = 0

    for path, is_replay in [(primary_input, False)] + [(r, True) for r in replay_inputs]:
        if not os.path.exists(path):
            continue
        budget = replay_budget_per_file if is_replay else 0

        manifest = _torch_load_retry(path)
        if isinstance(manifest, dict) and str(manifest.get("payload_format", "")).strip().lower() == "v1_sharded_manifest":
            shard_files_raw = manifest.get("shard_files", [])
            shard_sizes_raw = manifest.get("shard_sizes", [])
            base_dir = os.path.dirname(path) or "."

            resolved: List[Tuple[str, int]] = []
            for i, entry in enumerate(shard_files_raw):
                sp = str(entry).strip()
                if not sp:
                    continue
                full = sp if os.path.isabs(sp) else os.path.join(base_dir, sp)
                sz = int(shard_sizes_raw[i]) if i < len(shard_sizes_raw) else 0
                resolved.append((full, sz))

            if ddp_world > 1:
                resolved = [resolved[i] for i in range(len(resolved)) if (i % ddp_world) == ddp_rank]

            for full, sz in resolved:
                effective = min(sz, budget) if budget > 0 and sz > 0 else sz
                specs.append(ShardSpec(path=full, num_samples=sz, sample_budget=budget))
                total += effective if effective > 0 else sz
        else:
            n = 0
            if isinstance(manifest, TensorSelfPlayBatch):
                n = int(manifest.num_samples)
            elif isinstance(manifest, dict):
                st = manifest.get("state_tensors")
                if st is not None and hasattr(st, "shape"):
                    n = int(st.shape[0])
            effective = min(n, budget) if budget > 0 and n > 0 else n
            specs.append(ShardSpec(path=path, num_samples=n, sample_budget=budget))
            total += effective if effective > 0 else n
            del manifest

    gc.collect()
    return specs, total


def _load_shard_tensors(
    path: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load a shard file and return the 5 tensor fields, or None on failure."""
    try:
        payload = _torch_load_retry(path)
    except Exception:
        return None

    if isinstance(payload, TensorSelfPlayBatch):
        return (
            payload.state_tensors.cpu(),
            payload.legal_masks.cpu(),
            payload.policy_targets.cpu(),
            payload.value_targets.cpu(),
            payload.soft_value_targets.cpu(),
        )

    if isinstance(payload, dict):
        required = ("state_tensors", "legal_masks", "policy_targets", "value_targets", "soft_value_targets")
        if all(k in payload for k in required):
            result = tuple(payload[k].cpu() for k in required)
            del payload
            return result  # type: ignore[return-value]

    del payload
    return None


class StreamingSelfPlayDataset(torch.utils.data.IterableDataset):
    """Streams training samples from shard files without full pre-loading.

    Each DataLoader worker is assigned a disjoint subset of shard files.
    Within each shard, samples are shuffled; across epochs, shard order is
    re-shuffled.
    """

    def __init__(
        self,
        shard_specs: List[ShardSpec],
        *,
        epoch_seed: int = 0,
    ) -> None:
        super().__init__()
        self._specs = list(shard_specs)
        self._epoch_seed = int(epoch_seed)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        my_specs = [self._specs[i] for i in range(len(self._specs)) if (i % num_workers) == worker_id]
        if not my_specs:
            return

        rng = random.Random(self._epoch_seed * 10007 + worker_id)
        rng.shuffle(my_specs)

        for spec in my_specs:
            tensors = _load_shard_tensors(spec.path)
            if tensors is None:
                continue

            states, masks, policy, values, soft_values = tensors
            n = int(states.shape[0])
            if n <= 0:
                del tensors
                continue

            if spec.sample_budget > 0 and n > spec.sample_budget:
                idx = torch.randperm(n)[: spec.sample_budget]
                states = states.index_select(0, idx)
                masks = masks.index_select(0, idx)
                policy = policy.index_select(0, idx)
                values = values.index_select(0, idx)
                soft_values = soft_values.index_select(0, idx)
                n = spec.sample_budget

            perm = torch.randperm(n)
            for i in range(n):
                j = int(perm[i].item())
                yield (
                    states[j],
                    masks[j],
                    policy[j],
                    values[j],
                    soft_values[j],
                )

            del tensors, states, masks, policy, values, soft_values, perm
            gc.collect()


def build_streaming_dataloader(
    shard_specs: List[ShardSpec],
    *,
    batch_size: int,
    num_workers: int = 8,
    epoch_seed: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> torch.utils.data.DataLoader:
    dataset = StreamingSelfPlayDataset(shard_specs, epoch_seed=epoch_seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        persistent_workers=False,
    )
