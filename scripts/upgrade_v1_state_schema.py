#!/usr/bin/env python
"""Explicitly migrate a legacy 11-channel V1 model/optimizer pair to Markov v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from src.neural_network import (  # noqa: E402
    LEGACY_NUM_INPUT_CHANNELS,
    LEGACY_STATE_ENCODING_SCHEMA,
    MARKOV_NUM_INPUT_CHANNELS,
    MARKOV_STATE_ENCODING_SCHEMA,
    checkpoint_state_encoding_schema,
    state_encoding_feature_names,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_model_checkpoint(
    payload: dict[str, Any],
    *,
    source_sha256: str,
    parent_iteration: int,
) -> dict[str, Any]:
    source_schema = checkpoint_state_encoding_schema(payload)
    if source_schema != LEGACY_STATE_ENCODING_SCHEMA:
        raise ValueError(
            f"Expected a legacy_v1 source checkpoint, got {source_schema}."
        )
    source_state = payload.get("model_state_dict", payload)
    if not isinstance(source_state, dict):
        raise ValueError("Source checkpoint has no model_state_dict.")
    if "stem_conv.weight" not in source_state:
        raise ValueError("Source checkpoint is missing stem_conv.weight.")
    old_stem = source_state["stem_conv.weight"]
    if tuple(old_stem.shape[1:]) != (LEGACY_NUM_INPUT_CHANNELS, 3, 3):
        raise ValueError(f"Unexpected legacy stem shape: {tuple(old_stem.shape)}")
    new_stem = old_stem.new_zeros(
        (
            int(old_stem.shape[0]),
            MARKOV_NUM_INPUT_CHANNELS,
            int(old_stem.shape[2]),
            int(old_stem.shape[3]),
        )
    )
    new_stem[:, :LEGACY_NUM_INPUT_CHANNELS].copy_(old_stem)
    migrated_state = dict(source_state)
    migrated_state["stem_conv.weight"] = new_stem

    migrated = dict(payload)
    migrated["model_state_dict"] = migrated_state
    migrated["num_input_channels"] = MARKOV_NUM_INPUT_CHANNELS
    migrated["state_encoding_schema"] = MARKOV_STATE_ENCODING_SCHEMA
    migrated["state_encoding_features"] = list(
        state_encoding_feature_names(MARKOV_STATE_ENCODING_SCHEMA)
    )
    migrated["schema_migration"] = {
        "source_schema": LEGACY_STATE_ENCODING_SCHEMA,
        "target_schema": MARKOV_STATE_ENCODING_SCHEMA,
        "source_checkpoint_sha256": str(source_sha256),
        "parent_iteration": int(parent_iteration),
        "stem_old_channels": LEGACY_NUM_INPUT_CHANNELS,
        "stem_new_channels": MARKOV_NUM_INPUT_CHANNELS,
        "new_stem_channels_initialized_to_zero": True,
    }
    return migrated


def migrate_optimizer_state(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Optimizer checkpoint must be a mapping.")
    state = payload.get("state")
    groups = payload.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise ValueError("Optimizer checkpoint lacks state/param_groups.")

    migrated_state: dict[Any, Any] = {}
    stem_entries = 0
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, dict):
            migrated_state[parameter_id] = parameter_state
            continue
        migrated_parameter_state = dict(parameter_state)
        tensor_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
        matching_keys = [
            key
            for key in tensor_keys
            if isinstance(parameter_state.get(key), torch.Tensor)
            and parameter_state[key].dim() == 4
            and int(parameter_state[key].shape[1]) == LEGACY_NUM_INPUT_CHANNELS
            and tuple(parameter_state[key].shape[2:]) == (3, 3)
        ]
        if matching_keys:
            stem_entries += 1
            for key in matching_keys:
                old_tensor = parameter_state[key]
                new_tensor = old_tensor.new_zeros(
                    (
                        int(old_tensor.shape[0]),
                        MARKOV_NUM_INPUT_CHANNELS,
                        int(old_tensor.shape[2]),
                        int(old_tensor.shape[3]),
                    )
                )
                new_tensor[:, :LEGACY_NUM_INPUT_CHANNELS].copy_(old_tensor)
                migrated_parameter_state[key] = new_tensor
        migrated_state[parameter_id] = migrated_parameter_state
    if stem_entries != 1:
        raise ValueError(
            "Expected exactly one optimizer state entry for the legacy stem, "
            f"found {stem_entries}."
        )
    migrated = dict(payload)
    migrated["state"] = migrated_state
    return migrated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-optimizer")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-optimizer")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--parent-iteration", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_checkpoint = Path(args.source_checkpoint).expanduser().resolve()
    source_optimizer = (
        Path(args.source_optimizer).expanduser().resolve()
        if args.source_optimizer
        else None
    )
    output_checkpoint = Path(args.output_checkpoint).expanduser().resolve()
    output_optimizer = (
        Path(args.output_optimizer).expanduser().resolve()
        if args.output_optimizer
        else None
    )
    report_path = Path(args.report_json).expanduser().resolve()
    if (source_optimizer is None) != (output_optimizer is None):
        raise ValueError(
            "--source-optimizer and --output-optimizer must be provided together."
        )
    for source in (source_checkpoint, source_optimizer):
        if source is None:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
    for output in (output_checkpoint, output_optimizer, report_path):
        if output is None:
            continue
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing migration output: {output}"
            )

    source_checkpoint_sha = sha256_file(source_checkpoint)
    source_optimizer_sha = (
        sha256_file(source_optimizer)
        if source_optimizer is not None
        else None
    )
    checkpoint_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("Source checkpoint must be a dict payload.")
    migrated_checkpoint = migrate_model_checkpoint(
        checkpoint_payload,
        source_sha256=source_checkpoint_sha,
        parent_iteration=int(args.parent_iteration),
    )
    _atomic_torch_save(migrated_checkpoint, output_checkpoint)
    if source_optimizer is not None and output_optimizer is not None:
        optimizer_payload = torch.load(
            source_optimizer, map_location="cpu", weights_only=False
        )
        migrated_optimizer = migrate_optimizer_state(optimizer_payload)
        _atomic_torch_save(migrated_optimizer, output_optimizer)

    report = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_optimizer": (
            str(source_optimizer) if source_optimizer is not None else None
        ),
        "source_optimizer_sha256": source_optimizer_sha,
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "output_optimizer": (
            str(output_optimizer) if output_optimizer is not None else None
        ),
        "output_optimizer_sha256": (
            sha256_file(output_optimizer)
            if output_optimizer is not None
            else None
        ),
        "parent_iteration": int(args.parent_iteration),
        "source_schema": LEGACY_STATE_ENCODING_SCHEMA,
        "target_schema": MARKOV_STATE_ENCODING_SCHEMA,
        "old_replay_compatible": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
