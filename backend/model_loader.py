from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


class ModelLoadError(RuntimeError):
    """Custom error raised when a model checkpoint cannot be loaded."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _get_model_cached(
    resolved_path: str,
    device: str,
    checkpoint_sha256: str,
) -> ChessNet:
    del checkpoint_sha256  # Included in the cache key to invalidate mutable aliases.
    model = ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    )
    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict)
    model.to(device or "cpu")
    model.eval()
    return model


def get_model_with_metadata(
    model_path: Optional[str],
    device: str,
) -> Tuple[ChessNet, Dict[str, str]]:
    if not model_path:
        raise ModelLoadError(
            "No model checkpoint was configured; random-model fallback is disabled."
        )
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ModelLoadError(f"Model checkpoint not found: {path}")
    checkpoint_sha256 = _sha256(path)
    try:
        model = _get_model_cached(
            str(path),
            str(device or "cpu"),
            checkpoint_sha256,
        )
    except Exception as exc:
        raise ModelLoadError(f"Could not load model checkpoint {path}: {exc}") from exc
    return model, {
        "modelPath": str(path),
        "modelSha256": checkpoint_sha256,
        "device": str(device or "cpu"),
    }


def get_model(model_path: Optional[str], device: str) -> ChessNet:
    model, _metadata = get_model_with_metadata(model_path, device)
    return model


def clear_model_cache() -> None:
    """Reset cached checkpoint+device+SHA entries."""

    _get_model_cached.cache_clear()


def _extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            state_dict = payload["model_state_dict"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
    else:
        raise ModelLoadError("Checkpoint does not contain a valid state_dict mapping.")

    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
