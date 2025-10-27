import os
from functools import lru_cache
from typing import Dict, Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


class ModelLoadError(RuntimeError):
    """Custom error raised when a model checkpoint cannot be loaded."""


@lru_cache(maxsize=None)
def get_model(model_path: Optional[str], device: str) -> ChessNet:
    """
    Load (and cache) a ChessNet model from the given checkpoint path.

    If ``model_path`` is None the function returns a randomly initialised model,
    which is still useful for wiring up the end-to-end data flow.
    """
    device = device or "cpu"
    model = ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    ).to(device)
    model.eval()

    if model_path is None:
        return model

    expanded_path = os.path.expanduser(model_path)
    if not os.path.exists(expanded_path):
        raise ModelLoadError(f"Model checkpoint not found: {expanded_path}")

    payload = torch.load(expanded_path, map_location=device)
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict)
    return model


def clear_model_cache() -> None:
    """Reset the cached models (mainly for tests or hot reloads)."""
    get_model.cache_clear()  # type: ignore[attr-defined]


def _extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    """
    Handle the different checkpoint shapes produced by training scripts.
    """
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            state_dict = payload["model_state_dict"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
    else:
        raise ModelLoadError("Checkpoint does not contain a valid state_dict mapping.")

    # Strip potential DistributedDataParallel prefixes.
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
