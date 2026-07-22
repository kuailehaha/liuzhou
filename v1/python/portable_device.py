"""Explicit device selection for the portable V1 backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PortableDeviceResolution:
    requested: str
    device: torch.device
    fallback_count: int = 0
    fallback_reasons: Tuple[str, ...] = ()


def _mps_unavailable_reason() -> str:
    if not torch.backends.mps.is_built():
        return "installed PyTorch was not built with MPS support"
    return "torch.backends.mps.is_available() is False"


def resolve_portable_device(requested: str | torch.device) -> PortableDeviceResolution:
    """Resolve ``auto/cpu/mps/cuda`` without silent accelerator fallback.

    ``auto`` intentionally means MPS on an available Apple GPU and CPU otherwise.
    The production CUDA backend remains a separately selected V1 path.
    """

    raw = str(requested).strip().lower()
    if raw in {"", "auto"}:
        if torch.backends.mps.is_available():
            raw = "mps"
            resolution = PortableDeviceResolution(requested="auto", device=torch.device("mps"))
        else:
            reason = f"auto selected CPU because {_mps_unavailable_reason()}"
            resolution = PortableDeviceResolution(
                requested="auto",
                device=torch.device("cpu"),
                fallback_count=1,
                fallback_reasons=(reason,),
            )
    else:
        device = torch.device(raw)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(f"MPS was explicitly requested but {_mps_unavailable_reason()}.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is False.")
        if device.type not in {"cpu", "mps", "cuda"}:
            raise ValueError(
                f"Unsupported portable device {requested!r}; expected auto, cpu, mps, or cuda."
            )
        resolution = PortableDeviceResolution(requested=str(requested), device=device)

    fallback_env = str(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")).strip().lower()
    if resolution.device.type == "mps" and fallback_env in _TRUE_VALUES:
        raise RuntimeError(
            "Portable MPS execution refuses PYTORCH_ENABLE_MPS_FALLBACK because it can "
            "silently execute unsupported operators on CPU. Unset the variable and retry."
        )
    return resolution
