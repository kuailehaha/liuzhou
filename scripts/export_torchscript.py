#!/usr/bin/env python3
"""
Export ChessNet to TorchScript for C++ (libtorch) inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


def _parse_dtype(value: str) -> Optional[torch.dtype]:
    key = value.strip().lower()
    if key in ("", "auto", "none"):
        return None
    if key in ("float32", "fp32", "f32"):
        return torch.float32
    if key in ("float16", "fp16", "f16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def _dtype_name(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return "auto"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def _load_model(device: torch.device, dtype: Optional[torch.dtype], checkpoint: Optional[str]) -> ChessNet:
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def export_torchscript(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 export on CPU is not supported.")

    model = _load_model(device, dtype, args.checkpoint)
    model_dtype = next(model.parameters()).dtype
    batch_size = int(args.batch_size)
    input_shape = (
        batch_size,
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
    )
    dummy_input = torch.zeros(input_shape, device=device, dtype=model_dtype)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("TorchScript export config:")
    print(f"  output={output_path}")
    print(f"  device={device}")
    print(f"  dtype={_dtype_name(dtype)}")
    print(f"  batch_size={batch_size}")
    print(f"  input_shape={tuple(dummy_input.shape)}")
    print(f"  checkpoint={args.checkpoint or 'none'}")
    print(f"  use_script={args.use_script}")
    print(f"  freeze={args.freeze}")
    print(f"  optimize_for_inference={args.optimize}")

    with torch.inference_mode():
        if args.use_script:
            scripted = torch.jit.script(model)
        else:
            scripted = torch.jit.trace(model, dummy_input, strict=False)
        if args.freeze:
            scripted = torch.jit.freeze(scripted)
        if args.optimize:
            scripted = torch.jit.optimize_for_inference(scripted)
        scripted.save(str(output_path))

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ChessNet to TorchScript for libtorch inference."
    )
    parser.add_argument("--output", required=True, help="Output .pt path for TorchScript.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load.")
    parser.add_argument("--device", default="cpu", help="Device for export (cpu/cuda/cuda:0).")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Model/input dtype: float32|float16|bfloat16|auto.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Fixed batch size for tracing.")
    parser.add_argument("--use-script", action="store_true", help="Use torch.jit.script instead of trace.")
    parser.add_argument("--freeze", action="store_true", help="Freeze the TorchScript module.")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run torch.jit.optimize_for_inference on the scripted module.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_torchscript(args)


if __name__ == "__main__":
    main()
