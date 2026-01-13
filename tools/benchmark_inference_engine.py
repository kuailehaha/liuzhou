"""
Benchmark TorchScriptRunner vs InferenceEngine (CUDA Graph).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


def _parse_dtype(value: str) -> torch.dtype:
    key = value.strip().lower()
    if key in ("float32", "fp32", "f32"):
        return torch.float32
    if key in ("float16", "fp16", "f16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def _load_model(device: torch.device, dtype: torch.dtype, checkpoint: Optional[str]) -> ChessNet:
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _export_torchscript(model: ChessNet, sample_input: torch.Tensor, path: Path) -> Path:
    scripted = torch.jit.trace(model, sample_input, strict=False)
    scripted.save(str(path))
    return path


def _time_loop(fn, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    return (end - start) / max(1, iters)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchScriptRunner vs InferenceEngine (CUDA Graph)."
    )
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:0.")
    parser.add_argument("--dtype", default="float16", help="float32|float16|bfloat16.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for forward.")
    parser.add_argument("--iters", type=int, default=5000, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--engine-warmup", type=int, default=5, help="Warmup iterations inside engine.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load.")
    parser.add_argument("--model-path", default=None, help="Existing TorchScript path to load.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for input.")
    parser.add_argument("--include-eager", action="store_true", help="Include eager model timing.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 is not supported on CPU.")

    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 not supported on this CUDA device.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = _load_model(device, dtype, args.checkpoint)

    batch_size = int(args.batch_size)
    input_tensor = torch.randn(
        batch_size,
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=device,
        dtype=dtype,
    )

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
    else:
        model_path = Path("build") / "torchscript_engine_model.ts.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _export_torchscript(model, input_tensor, model_path)

    import v0_core

    runner = v0_core.TorchScriptRunner(
        str(model_path),
        device=str(device),
        dtype=args.dtype,
    )
    engine = v0_core.InferenceEngine(
        str(model_path),
        device=str(device),
        dtype=args.dtype,
        batch_size=batch_size,
        input_channels=NUM_INPUT_CHANNELS,
        height=GameState.BOARD_SIZE,
        width=GameState.BOARD_SIZE,
        warmup_iters=args.engine_warmup,
    )

    result = {
        "device": str(device),
        "dtype": args.dtype,
        "batch": batch_size,
        "graph_enabled": bool(engine.graph_enabled),
    }

    if args.include_eager:
        with torch.inference_mode():
            eager_time = _time_loop(lambda: model(input_tensor), args.iters, args.warmup, device)
        result["eager_ms"] = eager_time * 1e3

    ts_time = _time_loop(lambda: runner.forward(input_tensor), args.iters, args.warmup, device)
    graph_time = _time_loop(
        lambda: engine.forward(input_tensor, batch_size),
        args.iters,
        args.warmup,
        device,
    )

    result["torchscript_ms"] = ts_time * 1e3
    result["graph_ms"] = graph_time * 1e3
    result["torchscript_pos_s"] = batch_size / ts_time
    result["graph_pos_s"] = batch_size / graph_time
    result["speedup"] = ts_time / graph_time if graph_time > 0 else float("inf")

    print("InferenceEngine benchmark:")
    print(
        "  device={device} dtype={dtype} batch={batch} graph_enabled={graph_enabled}".format(**result)
    )
    if args.include_eager:
        print(f"  eager_ms={result['eager_ms']:.3f}")
    print(f"  torchscript_ms={result['torchscript_ms']:.3f}")
    print(f"  graph_ms={result['graph_ms']:.3f}")
    print(f"  speedup={result['speedup']:.2f}x")
    print(f"  torchscript_pos_s={result['torchscript_pos_s']:.2f}")
    print(f"  graph_pos_s={result['graph_pos_s']:.2f}")


if __name__ == "__main__":
    main()
