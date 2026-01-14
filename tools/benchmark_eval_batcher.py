"""Benchmark EvalBatcher by merging eval requests across multiple MCTS cores."""

from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import v0_core
from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


def _random_state(max_moves: int, rng: random.Random) -> GameState:
    state = GameState()
    for _ in range(rng.randint(0, max_moves)):
        legal = generate_all_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        state = apply_move(state, move, quiet=True)
        if state.is_game_over():
            break
    return state


def _dtype_from_string(value: str) -> torch.dtype:
    key = value.strip().lower()
    if key in ("float32", "fp32", "f32"):
        return torch.float32
    if key in ("float16", "fp16", "f16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def _default_torchscript_dtype(device: torch.device) -> str:
    return "float16" if device.type == "cuda" else "float32"


def _export_torchscript(
    model: ChessNet,
    device: torch.device,
    dtype_str: str,
    batch_size: int,
) -> str:
    dtype = _dtype_from_string(dtype_str)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 export is not supported on CPU.")
    model_copy = copy.deepcopy(model)
    model_copy.to(device=device, dtype=dtype)
    model_copy.eval()
    inputs = torch.zeros(
        batch_size,
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        scripted = torch.jit.trace(model_copy, inputs, strict=False)
        tmp = tempfile.NamedTemporaryFile(prefix="v0_eval_batcher_", suffix=".pt", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        scripted.save(str(tmp_path))
    return str(tmp_path)


def _hist_labels(batch_size: int, buckets: int) -> List[str]:
    if buckets <= 1:
        return ["1+"]
    width = max(1, batch_size // (buckets - 1))
    labels = []
    for i in range(buckets - 1):
        low = i * width + 1
        high = (i + 1) * width
        labels.append(f"{low}-{high}")
    labels.append(f"{(buckets - 1) * width + 1}+")
    return labels


def _format_eval_stats(stats: Dict[str, object], batch_size: int) -> Dict[str, object]:
    eval_calls = int(stats.get("eval_calls", 0))
    eval_leaves = int(stats.get("eval_leaves", 0))
    full512_calls = int(stats.get("full512_calls", 0))
    avg_batch = (eval_leaves / eval_calls) if eval_calls else 0.0
    full512_ratio = (full512_calls / eval_calls) if eval_calls else 0.0
    pad_leaves = eval_calls * batch_size - eval_leaves
    pad_ratio = (pad_leaves / (eval_calls * batch_size)) if eval_calls else 0.0

    hist_counts = list(stats.get("hist", []))
    labels = _hist_labels(batch_size, max(1, len(hist_counts)))
    hist = {label: int(hist_counts[i]) if i < len(hist_counts) else 0 for i, label in enumerate(labels)}

    return {
        "eval_calls": eval_calls,
        "eval_leaves": eval_leaves,
        "avg_batch": avg_batch,
        "full512_ratio": full512_ratio,
        "pad_leaves": pad_leaves,
        "pad_ratio": pad_ratio,
        "hist": hist,
        "graph_batch_size": batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EvalBatcher across multiple MCTS cores.")
    parser.add_argument("--num-cores", type=int, default=8, help="Number of parallel MCTS cores.")
    parser.add_argument("--sims", type=int, default=200, help="Simulations per core.")
    parser.add_argument("--iterations", type=int, default=1, help="Repeated runs to accumulate stats.")
    parser.add_argument("--batch-leaves", type=int, default=16, help="Leaf batch size per MCTS core.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cpu/cuda).")
    parser.add_argument("--max-moves", type=int, default=60, help="Max random plies for root states.")
    parser.add_argument("--inference-batch-size", type=int, default=512, help="EvalBatcher/InferenceEngine batch size.")
    parser.add_argument("--inference-warmup-iters", type=int, default=5, help="Warmup iterations for graph capture.")
    parser.add_argument("--timeout-ms", type=int, default=2, help="EvalBatcher flush timeout in ms.")
    parser.add_argument("--torchscript-path", type=str, default=None, help="Optional TorchScript model path.")
    parser.add_argument("--torchscript-dtype", type=str, default=None, help="TorchScript dtype (float16/float32/bfloat16).")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    dtype_str = args.torchscript_dtype or _default_torchscript_dtype(device)
    if dtype_str.strip().lower() in ("auto", "none"):
        dtype_str = _default_torchscript_dtype(device)

    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()

    created_path: Optional[str] = None
    model_path = args.torchscript_path
    if not model_path:
        model_path = _export_torchscript(
            model=model,
            device=device,
            dtype_str=dtype_str,
            batch_size=max(1, int(args.inference_batch_size)),
        )
        created_path = model_path

    engine = v0_core.InferenceEngine(
        model_path,
        device=str(device),
        dtype=dtype_str,
        batch_size=max(1, int(args.inference_batch_size)),
        input_channels=NUM_INPUT_CHANNELS,
        height=GameState.BOARD_SIZE,
        width=GameState.BOARD_SIZE,
        warmup_iters=max(0, int(args.inference_warmup_iters)),
    )
    print(
        "[benchmark_eval_batcher] graph_enabled={} dtype={} device={}".format(
            "yes" if engine.graph_enabled else "no",
            engine.dtype,
            engine.device,
        )
    )

    batcher = v0_core.EvalBatcher(
        engine,
        batch_size=max(1, int(args.inference_batch_size)),
        input_channels=NUM_INPUT_CHANNELS,
        height=GameState.BOARD_SIZE,
        width=GameState.BOARD_SIZE,
        timeout_ms=max(0, int(args.timeout_ms)),
    )

    cores = []
    for i in range(max(1, int(args.num_cores))):
        cfg = v0_core.MCTSConfig()
        cfg.num_simulations = int(args.sims)
        cfg.batch_size = int(args.batch_leaves)
        cfg.device = str(device)
        cfg.seed = int(args.seed + i + 1)
        core = v0_core.MCTSCore(cfg)
        core.set_eval_batcher(batcher)
        core.set_root_state(_random_state(args.max_moves, rng))
        cores.append(core)

    batcher.reset_eval_stats()

    def _run(core: v0_core.MCTSCore) -> None:
        core.run_simulations(int(args.sims))

    total_time = 0.0
    for _ in range(max(1, int(args.iterations))):
        threads = [threading.Thread(target=_run, args=(core,)) for core in cores]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time += time.perf_counter() - start

    stats = batcher.get_eval_stats()
    payload = _format_eval_stats(stats, batch_size=int(args.inference_batch_size))
    payload["num_cores"] = int(args.num_cores)
    payload["sims_per_core"] = int(args.sims)
    payload["iterations"] = int(args.iterations)
    payload["total_time_sec"] = total_time
    total_simulations = int(args.num_cores) * int(args.sims) * int(args.iterations)
    payload["simulations_per_sec"] = (total_simulations / total_time) if total_time > 0 else None
    print(json.dumps(payload, sort_keys=True))

    batcher.shutdown()
    if created_path:
        try:
            Path(created_path).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
