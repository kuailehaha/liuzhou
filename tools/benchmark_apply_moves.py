"""
Benchmark legacy apply_move vs tensor metadata Python path vs C++ fast batch apply.

Usage example:
    python -m tools.benchmark_apply_moves --states 512 --batch-size 64 --runs 3 --max-random-moves 40 --seed 0
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import List, Sequence

import torch

from src.game_state import GameState, Phase
from src.move_generator import apply_move, generate_all_legal_moves

from tools.cross_check_mcts import CrossCheckConfig, sample_states
from v1.game.fast_apply_moves import batch_apply_moves_fast
from v1.game.move_encoder import DEFAULT_ACTION_SPEC, ActionEncodingSpec, encode_actions
from v1.game.state_batch import TensorStateBatch, from_game_states


ACTION_KIND_PLACEMENT = 1
ACTION_KIND_MOVEMENT = 2
ACTION_KIND_MARK_SELECTION = 3
ACTION_KIND_CAPTURE_SELECTION = 4
ACTION_KIND_FORCED_REMOVAL_SELECTION = 5
ACTION_KIND_COUNTER_REMOVAL_SELECTION = 6
ACTION_KIND_NO_MOVES_REMOVAL_SELECTION = 7
ACTION_KIND_PROCESS_REMOVAL = 8


def _metadata_to_move(code, state: GameState):
    kind, primary, secondary, _extra = [int(x) for x in code]
    board_size = state.BOARD_SIZE
    if kind == ACTION_KIND_PLACEMENT:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.PLACEMENT, "action_type": "place", "position": (r, c)}
    if kind == ACTION_KIND_MOVEMENT:
        r_from, c_from = divmod(primary, board_size)
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        dr, dc = dirs[secondary]
        return {
            "phase": Phase.MOVEMENT,
            "action_type": "move",
            "from_position": (r_from, c_from),
            "to_position": (r_from + dr, c_from + dc),
        }
    if kind == ACTION_KIND_MARK_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.MARK_SELECTION, "action_type": "mark", "position": (r, c)}
    if kind == ACTION_KIND_CAPTURE_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.CAPTURE_SELECTION, "action_type": "capture", "position": (r, c)}
    if kind == ACTION_KIND_FORCED_REMOVAL_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.FORCED_REMOVAL, "action_type": "remove", "position": (r, c)}
    if kind == ACTION_KIND_COUNTER_REMOVAL_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.COUNTER_REMOVAL, "action_type": "counter_remove", "position": (r, c)}
    if kind == ACTION_KIND_NO_MOVES_REMOVAL_SELECTION:
        r, c = divmod(primary, board_size)
        return {"phase": Phase.MOVEMENT, "action_type": "no_moves_remove", "position": (r, c)}
    if kind == ACTION_KIND_PROCESS_REMOVAL:
        return {"phase": Phase.REMOVAL, "action_type": "process_removal"}
    raise ValueError(f"Unsupported action kind: {kind}")


@dataclass
class ApplyBenchmarkChunk:
    start: int
    end: int
    states: Sequence[GameState]
    tensor_cpu: TensorStateBatch
    legacy_moves: List[List[dict]]
    parent_indices: torch.LongTensor
    action_codes: torch.IntTensor
    python_moves: List[dict]


def prepare_chunks(
    states: Sequence[GameState],
    batch_size: int,
    spec: ActionEncodingSpec,
) -> List[ApplyBenchmarkChunk]:
    chunks: List[ApplyBenchmarkChunk] = []
    for offset in range(0, len(states), batch_size):
        slice_states = states[offset : offset + batch_size]
        tensor_cpu = from_game_states(slice_states, device=torch.device("cpu"))
        legacy_moves = [generate_all_legal_moves(state) for state in slice_states]

        mask_and_meta = encode_actions(tensor_cpu, spec, return_metadata=True)
        if not isinstance(mask_and_meta, tuple):
            raise RuntimeError("encode_actions did not return metadata; fast legal mask extension required.")
        legal_mask, metadata = mask_and_meta
        if metadata is None:
            raise RuntimeError("encode_actions metadata unavailable; fast legal mask extension required.")

        parent_indices: List[int] = []
        action_codes: List[List[int]] = []
        python_moves: List[dict] = []

        for local_idx, state in enumerate(slice_states):
            legal_indices = legal_mask[local_idx].nonzero(as_tuple=False).view(-1)
            if legal_indices.numel() == 0:
                continue
            for action_idx in legal_indices.tolist():
                code = metadata[local_idx, action_idx]
                parent_indices.append(local_idx)
                action_codes.append([int(code[0]), int(code[1]), int(code[2]), int(code[3])])
                python_moves.append(_metadata_to_move(code, state))

        parent_tensor = torch.tensor(parent_indices, dtype=torch.long)
        codes_tensor = torch.tensor(action_codes, dtype=torch.int32) if action_codes else torch.empty((0, 4), dtype=torch.int32)

        chunks.append(
            ApplyBenchmarkChunk(
                start=offset,
                end=offset + len(slice_states),
                states=slice_states,
                tensor_cpu=tensor_cpu,
                legacy_moves=legacy_moves,
                parent_indices=parent_tensor,
                action_codes=codes_tensor,
                python_moves=python_moves,
            )
        )
    return chunks


def benchmark_legacy(chunks: Sequence[ApplyBenchmarkChunk]) -> float:
    start = time.perf_counter()
    for chunk in chunks:
        for state, moves in zip(chunk.states, chunk.legacy_moves):
            for move in moves:
                apply_move(state, move, quiet=True)
    return time.perf_counter() - start


def benchmark_tensor_python(chunks: Sequence[ApplyBenchmarkChunk]) -> float:
    start = time.perf_counter()
    for chunk in chunks:
        if chunk.parent_indices.numel() == 0:
            continue
        for parent_idx, move in zip(chunk.parent_indices.tolist(), chunk.python_moves):
            apply_move(chunk.states[parent_idx], move, quiet=True)
    return time.perf_counter() - start


def benchmark_tensor_fast(chunks: Sequence[ApplyBenchmarkChunk]) -> float:
    start = time.perf_counter()
    for chunk in chunks:
        if chunk.parent_indices.numel() == 0:
            continue
        result = batch_apply_moves_fast(chunk.tensor_cpu, chunk.parent_indices, chunk.action_codes)
        if result is None:
            raise RuntimeError("fast apply moves extension unavailable.")
    return time.perf_counter() - start


def summarize(samples: Sequence[float]) -> str:
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return f"{mean * 1000:.2f} ± {stdev * 1000:.2f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark apply_move throughput across implementations.")
    parser.add_argument("--states", type=int, default=1000, help="Number of random states to sample.")
    parser.add_argument("--batch-size", type=int, default=64, help="Chunk size for tensor fast path.")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark repetitions.")
    parser.add_argument("--device", type=str, default="cpu", help="Device flag for parity; benchmark runs on CPU only.")
    parser.add_argument("--max-random-moves", type=int, default=80, help="Random rollout depth for sampled states.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for state sampling.")
    args = parser.parse_args()

    if args.device != "cpu":
        print("Warning: apply-moves fast benchmark currently supports CPU only. Forcing cpu.")

    cfg = CrossCheckConfig(
        num_states=args.states,
        max_random_moves=args.max_random_moves,
        seed=args.seed,
    )
    states = sample_states(cfg)
    if not states:
        raise RuntimeError("No states sampled for benchmarking.")

    spec = DEFAULT_ACTION_SPEC
    chunks = prepare_chunks(states, args.batch_size, spec)

    # Probe fast availability
    fast_available = False
    for chunk in chunks:
        if chunk.parent_indices.numel() == 0:
            continue
        fast_available = batch_apply_moves_fast(chunk.tensor_cpu, chunk.parent_indices, chunk.action_codes) is not None
        break

    legacy_times: List[float] = []
    python_times: List[float] = []
    fast_times: List[float] = []

    for run in range(args.runs):
        legacy_elapsed = benchmark_legacy(chunks)
        legacy_times.append(legacy_elapsed)

        python_elapsed = benchmark_tensor_python(chunks)
        python_times.append(python_elapsed)

        fast_elapsed = None
        if fast_available:
            fast_elapsed = benchmark_tensor_fast(chunks)
            fast_times.append(fast_elapsed)

        fast_str = f"{fast_elapsed * 1000:.2f} ms" if fast_elapsed is not None else "N/A"
        print(
            f"Run {run + 1}/{args.runs}: legacy={legacy_elapsed * 1000:.2f} ms, "
            f"tensor-python={python_elapsed * 1000:.2f} ms, "
            f"tensor-fast={fast_str}"
        )

    print("\nSummary:")
    print(f"  Legacy apply_move: {summarize(legacy_times)}")
    print(f"  Tensor apply (metadata Python): {summarize(python_times)}")
    if fast_available and fast_times:
        print(f"  Tensor apply (C++ fast): {summarize(fast_times)}")
    else:
        print("  Tensor apply (C++ fast): unavailable")


if __name__ == "__main__":
    main()

