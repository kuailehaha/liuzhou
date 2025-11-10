"""
Benchmark policy projection throughput across legacy, tensor-Python, and tensor-fast paths.

Usage example:
    python -m tools.benchmark_policy_projection --states 512 --batch-size 128 --runs 3 --device cpu --verify
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import List, Sequence

import torch

from src.game_state import GameState
from src.move_generator import generate_all_legal_moves
from src.neural_network import get_move_probabilities

from v1.game.move_encoder import DEFAULT_ACTION_SPEC, ActionEncodingSpec, encode_actions
from v1.game.state_batch import TensorStateBatch, from_game_states
from v1.net.encoding import _project_policy_logits_python
from v1.net.project_policy_logits_fast import project_policy_logits_fast

from tools.cross_check_mcts import CrossCheckConfig, sample_states


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BenchmarkChunk:
    start: int
    end: int
    states: Sequence[GameState]
    tensor_cpu: TensorStateBatch
    legal_mask_cpu: torch.BoolTensor
    legal_moves: List[list]


def prepare_chunks(
    states: Sequence[GameState],
    batch_size: int,
    spec: ActionEncodingSpec,
) -> List[BenchmarkChunk]:
    chunks: List[BenchmarkChunk] = []
    for offset in range(0, len(states), batch_size):
        slice_states = states[offset : offset + batch_size]
        tensor_cpu = from_game_states(slice_states, device=torch.device("cpu"))
        legal_mask = encode_actions(tensor_cpu, spec)
        moves = [generate_all_legal_moves(state) for state in slice_states]
        chunks.append(
            BenchmarkChunk(
                start=offset,
                end=offset + len(slice_states),
                states=slice_states,
                tensor_cpu=tensor_cpu,
                legal_mask_cpu=legal_mask.to(torch.device("cpu")),
                legal_moves=moves,
            )
        )
    return chunks


def benchmark_legacy(
    chunks: Sequence[BenchmarkChunk],
    log_p1: torch.Tensor,
    log_p2: torch.Tensor,
    log_pmc: torch.Tensor,
) -> float:
    start = time.perf_counter()
    for chunk in chunks:
        for local_idx, state in enumerate(chunk.states):
            idx = chunk.start + local_idx
            get_move_probabilities(
                log_p1[idx],
                log_p2[idx],
                log_pmc[idx],
                chunk.legal_moves[local_idx],
                board_size=state.BOARD_SIZE,
                device=str(log_p1.device),
            )
    return time.perf_counter() - start


def benchmark_tensor_python(
    chunks: Sequence[BenchmarkChunk],
    log_p1_cpu: torch.Tensor,
    log_p2_cpu: torch.Tensor,
    log_pmc_cpu: torch.Tensor,
    device: torch.device,
    spec: ActionEncodingSpec,
) -> float:
    start = time.perf_counter()
    for chunk in chunks:
        mask = chunk.legal_mask_cpu.to(device)
        l1 = log_p1_cpu[chunk.start : chunk.end].to(device)
        l2 = log_p2_cpu[chunk.start : chunk.end].to(device)
        l3 = log_pmc_cpu[chunk.start : chunk.end].to(device)
        _project_policy_logits_python((l1, l2, l3), mask, spec)
    _sync_if_needed(device)
    return time.perf_counter() - start


def benchmark_tensor_fast(
    chunks: Sequence[BenchmarkChunk],
    log_p1_cpu: torch.Tensor,
    log_p2_cpu: torch.Tensor,
    log_pmc_cpu: torch.Tensor,
    spec: ActionEncodingSpec,
) -> float:
    start = time.perf_counter()
    for chunk in chunks:
        result = project_policy_logits_fast(
            log_p1_cpu[chunk.start : chunk.end],
            log_p2_cpu[chunk.start : chunk.end],
            log_pmc_cpu[chunk.start : chunk.end],
            chunk.legal_mask_cpu,
            spec,
        )
        if result is None:
            raise RuntimeError("Fast policy projection extension unavailable.")
    return time.perf_counter() - start


def verify_outputs(
    chunks: Sequence[BenchmarkChunk],
    log_p1_cpu: torch.Tensor,
    log_p2_cpu: torch.Tensor,
    log_pmc_cpu: torch.Tensor,
    spec: ActionEncodingSpec,
) -> bool:
    fast_available = True
    for chunk in chunks:
        python_probs, python_logits = _project_policy_logits_python(
            (
                log_p1_cpu[chunk.start : chunk.end],
                log_p2_cpu[chunk.start : chunk.end],
                log_pmc_cpu[chunk.start : chunk.end],
            ),
            chunk.legal_mask_cpu,
            spec,
        )
        fast = project_policy_logits_fast(
            log_p1_cpu[chunk.start : chunk.end],
            log_p2_cpu[chunk.start : chunk.end],
            log_pmc_cpu[chunk.start : chunk.end],
            chunk.legal_mask_cpu,
            spec,
        )
        if fast is None:
            fast_available = False
        else:
            probs_fast, logits_fast = fast
            torch.testing.assert_close(
                probs_fast, python_probs, atol=1e-6, rtol=0.0
            )
            torch.testing.assert_close(
                logits_fast, python_logits, atol=1e-6, rtol=0.0
            )

        for local_idx, state in enumerate(chunk.states):
            legacy_probs, legacy_logits = get_move_probabilities(
                log_p1_cpu[chunk.start + local_idx],
                log_p2_cpu[chunk.start + local_idx],
                log_pmc_cpu[chunk.start + local_idx],
                chunk.legal_moves[local_idx],
                board_size=state.BOARD_SIZE,
                device="cpu",
            )
            legacy_probs_tensor = torch.tensor(
                legacy_probs, dtype=python_probs.dtype
            )
            legacy_logits_tensor = legacy_logits.to(python_logits.dtype)

            legal_indices = chunk.legal_mask_cpu[local_idx].nonzero(as_tuple=False).view(-1)
            torch.testing.assert_close(
                python_probs[local_idx, legal_indices],
                legacy_probs_tensor,
                atol=1e-6,
                rtol=0.0,
            )
            torch.testing.assert_close(
                python_logits[local_idx, legal_indices],
                legacy_logits_tensor,
                atol=1e-6,
                rtol=0.0,
            )
    return fast_available


def summarize(samples: Sequence[float]) -> str:
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return f"{mean * 1000:.2f} ± {stdev * 1000:.2f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark policy projection performance across implementations."
    )
    parser.add_argument("--states", type=int, default=1000, help="Number of random states to sample.")
    parser.add_argument("--batch-size", type=int, default=64, help="Chunk size for tensor paths.")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark repetitions.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for tensor-Python benchmark (cpu or cuda).")
    parser.add_argument("--max-random-moves", type=int, default=80, help="Random rollout depth for sampled states.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for state sampling and logits.")
    parser.add_argument("--verify", action="store_true", help="Also verify outputs against legacy logic.")
    args = parser.parse_args()

    spec = DEFAULT_ACTION_SPEC
    device = torch.device(args.device)

    cfg = CrossCheckConfig(
        num_states=args.states,
        max_random_moves=args.max_random_moves,
        seed=args.seed,
    )
    states = sample_states(cfg)
    if not states:
        raise RuntimeError("No states sampled for benchmarking.")

    chunks = prepare_chunks(states, args.batch_size, spec)

    torch.manual_seed(args.seed)
    head_dim = spec.placement_dim
    log_p1_cpu = torch.randn(len(states), head_dim)
    log_p2_cpu = torch.randn_like(log_p1_cpu)
    log_pmc_cpu = torch.randn_like(log_p1_cpu)

    fast_available = False
    if args.verify:
        print("Verifying tensor implementations against legacy outputs...")
        fast_available = verify_outputs(chunks, log_p1_cpu, log_p2_cpu, log_pmc_cpu, spec)
        print("Verification succeeded.")
    else:
        probe_chunk = chunks[0]
        fast_available = project_policy_logits_fast(
            log_p1_cpu[probe_chunk.start : probe_chunk.end],
            log_p2_cpu[probe_chunk.start : probe_chunk.end],
            log_pmc_cpu[probe_chunk.start : probe_chunk.end],
            probe_chunk.legal_mask_cpu,
            spec,
        ) is not None

    legacy_times: List[float] = []
    python_times: List[float] = []
    fast_times: List[float] = []

    for run in range(args.runs):
        legacy_elapsed = benchmark_legacy(chunks, log_p1_cpu, log_p2_cpu, log_pmc_cpu)
        legacy_times.append(legacy_elapsed)

        python_elapsed = benchmark_tensor_python(
            chunks,
            log_p1_cpu,
            log_p2_cpu,
            log_pmc_cpu,
            device,
            spec,
        )
        python_times.append(python_elapsed)

        fast_elapsed_str = "N/A"
        if fast_available:
            fast_elapsed = benchmark_tensor_fast(chunks, log_p1_cpu, log_p2_cpu, log_pmc_cpu, spec)
            fast_times.append(fast_elapsed)
            fast_elapsed_str = f"{fast_elapsed * 1000:.2f} ms"

        print(
            f"Run {run + 1}/{args.runs}: legacy={legacy_elapsed * 1000:.2f} ms, "
            f"tensor-python={python_elapsed * 1000:.2f} ms, "
            f"tensor-fast={fast_elapsed_str}"
        )

    print("\nSummary:")
    print(f"  Legacy combination: {summarize(legacy_times)}")
    print(f"  Tensor project_policy_logits (Python): {summarize(python_times)}")
    if fast_available and fast_times:
        print(f"  Tensor project_policy_logits (C++ fast): {summarize(fast_times)}")
    else:
        print("  Tensor project_policy_logits (C++ fast): unavailable")


if __name__ == "__main__":
    main()

