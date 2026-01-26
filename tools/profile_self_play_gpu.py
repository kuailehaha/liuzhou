#!/usr/bin/env python
"""
GPU Profiler for Self-Play: Diagnose low power / high memory usage.

This script profiles a single self-play game with detailed timing and memory tracking
to identify where GPU time is actually being spent.

Usage:
    # Basic profiling
    python -m tools.profile_self_play_gpu --device cuda --sims 1024 --batch-leaves 512
    
    # With CUDA profiling (requires nsight or torch profiler)
    python -m tools.profile_self_play_gpu --device cuda --torch-profile
    
    # Memory tracking
    python -m tools.profile_self_play_gpu --device cuda --memory-tracking
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

# Check if v0 module is available
try:
    from v0.python.mcts import MCTS as V0MCTS
    V0_AVAILABLE = True
except ImportError:
    V0_AVAILABLE = False
    print("Warning: v0 module not available, will use legacy MCTS")


class GPUProfiler:
    """Tracks GPU timing and memory usage."""
    
    def __init__(self, device: torch.device, track_memory: bool = False):
        self.device = device
        self.track_memory = track_memory
        self.timings: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Dict[str, int]] = []
        self.is_cuda = device.type == "cuda"
        
    @contextmanager
    def time_section(self, name: str):
        """Context manager to time a code section."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        
        try:
            yield
        finally:
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
            
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
    
    def snapshot_memory(self, tag: str = ""):
        """Record current GPU memory state."""
        if not self.is_cuda or not self.track_memory:
            return
            
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        
        self.memory_snapshots.append({
            "tag": tag,
            "allocated_mb": allocated / 1024 / 1024,
            "reserved_mb": reserved / 1024 / 1024,
            "max_allocated_mb": max_allocated / 1024 / 1024,
            "timestamp": time.perf_counter(),
        })
    
    def print_timing_summary(self):
        """Print timing breakdown."""
        print("\n" + "=" * 60)
        print("GPU TIMING BREAKDOWN")
        print("=" * 60)
        
        total_time = 0
        for name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times) if times else 0
            total = sum(times)
            total_time += total
            count = len(times)
            print(f"  {name:30s}: {total:8.3f}s ({count:5d} calls, avg={avg*1000:.2f}ms)")
        
        print("-" * 60)
        print(f"  {'TOTAL':30s}: {total_time:8.3f}s")
        print()
        
        # Calculate ratios
        if "forward_pass" in self.timings and total_time > 0:
            forward_time = sum(self.timings["forward_pass"])
            forward_ratio = forward_time / total_time * 100
            print(f"  GPU Forward Pass Ratio: {forward_ratio:.1f}%")
            print(f"  CPU/Other Overhead: {100 - forward_ratio:.1f}%")
    
    def print_memory_summary(self):
        """Print memory usage summary."""
        if not self.memory_snapshots:
            return
            
        print("\n" + "=" * 60)
        print("GPU MEMORY USAGE")
        print("=" * 60)
        
        for snap in self.memory_snapshots[-10:]:  # Last 10 snapshots
            print(f"  [{snap['tag']:20s}] allocated={snap['allocated_mb']:.1f}MB "
                  f"reserved={snap['reserved_mb']:.1f}MB max={snap['max_allocated_mb']:.1f}MB")


def profile_single_game(
    model: ChessNet,
    device: torch.device,
    mcts_simulations: int,
    batch_leaves: int,
    inference_batch_size: int,
    max_moves: int = 50,
    profiler: Optional[GPUProfiler] = None,
) -> Dict[str, float]:
    """Profile a single self-play game and return timing stats."""
    
    if profiler is None:
        profiler = GPUProfiler(device)
    
    profiler.snapshot_memory("game_start")
    
    mcts = V0MCTS(
        model=model,
        num_simulations=mcts_simulations,
        exploration_weight=1.0,
        temperature=1.0,
        device=str(device),
        add_dirichlet_noise=True,
        batch_K=batch_leaves,
        inference_backend="graph",
        inference_batch_size=inference_batch_size,
    )
    
    profiler.snapshot_memory("mcts_created")
    
    state = GameState()
    move_count = 0
    search_times = []
    
    while move_count < max_moves and not state.is_game_over():
        with profiler.time_section("mcts_search"):
            moves, policy = mcts.search(state)
        
        if not moves:
            break
            
        # Select move
        import numpy as np
        policy = np.array(policy)
        if policy.sum() > 0:
            policy = policy / policy.sum()
            idx = np.random.choice(len(moves), p=policy)
        else:
            idx = np.random.randint(len(moves))
        
        from src.move_generator import apply_move
        state = apply_move(state, moves[idx], quiet=True)
        mcts.advance_root(moves[idx])
        
        move_count += 1
        
        if move_count % 10 == 0:
            profiler.snapshot_memory(f"move_{move_count}")
    
    profiler.snapshot_memory("game_end")
    
    return {
        "moves_played": move_count,
        "total_search_time": sum(profiler.timings.get("mcts_search", [])),
    }


def profile_with_torch_profiler(
    model: ChessNet,
    device: torch.device,
    mcts_simulations: int,
    batch_leaves: int,
    inference_batch_size: int,
):
    """Use torch.profiler for detailed GPU kernel analysis."""
    from torch.profiler import profile, ProfilerActivity, schedule
    
    print("\n" + "=" * 60)
    print("TORCH PROFILER ANALYSIS")
    print("=" * 60)
    
    mcts = V0MCTS(
        model=model,
        num_simulations=mcts_simulations,
        exploration_weight=1.0,
        temperature=1.0,
        device=str(device),
        add_dirichlet_noise=False,
        batch_K=batch_leaves,
        inference_backend="graph",
        inference_batch_size=inference_batch_size,
    )
    
    state = GameState()
    
    # Profile a few MCTS searches
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(3):  # 3 searches
            moves, policy = mcts.search(state)
            if moves:
                from src.move_generator import apply_move
                state = apply_move(state, moves[0], quiet=True)
                mcts.advance_root(moves[0])
    
    # Print key insights
    print("\nTop 20 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print("\nTop 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # Export for visualization
    trace_path = "self_play_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace exported to {trace_path} (open in chrome://tracing)")


def analyze_inference_efficiency(
    model: ChessNet,
    device: torch.device,
    batch_size: int,
):
    """Analyze pure inference efficiency to isolate GPU compute from MCTS overhead."""
    print("\n" + "=" * 60)
    print("PURE INFERENCE BENCHMARK")
    print("=" * 60)
    
    model.eval()
    
    # Create dummy inputs
    inputs = torch.randn(
        batch_size, NUM_INPUT_CHANNELS, GameState.BOARD_SIZE, GameState.BOARD_SIZE,
        device=device, dtype=torch.float32
    )
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(inputs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    iterations = 100
    start = time.perf_counter()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(inputs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / iterations
    samples_per_sec = batch_size * iterations / elapsed
    
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Avg per forward: {avg_time * 1000:.2f}ms")
    print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
    
    # Estimate theoretical GPU utilization
    # H20 specs: ~60 TFLOPS FP32, ~120 TFLOPS TF32
    # If we knew model FLOPs, we could estimate utilization


def main():
    parser = argparse.ArgumentParser(description="Profile GPU usage during self-play")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--sims", type=int, default=1024, help="MCTS simulations per move")
    parser.add_argument("--batch-leaves", type=int, default=512, help="Batch leaves for MCTS")
    parser.add_argument("--inference-batch-size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--max-moves", type=int, default=30, help="Max moves per game")
    parser.add_argument("--memory-tracking", action="store_true", help="Track memory usage")
    parser.add_argument("--torch-profile", action="store_true", help="Use torch.profiler")
    parser.add_argument("--inference-only", action="store_true", help="Benchmark pure inference")
    args = parser.parse_args()
    
    if not V0_AVAILABLE:
        print("Error: v0 module required for this profiler")
        return
    
    device = torch.device(args.device)
    print(f"Profiling on device: {device}")
    
    # Load model
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()
    
    # Pure inference benchmark
    if args.inference_only:
        analyze_inference_efficiency(model, device, args.inference_batch_size)
        return
    
    # Create profiler
    profiler = GPUProfiler(device, track_memory=args.memory_tracking)
    
    print(f"\nRunning self-play with:")
    print(f"  MCTS simulations: {args.sims}")
    print(f"  Batch leaves: {args.batch_leaves}")
    print(f"  Inference batch size: {args.inference_batch_size}")
    print(f"  Max moves: {args.max_moves}")
    
    # Profile single game
    with profiler.time_section("total_game"):
        stats = profile_single_game(
            model=model,
            device=device,
            mcts_simulations=args.sims,
            batch_leaves=args.batch_leaves,
            inference_batch_size=args.inference_batch_size,
            max_moves=args.max_moves,
            profiler=profiler,
        )
    
    print(f"\nGame completed: {stats['moves_played']} moves")
    
    profiler.print_timing_summary()
    profiler.print_memory_summary()
    
    # Torch profiler for detailed analysis
    if args.torch_profile:
        profile_with_torch_profiler(
            model, device, args.sims, args.batch_leaves, args.inference_batch_size
        )
    
    # Print GPU info
    if device.type == "cuda":
        print("\n" + "=" * 60)
        print("GPU INFO")
        print("=" * 60)
        print(f"  Device: {torch.cuda.get_device_name(device)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        if args.memory_tracking:
            print(f"  Peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            print(f"  Peak reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
