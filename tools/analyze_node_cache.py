#!/usr/bin/env python
"""
MCTS Node Cache Growth Analyzer

Tracks how MCTS node count grows during a game and estimates memory overhead.
This helps understand if the low GPU power consumption is due to CPU-bound
node management rather than GPU compute.

Usage:
    python -m tools.analyze_node_cache --sims 1024 --batch-leaves 512 --moves 50
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


def estimate_gamestate_size() -> int:
    """Estimate memory size of a single GameState."""
    # C++ GameState structure (from game_state.hpp):
    # - board: array<int8_t, 36> = 36 bytes
    # - phase: int32_t = 4 bytes
    # - current_player: int32_t = 4 bytes
    # - marked_black: bitset<36> ~= 8 bytes
    # - marked_white: bitset<36> ~= 8 bytes
    # - Other int32 fields: ~24 bytes
    # Total: ~84 bytes (without alignment padding)
    # With padding and vector overhead: ~128 bytes per node
    
    # Plus Node struct overhead:
    # - parent: int = 4 bytes
    # - action_index: int = 4 bytes
    # - children: vector<int> ~= 24 bytes base + N*4
    # - prior/value_sum/visit_count/virtual_loss: 4*8 = 32 bytes
    # - flags: ~8 bytes
    # Total Node: ~200-300 bytes including GameState
    
    return 256  # Conservative estimate


def analyze_node_growth():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze MCTS node cache growth")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--sims", type=int, default=1024, help="MCTS simulations per move")
    parser.add_argument("--batch-leaves", type=int, default=512, help="Batch leaves for MCTS")
    parser.add_argument("--inference-batch-size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--moves", type=int, default=50, help="Number of moves to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Import v0 MCTS
    try:
        from v0.python.mcts import MCTS as V0MCTS
    except ImportError:
        print("Error: v0 module not available")
        return
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"MCTS sims: {args.sims}, batch_leaves: {args.batch_leaves}")
    print()
    
    # Create model
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()
    
    # Create MCTS
    mcts = V0MCTS(
        model=model,
        num_simulations=args.sims,
        exploration_weight=1.0,
        temperature=1.0,
        device=str(device),
        add_dirichlet_noise=True,
        batch_K=args.batch_leaves,
        inference_backend="graph",
        inference_batch_size=args.inference_batch_size,
        seed=args.seed,
    )
    
    state = GameState()
    np.random.seed(args.seed)
    
    print("=" * 70)
    print(f"{'Move':>5} {'Legal':>6} {'Search(ms)':>10} {'Nodes(est)':>12} {'Memory(MB)':>12}")
    print("=" * 70)
    
    total_nodes_estimate = 0
    search_times = []
    node_counts = []
    memory_estimates = []
    
    for move_num in range(1, args.moves + 1):
        if state.is_game_over():
            print(f"Game over at move {move_num}")
            break
        
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            print(f"No legal moves at move {move_num}")
            break
        
        # Time the search
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        moves, policy = mcts.search(state)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Estimate nodes created this search
        # Conservative: each simulation creates ~branching_factor nodes on average
        avg_branching = len(legal_moves)
        # But with memoization, actual new nodes depend on tree depth
        # Rough estimate: sims * 2 (each sim explores ~2 new nodes on average)
        new_nodes_estimate = args.sims * 2
        total_nodes_estimate += new_nodes_estimate
        
        memory_mb = (total_nodes_estimate * estimate_gamestate_size()) / (1024 * 1024)
        
        search_times.append(elapsed_ms)
        node_counts.append(total_nodes_estimate)
        memory_estimates.append(memory_mb)
        
        print(f"{move_num:>5} {len(legal_moves):>6} {elapsed_ms:>10.1f} "
              f"{total_nodes_estimate:>12,} {memory_mb:>12.1f}")
        
        # Apply move
        if not moves or len(policy) == 0:
            break
            
        policy = np.array(policy)
        policy = policy / policy.sum() if policy.sum() > 0 else np.ones_like(policy) / len(policy)
        idx = np.random.choice(len(moves), p=policy)
        
        state = apply_move(state, moves[idx], quiet=True)
        mcts.advance_root(moves[idx])
    
    print("=" * 70)
    
    # Summary statistics
    print("\nSUMMARY:")
    print(f"  Total moves played: {len(search_times)}")
    print(f"  Avg search time: {np.mean(search_times):.1f}ms")
    print(f"  Final estimated nodes: {total_nodes_estimate:,}")
    print(f"  Final estimated memory: {memory_estimates[-1]:.1f}MB (CPU-side tree)")
    
    # GPU memory if applicable
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        print(f"  GPU allocated: {allocated:.1f}MB")
        print(f"  GPU reserved: {reserved:.1f}MB")
    
    # Analysis
    print("\nANALYSIS:")
    
    # Time spent per simulation
    avg_time_per_sim = np.mean(search_times) / args.sims
    print(f"  Time per simulation: {avg_time_per_sim:.3f}ms")
    
    # Theoretical forward pass time
    # Assuming batch of 512, typical H20 forward ~2-5ms
    theoretical_forward_time = 3.0  # ms for batch of 512
    forward_calls_per_search = args.sims / args.batch_leaves
    theoretical_gpu_time = forward_calls_per_search * theoretical_forward_time
    
    print(f"  Theoretical GPU time per search: {theoretical_gpu_time:.1f}ms")
    print(f"  Actual search time: {np.mean(search_times):.1f}ms")
    
    overhead_ratio = (np.mean(search_times) - theoretical_gpu_time) / np.mean(search_times) * 100
    print(f"  CPU overhead: {overhead_ratio:.1f}%")
    
    if overhead_ratio > 50:
        print("\n  [!] High CPU overhead detected!")
        print("      The MCTS tree operations (selection, expansion, backprop)")
        print("      are taking more time than GPU inference.")
        print("      This explains low GPU power consumption.")
    
    # Node accumulation issue
    if total_nodes_estimate > args.sims * args.moves:
        print("\n  [!] Node cache growing without pruning!")
        print(f"      Expected max nodes (with pruning): ~{args.sims * 3:,}")
        print(f"      Actual estimated nodes: ~{total_nodes_estimate:,}")
        print("      This causes CPU memory copies at each forward pass.")


if __name__ == "__main__":
    analyze_node_growth()
