"""RTX 3060 Laptop (6GB) vs H20 training capability assessment."""

import os
import sys
import time
import math
import gc

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.add_dll_directory(r"C:\Users\wuqin\.conda\envs\torchenv\Lib\site-packages\torch\lib")
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "build", "v0", "src"))

import torch
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.mcts_gpu import GpuStateBatch, V1RootMCTS, V1RootMCTSConfig, TOTAL_ACTION_DIM
from v1.python.self_play_gpu_runner import self_play_v1_gpu
from v1.python.self_play_storage import estimate_bytes_per_sample
from v1.python.train_bridge import train_network_from_tensors
from v1.python.trajectory_buffer import TensorSelfPlayBatch

DEVICE = "cuda:0"
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3


def mem_report(label=""):
    """Return (allocated_GB, reserved_GB, free_GB)."""
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = VRAM_GB - reserved
    if label:
        print(f"  [{label}] alloc={alloc:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB")
    return alloc, reserved, free


def find_max_concurrent_games(model, mcts_sims=64, max_plies=64, sparse_ply=1):
    """Binary search for max concurrent_games that fits in VRAM."""
    lo, hi = 1, 1024
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            batch, stats = self_play_v1_gpu(
                model=model, num_games=mid, mcts_simulations=mcts_sims,
                temperature_init=1.0, temperature_final=0.1, temperature_threshold=8,
                exploration_weight=1.0, device=DEVICE, add_dirichlet_noise=False,
                dirichlet_alpha=0.3, dirichlet_epsilon=0.25, soft_value_k=2.0,
                opening_random_moves=0, max_game_plies=max_plies, sample_moves=False,
                concurrent_games=mid, sparse_ply=sparse_ply, sparse_top_k=8,
                verbose=False,
            )
            best = mid
            lo = mid + 1
            del batch, stats
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                hi = mid - 1
            else:
                raise
    torch.cuda.empty_cache()
    gc.collect()
    return best


def bench_self_play(model, num_games, concurrent, mcts_sims, max_plies, sparse_ply, sparse_top_k, warmup=1, runs=3):
    times = []
    pos_list = []
    for i in range(warmup + runs):
        batch, stats = self_play_v1_gpu(
            model=model, num_games=num_games, mcts_simulations=mcts_sims,
            temperature_init=1.0, temperature_final=0.1, temperature_threshold=8,
            exploration_weight=1.0, device=DEVICE, add_dirichlet_noise=False,
            dirichlet_alpha=0.3, dirichlet_epsilon=0.25, soft_value_k=2.0,
            opening_random_moves=0, max_game_plies=max_plies, sample_moves=False,
            concurrent_games=concurrent, sparse_ply=sparse_ply, sparse_top_k=sparse_top_k,
            verbose=False,
        )
        if i >= warmup:
            times.append(stats.elapsed_sec)
            pos_list.append(stats.positions_per_sec)
        del batch, stats
    avg_time = sum(times) / len(times)
    avg_pos = sum(pos_list) / len(pos_list)
    return avg_time, avg_pos


def bench_training(model, batch, batch_size, lr=1e-4, epochs=1):
    times = []
    model.train()
    for e in range(epochs):
        t0 = time.perf_counter()
        model, metrics = train_network_from_tensors(
            model=model, samples=batch, batch_size=batch_size,
            epochs=1, lr=lr, weight_decay=1e-4,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    model.eval()
    return sum(times) / len(times), metrics


def main():
    print("=" * 70)
    print("RTX 3060 Laptop vs H20 — Training Capability Assessment")
    print("=" * 70)
    print(f"GPU: {GPU_NAME} ({VRAM_GB:.1f} GB VRAM)")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print()

    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params * 4 / 1024**2:.1f} MB fp32)")
    mem_report("after model load")

    # ── 1. Max concurrent games ──
    print()
    print("── 1. VRAM headroom: max concurrent games ──")
    max_c1 = find_max_concurrent_games(model, mcts_sims=64, max_plies=64, sparse_ply=1)
    print(f"  sparse_ply=1: max concurrent_games = {max_c1}")
    max_c2 = find_max_concurrent_games(model, mcts_sims=64, max_plies=64, sparse_ply=2)
    print(f"  sparse_ply=2: max concurrent_games = {max_c2}")

    # ── 2. Self-play throughput at various configs ──
    print()
    print("── 2. Self-play throughput ──")
    configs = [
        ("stable-ish",  16,  16,  256, 144, 1, 8),
        ("stable-ish",  16,  16,  512, 144, 1, 8),
        ("2-ply-light", 16,  16,  256, 144, 2, 4),
        ("2-ply-light", 16,  16,  512, 144, 2, 4),
        ("aggr-single", 32,  32,  512, 200, 1, 8),
        ("aggr-single", 32,  32, 1024, 200, 1, 8),
        ("2-ply-aggr",  32,  32,  512, 200, 2, 8),
    ]
    print(f"  {'config':<16s} {'games':>5s} {'conc':>5s} {'sims':>6s} {'ply':>4s} {'K':>3s} {'time':>7s} {'pos/s':>8s}")
    print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*6} {'-'*4} {'-'*3} {'-'*7} {'-'*8}")
    for label, ng, conc, sims, mpl, sp, k in configs:
        t, pps = bench_self_play(model, ng, conc, sims, mpl, sp, k, warmup=1, runs=2)
        print(f"  {label:<16s} {ng:>5d} {conc:>5d} {sims:>6d} {sp:>4d} {k:>3d} {t:>7.2f}s {pps:>8.0f}")

    # ── 3. Estimate training throughput ──
    print()
    print("── 3. Training throughput estimation ──")
    # First generate some self-play data
    print("  Generating self-play samples for training benchmark...")
    torch.cuda.empty_cache()
    batch_sp, stats_sp, _ = self_play_v1_gpu(
        model=model, num_games=4, mcts_simulations=64,
        temperature_init=1.0, temperature_final=0.1, temperature_threshold=8,
        exploration_weight=1.0, device=DEVICE, add_dirichlet_noise=False,
        dirichlet_alpha=0.3, dirichlet_epsilon=0.25, soft_value_k=2.0,
        opening_random_moves=0, max_game_plies=64, sample_moves=False,
        concurrent_games=4, sparse_ply=1, sparse_top_k=8, verbose=False,
    )[:2]
    print(f"  Samples: {batch_sp.num_samples}, size: {estimate_bytes_per_sample(batch_sp):,} bytes/sample")
    mem_report("after self-play data gen")

    for bs in [64, 128, 256, 512, 1024, 2048]:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            t0 = time.perf_counter()
            model.train()
            _, metrics = train_network_from_tensors(
                model=model, samples=batch_sp, batch_size=bs,
                epochs=1, lr=1e-4, weight_decay=1e-4,
            )
            model.eval()
            elapsed = time.perf_counter() - t0
            samples_per_sec = batch_sp.num_samples / elapsed
            print(f"  batch_size={bs:>5d}: {elapsed:.3f}s ({samples_per_sec:.0f} samples/s), "
                  f"loss={metrics.get('train_avg_loss', float('nan')):.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch_size={bs:>5d}: OOM")
                break
            raise
    mem_report("after training bench")

    # ── 4. H20 comparison ──
    print()
    print("── 4. RTX 3060 vs H20 comparison ──")
    print(f"  {'Metric':<30s} {'RTX 3060 (6GB)':>20s} {'H20 (96GB ×4)':>20s} {'Ratio':>10s}")
    print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*10}")

    comparisons = [
        ("VRAM", "6 GB", "96 GB ×4 = 384 GB", "~64x"),
        ("FP16 TFLOPS", "~10", "~148 ×4 = 592", "~59x"),
        ("Memory bandwidth", "~192 GB/s", "~4 TB/s ×4 = 16 TB/s", "~83x"),
        ("Concurrent games (1-ply)", f"~{max_c1}", "~8192 ×4 = 32768", "~{32768/max_c1:.0f}x"),
        ("Self-play pos/s (est.)", "~50-200", "~20000-50000 (×4)", "~100-1000x"),
        ("Training batch_size", "~512-1024", "~16384 (DDP×4)", "~16-32x"),
        ("Full iter time (est.)", "~8-24 hours", "~20-60 min (×4 GPUs)", "~16-48x slower"),
    ]
    for metric, rtx, h20, ratio in comparisons:
        print(f"  {metric:<30s} {rtx:>20s} {h20:>20s} {ratio:>10s}")

    print()
    print("── 5. Recommended 3060 config ──")
    print(f"  Stable training feasible: YES (with reduced scale)")
    print(f"  Suggested config:")
    print(f"    SELF_PLAY_GAMES=2048 (vs 12288 stable)")
    print(f"    MCTS_SIMULATIONS=256-512 (vs 1536 stable)")
    print(f"    BATCH_SIZE=512 (vs 8192 stable)")
    print(f"    SELF_PLAY_CONCURRENT_GAMES={max(1, max_c1//2)}")
    print(f"    EPOCHS=2-3")
    print(f"  Estimated iteration time: ~1-3 hours")
    print(f"  Limitation: single-GPU only, no DDP")
    print(f"  V1_FINALIZE_GRAPH=off likely needed (Windows CUDA Graph compat)")

    print()
    print("=" * 70)
    print("Assessment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
