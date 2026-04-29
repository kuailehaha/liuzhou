"""Multi-ply search verification experiment.

Validates:
1. Root-only vs multi-ply produce different policy targets
2. Throughput comparison (positions/sec)
3. Determinism: same seed → same output
4. Leaf value refinement: 2-ply sees deeper tactics
"""

import os
import sys
import time
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.add_dll_directory(r"C:\Users\wuqin\.conda\envs\torchenv\Lib\site-packages\torch\lib")
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "build", "v0", "src"))

import torch
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.mcts_gpu import GpuStateBatch, V1RootMCTS, V1RootMCTSConfig
from v1.python.self_play_gpu_runner import self_play_v1_gpu

DEVICE = "cuda:0"
NUM_WARMUP = 1
NUM_BENCH = 3
NUM_GAMES = 4
MCTS_SIMS = 256

def setup():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(DEVICE)
    model.eval()
    return model

def run_self_play(model, sparse_ply, sparse_top_k, num_games, mcts_sims, seed):
    """Run self-play and return batch, stats, elapsed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    t0 = time.perf_counter()
    batch, stats = self_play_v1_gpu(
        model=model,
        num_games=num_games,
        mcts_simulations=mcts_sims,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=8,
        exploration_weight=1.0,
        device=DEVICE,
        add_dirichlet_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        opening_random_moves=0,
        max_game_plies=64,
        sample_moves=False,
        concurrent_games=num_games,
        sparse_ply=sparse_ply,
        sparse_top_k=sparse_top_k,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return batch, stats, elapsed

def main():
    print("=" * 60)
    print("V1 Multi-Ply Search Verification")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    model = setup()

    # ── Experiment 1: Determinism ──
    print("── Exp 1: Determinism check ──")
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    b1, s1, _ = run_self_play(model, sparse_ply=2, sparse_top_k=4, num_games=2, mcts_sims=64, seed=123)
    b2, s2, _ = run_self_play(model, sparse_ply=2, sparse_top_k=4, num_games=2, mcts_sims=64, seed=123)

    policy_match = torch.allclose(b1.policy_targets, b2.policy_targets)
    value_match = torch.allclose(b1.value_targets, b2.value_targets)
    print(f"  Policy targets match: {policy_match}")
    print(f"  Value targets match:  {value_match}")
    print(f"  Determinism: {'OK' if (policy_match and value_match) else 'FAIL'}")
    print()

    # ── Experiment 2: Root-only vs Multi-ply produce different results ──
    print("── Exp 2: Root-only vs 2-ply comparison ──")
    b_root, s_root, t_root = run_self_play(model, sparse_ply=1, sparse_top_k=8, num_games=NUM_GAMES, mcts_sims=MCTS_SIMS, seed=99)
    b_multi, s_multi, t_multi = run_self_play(model, sparse_ply=2, sparse_top_k=8, num_games=NUM_GAMES, mcts_sims=MCTS_SIMS, seed=99)

    policy_diff = (b_root.policy_targets - b_multi.policy_targets).abs().max().item()
    policy_same_frac = (b_root.policy_targets.argmax(dim=1) == b_multi.policy_targets.argmax(dim=1)).float().mean().item()
    value_diff = (b_root.value_targets - b_multi.value_targets).abs().max().item()

    print(f"  Policy max abs diff:  {policy_diff:.6f}")
    print(f"  Policy top-1 agreement: {policy_same_frac*100:.1f}%")
    print(f"  Value max abs diff:   {value_diff:.6f}")
    print(f"  Root-only positions:  {s_root.num_positions} in {t_root:.2f}s ({s_root.positions_per_sec:.0f} pos/s)")
    print(f"  2-ply positions:      {s_multi.num_positions} in {t_multi:.2f}s ({s_multi.positions_per_sec:.0f} pos/s)")
    if t_root > 0 and t_multi > 0:
        print(f"  Throughput ratio:     {s_multi.positions_per_sec / max(1, s_root.positions_per_sec):.2f}x")
    print(f"  Produce different policies: {'OK' if policy_diff > 1e-6 else 'SAME (unexpected if 2-ply changes anything)'}")
    print()

    # ── Experiment 3: Throughput benchmark ──
    print("── Exp 3: Throughput benchmark ──")
    configs = [
        ("1-ply (root-only)", 1, 8),
        ("2-ply K=4", 2, 4),
        ("2-ply K=8", 2, 8),
        ("3-ply K=4", 3, 4),
    ]
    results = []
    for name, ply, k in configs:
        times = []
        pos = []
        for i in range(NUM_WARMUP + NUM_BENCH):
            _, stats, elapsed = run_self_play(model, sparse_ply=ply, sparse_top_k=k, num_games=2, mcts_sims=128, seed=42 + i)
            if i >= NUM_WARMUP:
                times.append(elapsed)
                pos.append(stats.positions_per_sec)
        avg_time = sum(times) / len(times)
        avg_pos = sum(pos) / len(pos)
        results.append((name, avg_time, avg_pos))
        print(f"  {name:20s}: {avg_time:.3f}s (2 games), {avg_pos:.0f} pos/s")
    if results:
        baseline = results[0][2]
        for name, avg_time, avg_pos in results[1:]:
            ratio = avg_pos / baseline
            print(f"    vs baseline: {ratio:.2f}x")
    print()

    # ── Experiment 4: Game outcome comparison ──
    print("── Exp 4: Game outcome stats ──")
    for name, ply, k in [("1-ply", 1, 8), ("2-ply K=8", 2, 8), ("3-ply K=4", 3, 4)]:
        _, stats, _ = run_self_play(model, sparse_ply=ply, sparse_top_k=k, num_games=16, mcts_sims=MCTS_SIMS, seed=0)
        total = stats.black_wins + stats.white_wins + stats.draws
        print(f"  {name:15s}: W/L/D = {stats.black_wins}/{stats.white_wins}/{stats.draws} "
              f"(decisive={stats.black_wins + stats.white_wins}/{total}), "
              f"avg_len={stats.avg_game_length:.1f}ply")
    print()

    # ── Experiment 5: Value target distribution comparison ──
    print("── Exp 5: Value target distribution ──")
    b1, _, _ = run_self_play(model, sparse_ply=1, sparse_top_k=8, num_games=8, mcts_sims=MCTS_SIMS, seed=7)
    b2, _, _ = run_self_play(model, sparse_ply=2, sparse_top_k=8, num_games=8, mcts_sims=MCTS_SIMS, seed=7)
    for label, batch in [("1-ply", b1), ("2-ply", b2)]:
        v = batch.value_targets
        print(f"  {label}: value mean={v.mean().item():.4f}, std={v.std().item():.4f}, "
              f"min={v.min().item():.4f}, max={v.max().item():.4f}, "
              f"nonzero_frac={(v.abs() > 1e-6).float().mean().item()*100:.1f}%")
    print()

    # ── Experiment 6: Policy entropy comparison ──
    print("── Exp 6: Policy entropy ──")
    for name, ply, k in [("1-ply", 1, 8), ("2-ply K=8", 2, 8), ("3-ply K=4", 3, 4)]:
        b, _, _ = run_self_play(model, sparse_ply=ply, sparse_top_k=k, num_games=8, mcts_sims=MCTS_SIMS, seed=3)
        p = b.policy_targets.clamp_min(1e-9)
        entropy = -(p * p.log()).sum(dim=1).mean().item()
        print(f"  {name:15s}: mean policy entropy = {entropy:.4f}")
    print()

    print("=" * 60)
    print("Verification complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
