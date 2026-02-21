import os
import sys
root = os.path.abspath(os.getcwd())
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'build', 'v0', 'src'))
sys.path.insert(0, os.path.join(root, 'v0', 'build', 'src'))

import torch
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from src.game_state import GameState
from v1.python.self_play_gpu_runner import self_play_v1_gpu

def run(mcts_simulations: int):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(17 + mcts_simulations)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17 + mcts_simulations)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS).to(device)
    model.eval()
    _samples, stats = self_play_v1_gpu(
        model=model,
        num_games=32,
        mcts_simulations=int(mcts_simulations),
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=10,
        exploration_weight=1.0,
        device=device,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        opening_random_moves=8,
        max_game_plies=128,
        sample_moves=True,
        concurrent_games=8,
        verbose=False,
    )
    decisive = int(stats.black_wins + stats.white_wins)
    draw = int(stats.draws)
    total = int(stats.num_games)
    abs_sum = 0
    for k, v in stats.piece_delta_buckets.items():
        abs_sum += abs(int(k)) * int(v)
    mean_abs_delta = abs_sum / max(1, total)
    print(
        f'mcts_simulations={mcts_simulations:3d} decisive={decisive:2d}/{total} '
        f'draw={draw:2d}/{total} mean_abs_delta={mean_abs_delta:.3f}'
    )

if __name__ == '__main__':
    print('device=' + ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    for sims in [8, 16, 32, 64]:
        run(sims)
