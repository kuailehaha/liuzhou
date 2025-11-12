import random
import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet
from v0.python.mcts import MCTS as V0MCTS
from src.mcts import MCTS as LegacyMCTS

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

rng = random.Random(seed + 1)
state = GameState()
for _ in range(rng.randint(0, 60)):
    legal = generate_all_legal_moves(state)
    if not legal:
        break
    move = rng.choice(legal)
    state = apply_move(state, move, quiet=True)
    if state.is_game_over():
        break

model = ChessNet(board_size=GameState.BOARD_SIZE)
model.eval()

v0 = V0MCTS(
    model=model,
    num_simulations=1,
    exploration_weight=1.0,
    temperature=1.0,
    device='cpu',
    add_dirichlet_noise=False,
    seed=seed,
    batch_K=8,
)
legacy = LegacyMCTS(
    model=model,
    num_simulations=1,
    exploration_weight=1.0,
    temperature=1.0,
    device='cpu',
    add_dirichlet_noise=False,
    virtual_loss_weight=0.0,
    batch_K=8,
    verbose=False,
)

v0_moves, v0_probs = v0.search(state)
legacy_moves, legacy_probs = legacy.search(state)
print('len', len(v0_moves), len(legacy_moves))
print('v0', v0_probs)
print('legacy', legacy_probs)
print('max diff', max(abs(float(a)-float(b)) for a,b in zip(v0_probs, legacy_probs)))
