import random
import numpy as np
import torch
from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.mcts import MCTS as LegacyMCTS, move_to_key
from src.neural_network import ChessNet
from v0.python.mcts import MCTS as V0MCTS

def random_state(max_moves, rng):
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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

rng = random.Random(seed + 1)
state = random_state(60, rng)

model = ChessNet(board_size=GameState.BOARD_SIZE)
model.eval()

legacy = LegacyMCTS(
    model=model,
    num_simulations=64,
    exploration_weight=1.0,
    temperature=1.0,
    device="cpu",
    add_dirichlet_noise=False,
    virtual_loss_weight=0.0,
    batch_K=8,
    verbose=False,
)

v0 = V0MCTS(
    model=model,
    num_simulations=64,
    exploration_weight=1.0,
    temperature=1.0,
    device="cpu",
    add_dirichlet_noise=False,
    seed=seed,
    batch_K=8,
)

legacy_moves, legacy_probs = legacy.search(state)
v0_moves, v0_probs = v0.search(state)

legacy_map = {move_to_key(move): float(prob) for move, prob in zip(legacy_moves, legacy_probs)}
v0_map = {move_to_key(move): float(prob) for move, prob in zip(v0_moves, v0_probs)}

common_keys = sorted(legacy_map.keys())
diffs = [(key, legacy_map[key], v0_map.get(key)) for key in common_keys]
max_diff = max(abs(a - (b if b is not None else 0.0)) for key, a, b in diffs)
print(f"moves: {len(common_keys)} max_diff={max_diff}")
for key, a, b in diffs:
    delta = None if b is None else a - b
    if delta is None or abs(delta) > 1e-6:
        print(key, a, b, delta)
