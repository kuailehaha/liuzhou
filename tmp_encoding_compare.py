import random
import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import state_to_tensor
from v1.game.state_batch import from_game_states
import v0_core

seed=42
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

python_tensor = state_to_tensor(state, state.current_player)

batch = from_game_states([state], torch.device('cpu'))
cpp_tensor = v0_core.states_to_model_input(
    batch.board,
    batch.marks_black,
    batch.marks_white,
    batch.phase,
    batch.current_player,
)

diff = (python_tensor - cpp_tensor).abs().max().item()
print('max diff', diff)
print('python shape', python_tensor.shape, 'cpp shape', cpp_tensor.shape)
