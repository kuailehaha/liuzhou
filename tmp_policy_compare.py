import random
import numpy as np
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, get_move_probabilities, state_to_tensor
from v1.game.move_encoder import DEFAULT_ACTION_SPEC, action_to_index
from v1.game.state_batch import from_game_states
from v1.game.fast_legal_mask import encode_actions_fast
import v0_core

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

inp = state_to_tensor(state, state.current_player)
log_p1, log_p2, log_pmc, _ = model(inp)

batch = from_game_states([state], torch.device('cpu'))
mask, metadata = encode_actions_fast(batch, DEFAULT_ACTION_SPEC, return_metadata=True)
probs, _ = v0_core.project_policy_logits_fast(
    log_p1,
    log_p2,
    log_pmc,
    mask,
    DEFAULT_ACTION_SPEC.placement_dim,
    DEFAULT_ACTION_SPEC.movement_dim,
    DEFAULT_ACTION_SPEC.selection_dim,
    DEFAULT_ACTION_SPEC.auxiliary_dim,
)
probs = probs.squeeze(0)

legal_moves = generate_all_legal_moves(state)
python_probs, _ = get_move_probabilities(
    log_p1[0], log_p2[0], log_pmc[0], legal_moves, state.BOARD_SIZE
)

cpp_probs = []
for move in legal_moves:
    idx = action_to_index(move, state.BOARD_SIZE, DEFAULT_ACTION_SPEC)
    cpp_probs.append(float(probs[idx]))

print('len legal', len(legal_moves))
print('python probs', python_probs)
print('cpp probs', cpp_probs)
print('diffs', [abs(a-b) for a,b in zip(python_probs, cpp_probs)])
