import torch
import v0_core
from src.game_state import GameState
from v0.python.state_batch import from_game_states
from v0.python.move_encoder import DEFAULT_ACTION_SPEC

state = GameState()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch = from_game_states([state], device=device)
mask, meta = v0_core.encode_actions_fast(
    batch.board,
    batch.marks_black,
    batch.marks_white,
    batch.phase,
    batch.current_player,
    batch.pending_marks_required,
    batch.pending_marks_remaining,
    batch.pending_captures_required,
    batch.pending_captures_remaining,
    batch.forced_removals_done,
    DEFAULT_ACTION_SPEC.placement_dim,
    DEFAULT_ACTION_SPEC.movement_dim,
    DEFAULT_ACTION_SPEC.selection_dim,
    DEFAULT_ACTION_SPEC.auxiliary_dim,
)
legal_idx = torch.nonzero(mask[0]).view(-1)
action_codes = meta[0, legal_idx[:4]].to(torch.int32)
parent = torch.zeros(action_codes.size(0), dtype=torch.int64, device=device)
out = v0_core.batch_apply_moves(
    batch.board,
    batch.marks_black,
    batch.marks_white,
    batch.phase,
    batch.current_player,
    batch.pending_marks_required,
    batch.pending_marks_remaining,
    batch.pending_captures_required,
    batch.pending_captures_remaining,
    batch.forced_removals_done,
    batch.move_count,
    action_codes,
    parent,
)
print('ok', len(out), out[0].shape, out[0].device)
