import torch
import v0_core
from src.game_state import GameState
from v0.python.state_batch import from_game_states
from v0.python.move_encoder import DEFAULT_ACTION_SPEC

print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('v0_core', v0_core.version())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

state = GameState()
batch = from_game_states([state], device=device)

x = v0_core.states_to_model_input(batch.board, batch.marks_black, batch.marks_white, batch.phase, batch.current_player)
print('input', tuple(x.shape), x.device)

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
print('mask', tuple(mask.shape), mask.device, mask.dtype)
print('meta', tuple(meta.shape), meta.device, meta.dtype)
legal_idx = torch.nonzero(mask[0]).view(-1)
print('legal_count', int(legal_idx.numel()))

if legal_idx.numel() > 0:
    action_codes = meta[0, legal_idx[:min(4, legal_idx.numel())]].to(torch.int32)
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
        torch.zeros_like(batch.move_count),
        action_codes,
        parent,
    )
    print('apply_out_len', len(out), 'board_shape', tuple(out[0].shape), 'device', out[0].device)

