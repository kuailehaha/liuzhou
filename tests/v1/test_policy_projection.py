import pytest

torch = pytest.importorskip("torch")

from src.game_state import GameState, Phase, Player
from src.move_generator import generate_all_legal_moves
from src.neural_network import (
    ChessNet,
    NUM_INPUT_CHANNELS,
    get_move_probabilities,
    state_to_tensor,
)

from v1.game.move_encoder import DEFAULT_ACTION_SPEC, action_to_index, encode_actions
from v1.game.state_batch import from_game_states
from v1.net.encoding import project_policy_logits, states_to_model_input


def _movement_state_multiple() -> GameState:
    state = GameState()
    state.phase = Phase.MOVEMENT
    state.current_player = Player.BLACK
    state.board[2][2] = Player.BLACK.value
    state.board[2][3] = 0
    state.board[2][1] = 0
    state.board[3][2] = 0
    state.board[1][2] = 0
    return state


def _forced_removal_single() -> GameState:
    state = GameState()
    state.phase = Phase.FORCED_REMOVAL
    state.current_player = Player.WHITE
    state.forced_removals_done = 0
    state.board = [[0] * state.BOARD_SIZE for _ in range(state.BOARD_SIZE)]
    state.board[0][0] = Player.BLACK.value
    state.board[5][5] = Player.WHITE.value
    return state


def test_project_policy_logits_matches_legacy_multi_move():
    state = _movement_state_multiple()
    batch = from_game_states([state])
    legal_mask = encode_actions(batch, DEFAULT_ACTION_SPEC)

    torch.manual_seed(1234)
    log_p1 = torch.randn(1, state.BOARD_SIZE * state.BOARD_SIZE)
    log_p2 = torch.randn_like(log_p1)
    log_pmc = torch.randn_like(log_p1)

    probs, masked_logits = project_policy_logits(
        (log_p1, log_p2, log_pmc), legal_mask, DEFAULT_ACTION_SPEC
    )

    legal_moves = generate_all_legal_moves(state)
    legacy_probs, legacy_raw = get_move_probabilities(
        log_p1.squeeze(0),
        log_p2.squeeze(0),
        log_pmc.squeeze(0),
        legal_moves,
        board_size=state.BOARD_SIZE,
        device=str(log_p1.device),
    )
    legacy_probs = torch.tensor(legacy_probs, dtype=probs.dtype)

    for move_idx, move in enumerate(legal_moves):
        flat_index = action_to_index(move, state.BOARD_SIZE, DEFAULT_ACTION_SPEC)
        assert flat_index is not None
        torch.testing.assert_close(
            probs[0, flat_index], legacy_probs[move_idx], rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            masked_logits[0, flat_index], legacy_raw[move_idx], rtol=0, atol=1e-6
        )

    total_prob = probs[0, legal_mask.squeeze(0)].sum()
    torch.testing.assert_close(total_prob, torch.tensor(1.0, dtype=probs.dtype))


def test_project_policy_logits_single_move_forced_removal():
    state = _forced_removal_single()
    batch = from_game_states([state])
    legal_mask = encode_actions(batch, DEFAULT_ACTION_SPEC)

    log_shape = (1, state.BOARD_SIZE * state.BOARD_SIZE)
    log_p1 = torch.zeros(log_shape)
    log_p2 = torch.zeros(log_shape)
    log_pmc = torch.zeros(log_shape)

    probs, masked_logits = project_policy_logits(
        (log_p1, log_p2, log_pmc), legal_mask, DEFAULT_ACTION_SPEC
    )

    legal_moves = generate_all_legal_moves(state)
    assert len(legal_moves) == 1
    flat_index = action_to_index(legal_moves[0], state.BOARD_SIZE, DEFAULT_ACTION_SPEC)
    assert flat_index is not None

    torch.testing.assert_close(probs[0, flat_index], torch.tensor(1.0))
    assert masked_logits[0, flat_index].isfinite()
    off_indices = (~legal_mask.squeeze(0)).nonzero(as_tuple=False).view(-1)
    assert torch.all(~torch.isfinite(masked_logits[0, off_indices]))


def test_project_policy_logits_handles_no_legal_actions():
    board_size = GameState.BOARD_SIZE
    cells = board_size * board_size
    log_p1 = torch.randn(1, cells)
    log_p2 = torch.randn(1, cells)
    log_pmc = torch.randn(1, cells)

    spec = DEFAULT_ACTION_SPEC
    legal_mask = torch.zeros((1, spec.total_dim), dtype=torch.bool)

    probs, masked_logits = project_policy_logits((log_p1, log_p2, log_pmc), legal_mask, spec)

    assert torch.all(probs == 0)
    assert torch.all(~torch.isfinite(masked_logits))


def test_end_to_end_policy_alignment_with_network():
    torch.manual_seed(2025)
    states = [
        _movement_state_multiple(),
        _forced_removal_single(),
    ]
    device = torch.device("cpu")
    net = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS).to(device)
    net.eval()

    batch = from_game_states(states, device=device)
    enc_inputs = states_to_model_input(batch)
    log_p1_new, log_p2_new, log_pmc_new, _ = net(enc_inputs)
    legal_mask = encode_actions(batch, DEFAULT_ACTION_SPEC)

    probs_new, logits_new = project_policy_logits(
        (log_p1_new, log_p2_new, log_pmc_new),
        legal_mask,
        DEFAULT_ACTION_SPEC,
    )

    for idx, state in enumerate(states):
        legacy_input = state_to_tensor(state, state.current_player).to(device)
        log_p1_old, log_p2_old, log_pmc_old, _ = net(legacy_input)
        legal_moves = generate_all_legal_moves(state)

        legacy_probs, legacy_raw = get_move_probabilities(
            log_p1_old.squeeze(0),
            log_p2_old.squeeze(0),
            log_pmc_old.squeeze(0),
            legal_moves,
            board_size=state.BOARD_SIZE,
            device=str(device),
        )
        legacy_probs_tensor = torch.tensor(legacy_probs, dtype=probs_new.dtype, device=device)
        legacy_raw_tensor = legacy_raw.to(device)

        for mv_idx, move in enumerate(legal_moves):
            flat_index = action_to_index(move, state.BOARD_SIZE, DEFAULT_ACTION_SPEC)
            assert flat_index is not None
            torch.testing.assert_close(
                probs_new[idx, flat_index],
                legacy_probs_tensor[mv_idx],
                atol=1e-6,
                rtol=0,
            )
            torch.testing.assert_close(
                logits_new[idx, flat_index],
                legacy_raw_tensor[mv_idx],
                atol=1e-6,
                rtol=0,
            )
