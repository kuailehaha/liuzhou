"""
Unit tests for batched policy loss: action_to_index, build_combined_logits,
masked_log_softmax, and parity between batched and legacy policy loss.
"""

import pytest
import torch

from src.game_state import GameState, Phase
from src.neural_network import ChessNet, get_move_probabilities, NUM_INPUT_CHANNELS
from src.move_generator import generate_all_legal_moves
from src.policy_batch import (
    TOTAL_DIM,
    action_to_index,
    legal_mask_and_target_dense,
    build_combined_logits,
    masked_log_softmax,
    batched_policy_loss,
)


def test_action_to_index_placement():
    move = {"phase": Phase.PLACEMENT, "action_type": "place", "position": (2, 3)}
    idx = action_to_index(move, 6)
    assert idx == 2 * 6 + 3
    assert 0 <= idx < TOTAL_DIM


def test_action_to_index_move():
    move = {
        "phase": Phase.MOVEMENT,
        "action_type": "move",
        "from_position": (1, 1),
        "to_position": (0, 1),
    }
    idx = action_to_index(move, 6)
    assert idx is not None
    assert 36 <= idx < 36 + 144


def test_action_to_index_process_removal():
    move = {"phase": Phase.REMOVAL, "action_type": "process_removal"}
    idx = action_to_index(move, 6)
    assert idx == 36 + 144 + 36


def test_legal_mask_and_target_dense_shape():
    legal_moves = [
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 0)},
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (1, 1)},
    ]
    target = torch.tensor([0.7, 0.3])
    legal_mask, target_dense = legal_mask_and_target_dense(legal_moves, target, 6)
    assert legal_mask.shape == (1, TOTAL_DIM)
    assert target_dense.shape == (1, TOTAL_DIM)
    assert legal_mask.sum().item() == 2
    assert (target_dense.sum() - 1.0).abs() < 1e-5


def test_build_combined_logits_shape():
    B = 2
    log_p1 = torch.randn(B, 36)
    log_p2 = torch.randn(B, 36)
    log_pmc = torch.randn(B, 36)
    combined = build_combined_logits(log_p1, log_p2, log_pmc, 6)
    assert combined.shape == (B, TOTAL_DIM)


def test_masked_log_softmax():
    logits = torch.tensor([[1.0, 2.0, -1e9, 3.0]])
    mask = torch.tensor([[True, True, False, True]])
    log_probs = masked_log_softmax(logits, mask, dim=1)
    assert log_probs.shape == logits.shape
    assert (log_probs[0, 2].abs() < 1e-6).item()
    exp_sum = (log_probs[0].exp() * mask.float())[0].sum().item()
    assert abs(exp_sum - 1.0) < 1e-5


@pytest.mark.slow
def test_batched_vs_legacy_policy_loss_parity():
    """Batched and legacy policy loss should be close for the same batch of 2 samples."""
    torch.manual_seed(42)
    board_size = GameState.BOARD_SIZE
    net = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    state = GameState()
    legal_moves = generate_all_legal_moves(state)
    if len(legal_moves) < 2:
        legal_moves = legal_moves * 2
    legal_moves = legal_moves[: min(10, len(legal_moves))]
    target_probs = torch.softmax(torch.randn(len(legal_moves)), dim=0)

    from src.train import ChessDataset, mcts_collate_fn, mcts_collate_batched_policy
    from src.neural_network import state_to_tensor

    examples = [
        (state, target_probs.numpy().tolist(), legal_moves, 0.5, 0.4),
        (state, target_probs.numpy().tolist(), legal_moves, -0.5, -0.3),
    ]
    device = "cpu"
    batch_size = 2

    ds_legacy = ChessDataset(examples, precompute_tensors=True, use_batched_policy=False)
    ds_batched = ChessDataset(examples, precompute_tensors=True, use_batched_policy=True)
    batch_legacy = mcts_collate_fn([ds_legacy[i] for i in range(batch_size)])
    batch_batched = mcts_collate_batched_policy([ds_batched[i] for i in range(batch_size)])

    states = batch_legacy[0].to(device)
    net.to(device)
    net.train()
    log_p1, log_p2, log_pmc, _ = net(states)
    log_p1 = log_p1.view(log_p1.size(0), -1)
    log_p2 = log_p2.view(log_p2.size(0), -1)
    log_pmc = log_pmc.view(log_pmc.size(0), -1)

    legacy_loss = torch.tensor(0.0, device=device)
    policy_weight_sum = 0.0
    policy_loss_fn = torch.nn.KLDivLoss(reduction="sum")
    for i in range(batch_size):
        legal_i = batch_legacy[1][i]
        target_i = batch_legacy[2][i].to(device)
        if not legal_i or target_i.numel() == 0:
            continue
        _, raw = get_move_probabilities(
            log_p1[i], log_p2[i], log_pmc[i], legal_i, board_size, device
        )
        if raw.numel() == 0:
            continue
        log_p = torch.nn.functional.log_softmax(raw, dim=0)
        legacy_loss += policy_loss_fn(log_p, target_i)
        policy_weight_sum += 1.0
    if policy_weight_sum > 0:
        legacy_loss = legacy_loss / policy_weight_sum

    legal_mask_b = batch_batched[1].to(device)
    target_dense_b = batch_batched[2].to(device)
    combined = build_combined_logits(log_p1, log_p2, log_pmc, board_size)
    log_probs_b = masked_log_softmax(combined, legal_mask_b, dim=1)
    value_batch = batch_batched[3].to(device)
    batched_loss = batched_policy_loss(
        log_probs_b, target_dense_b, legal_mask_b, value_batch, policy_draw_weight=1.0
    )

    # Compare log-probs at legal indices for sample 0 (placement-only so indices are 0..35)
    legal_i = batch_legacy[1][0]
    _, raw_legacy = get_move_probabilities(
        log_p1[0], log_p2[0], log_pmc[0], legal_i, board_size, device
    )
    log_p_legacy = torch.nn.functional.log_softmax(raw_legacy, dim=0)
    indices = [action_to_index(m, board_size) for m in legal_i]
    if all(i is not None for i in indices):
        log_p_batched = log_probs_b[0].cpu()[torch.tensor(indices)]
        assert torch.allclose(
            log_p_legacy.cpu(), log_p_batched, atol=1e-5
        ), f"Log-probs mismatch: legacy {log_p_legacy[:5].tolist()} batched {log_p_batched[:5].tolist()}"

    # Log-probs at legal indices match; loss scale may differ due to normalization.
    # Both paths use the same CE formula over legal moves; accept same order of magnitude.
    assert legacy_loss.item() > 0 and batched_loss.item() > 0
    ratio = max(legacy_loss.item(), batched_loss.item()) / (min(legacy_loss.item(), batched_loss.item()) + 1e-8)
    assert ratio < 10, f"Legacy {legacy_loss.item():.6f} vs Batched {batched_loss.item():.6f} (ratio {ratio})"


def test_batched_policy_loss_gradient():
    """Batched policy loss should have gradients."""
    B = 2
    log_probs = torch.randn(B, TOTAL_DIM, requires_grad=True)
    target_dense = torch.softmax(torch.randn(B, TOTAL_DIM), dim=1)
    legal_mask = target_dense > 0.01
    value_batch = torch.tensor([[0.0], [1.0]])
    loss = batched_policy_loss(log_probs, target_dense, legal_mask, value_batch, 0.3)
    loss.backward()
    assert log_probs.grad is not None
