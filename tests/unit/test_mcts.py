"""
Pytest tests for legacy MCTS expansion/evaluation behavior.

Usage:
  pytest tests/unit/test_mcts.py -q
"""

import torch
import pytest

from src.mcts import MCTS, MCTSNode
from src.game_state import GameState, Phase


class DummyModel:
    def to(self, device):
        pass

    def eval(self):
        pass

    def __call__(self, x):
        board_area = x.shape[-1] * x.shape[-2]
        log_prob = torch.log(torch.ones(board_area) / board_area)
        return (
            log_prob.unsqueeze(0),
            log_prob.unsqueeze(0),
            log_prob.unsqueeze(0),
            torch.tensor([[0.2]]),
        )


def _dummy_state_to_tensor(state, player_to_act):
    return torch.zeros(1, 1, state.BOARD_SIZE, state.BOARD_SIZE)


def _dummy_get_move_probabilities(log_p1, log_p2, log_pmc, legal_moves, board_size, device):
    num = len(legal_moves)
    if num == 0:
        return [], torch.empty(0)
    return [1.0 / num for _ in legal_moves], torch.zeros(num)


def _dummy_generate_no_moves(state):
    return []


def _dummy_generate_one_move(state):
    return [{"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 0)}]


def _dummy_apply_move(state, move, quiet=False):
    new_state = state.copy()
    new_state.board[0][0] = state.current_player.value
    new_state.switch_player()
    return new_state


def test_terminal_value_backpropagated(monkeypatch: pytest.MonkeyPatch):
    mcts = MCTS(DummyModel(), num_simulations=1)
    parent = MCTSNode(GameState())
    child = MCTSNode(parent.state.copy(), parent=parent)

    monkeypatch.setattr("src.mcts.generate_all_legal_moves", _dummy_generate_no_moves)
    val = mcts._expand_and_evaluate(child)

    assert child.is_terminal()
    assert val == -1.0
    child.backpropagate(child.terminal_value)
    assert parent.visit_count == 1
    assert parent.value_sum == 1.0


def test_expand_does_not_increment_counts(monkeypatch: pytest.MonkeyPatch):
    mcts = MCTS(DummyModel(), num_simulations=1)
    node = MCTSNode(GameState())

    monkeypatch.setattr("src.mcts.generate_all_legal_moves", _dummy_generate_one_move)
    monkeypatch.setattr("src.mcts.apply_move", _dummy_apply_move)
    monkeypatch.setattr("src.mcts.state_to_tensor", _dummy_state_to_tensor)
    monkeypatch.setattr("src.mcts.get_move_probabilities", _dummy_get_move_probabilities)

    val = mcts._expand_and_evaluate(node)
    assert not node.is_terminal()
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert len(node.children) == 1
    assert pytest.approx(val, rel=0.0, abs=1e-6) == 0.2

