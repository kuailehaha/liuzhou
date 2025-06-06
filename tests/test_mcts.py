import unittest
from unittest.mock import patch
import torch

from src.mcts import MCTS, MCTSNode
from src.game_state import GameState, Player, Phase

class DummyModel:
    def to(self, device):
        pass
    def eval(self):
        pass
    def __call__(self, x):
        board_area = x.shape[-1] * x.shape[-2]
        log_prob = torch.log(torch.ones(board_area) / board_area)
        return log_prob.unsqueeze(0), log_prob.unsqueeze(0), log_prob.unsqueeze(0), torch.tensor([[0.2]])

def dummy_state_to_tensor(state, player_to_act):
    return torch.zeros(1, 1, state.BOARD_SIZE, state.BOARD_SIZE)

def dummy_get_move_probabilities(log_p1, log_p2, log_pmc, legal_moves, board_size, device):
    num = len(legal_moves)
    if num == 0:
        return [], torch.empty(0)
    return [1.0/num for _ in legal_moves], torch.zeros(num)

def dummy_generate_no_moves(state):
    return []

def dummy_generate_one_move(state):
    return [{'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (0,0), 'mark_positions': None}]

def dummy_apply_move(state, move):
    new_state = state.copy()
    new_state.board[0][0] = state.current_player.value
    new_state.switch_player()
    return new_state

class MCTSTestCase(unittest.TestCase):
    def test_terminal_value_backpropagated(self):
        mcts = MCTS(DummyModel(), num_simulations=1)
        parent = MCTSNode(GameState())
        child = MCTSNode(parent.state.copy(), parent=parent)
        with patch('src.mcts.generate_all_legal_moves', dummy_generate_no_moves):
            val = mcts._expand_and_evaluate(child)
        self.assertTrue(child.is_terminal())
        self.assertEqual(val, -1.0)
        child.backpropagate(child.terminal_value)
        self.assertEqual(parent.visit_count, 1)
        self.assertEqual(parent.value_sum, 1.0)

    def test_expand_does_not_increment_counts(self):
        mcts = MCTS(DummyModel(), num_simulations=1)
        node = MCTSNode(GameState())
        with patch('src.mcts.generate_all_legal_moves', dummy_generate_one_move), \
             patch('src.mcts.apply_move', dummy_apply_move), \
             patch('src.mcts.state_to_tensor', dummy_state_to_tensor), \
             patch('src.mcts.get_move_probabilities', dummy_get_move_probabilities):
            val = mcts._expand_and_evaluate(node)
        self.assertFalse(node.is_terminal())
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(len(node.children), 1)
        self.assertAlmostEqual(val, 0.2)

if __name__ == '__main__':
    unittest.main()
