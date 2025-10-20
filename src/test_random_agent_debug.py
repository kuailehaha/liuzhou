import random
from typing import List, Tuple

from src.game_state import GameState, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.random_agent import RandomAgent


def _play_random_game(seed: int = 42, max_turns: int = 60) -> Tuple[bool, List[GameState]]:
    """Execute a short random-vs-random game and return success flag plus history."""
    random.seed(seed)
    state = GameState()
    agent = RandomAgent()
    history = [state.copy()]

    for _ in range(max_turns):
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            return True, history

        move = agent.select_move(state)
        try:
            state = apply_move(state, move, quiet=True)
        except Exception:
            return False, history

        history.append(state.copy())

        if state.count_player_pieces(Player.BLACK) == 0 or state.count_player_pieces(Player.WHITE) == 0:
            return True, history

    return True, history


def test_single_game_smoke():
    """Basic smoke test to ensure a short random game runs without errors."""
    success, history = _play_random_game()
    assert success
    assert len(history) >= 1
