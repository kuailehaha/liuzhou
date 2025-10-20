import random

from src.game_state import GameState
from src.move_generator import generate_all_legal_moves, apply_move
from src.random_agent import RandomAgent


def _play_turn(state: GameState, agent: RandomAgent) -> GameState:
    legal_moves = generate_all_legal_moves(state)
    if not legal_moves:
        return state
    move = agent.select_move(state)
    return apply_move(state, move, quiet=True)


def test_random_agent_multiple_turns():
    """Ensure repeated random moves progress the game without raising errors."""
    random.seed(7)
    state = GameState()
    agent = RandomAgent()

    for _ in range(30):
        state = _play_turn(state, agent)

    assert isinstance(state, GameState)
