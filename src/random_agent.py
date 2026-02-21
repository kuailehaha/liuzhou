import random
from typing import Any, Dict, List

from src.game_state import GameState, Player
from src.move_generator import MoveType, apply_move, generate_all_legal_moves


class RandomAgent:
    """Random baseline agent that picks uniformly from legal moves."""

    def select_move(self, state: GameState) -> MoveType:
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)


def simulate_game(max_turns: int = 1000) -> List[GameState]:
    """Simulate a random-vs-random game and return state history."""

    state = GameState()
    agent = RandomAgent()
    history = [state.copy()]

    for turn in range(max_turns):
        current_player = state.current_player

        try:
            move = agent.select_move(state)
        except ValueError as exc:
            print(f"Game ended: {exc}")
            break

        try:
            state = apply_move(state, move, quiet=True)
            history.append(state.copy())
            print(f"Turn {turn + 1}, {current_player.name}: {move}")
            print(state)

            winner = state.get_winner()
            if winner == Player.WHITE:
                print("Game over: WHITE wins")
                break
            if winner == Player.BLACK:
                print("Game over: BLACK wins")
                break
        except Exception as exc:
            print(f"Move application error: {exc}")
            break

    if turn == max_turns - 1:
        print(f"Reached max_turns={max_turns}, forced stop")

    return history


if __name__ == "__main__":
    random.seed(42)
    game_history = simulate_game(max_turns=200)
    print(f"Total turns: {len(game_history) - 1}")
