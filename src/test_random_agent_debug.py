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


def test_single_game(seed: int, max_turns: int = 60, verbose: bool = True) -> List[GameState]:
    """Run a single random-agent game and optionally print a brief summary."""
    success, history = _play_random_game(seed=seed, max_turns=max_turns)
    if verbose:
        print(f"Single game finished with {len(history) - 1} turns (seed={seed}).")
    if not success:
        raise RuntimeError("Random agent produced an invalid move during single-game test.")
    return history


def test_multiple_games(
    num_games: int,
    seed: int,
    max_turns: int = 60,
    verbose: bool = True,
) -> int:
    """Run several random-agent games in sequence and raise if any fail."""
    if num_games <= 0:
        raise ValueError("num_games must be a positive integer")

    failures = 0
    for idx in range(num_games):
        game_seed = seed + idx
        success, _ = _play_random_game(seed=game_seed, max_turns=max_turns)
        if not success:
            failures += 1
            if verbose:
                print(f"  Game {idx + 1} failed (seed={game_seed}).")

    if verbose:
        print(f"Completed {num_games - failures}/{num_games} games successfully.")

    if failures:
        raise RuntimeError(f"{failures} random-agent games failed validation.")

    return num_games


def test_single_game_smoke():
    """Basic smoke test to ensure a short random game runs without errors."""
    success, history = _play_random_game()
    assert success
    assert len(history) >= 1
