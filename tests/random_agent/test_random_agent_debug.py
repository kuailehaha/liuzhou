"""
Random agent smoke tests.

Usage:
  pytest tests/random_agent/test_random_agent_debug.py -q
Seed defaults are set inside helpers (e.g., 42, 7).
"""
import random
from typing import List, Tuple

import pytest

from src.game_state import GameState, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.random_agent import RandomAgent


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def num_games() -> int:
    return 10


@pytest.fixture
def max_turns() -> int:
    return 60


@pytest.fixture
def verbose() -> bool:
    return True


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

        if state.get_winner() is not None:
            return True, history

    return True, history


def run_single_game(seed: int, max_turns: int = 60, verbose: bool = True) -> List[GameState]:
    """Run a single random-agent game and optionally print a brief summary."""
    success, history = _play_random_game(seed=seed, max_turns=max_turns)
    if verbose:
        print(f"Single game finished with {len(history) - 1} turns (seed={seed}).")
    if not success:
        raise RuntimeError("Random agent produced an invalid move during single-game test.")
    return history


def run_multiple_games(
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


def test_single_game(seed: int, max_turns: int = 60, verbose: bool = True) -> None:
    """Run a single random-agent game and assert it finishes cleanly."""
    history = run_single_game(seed=seed, max_turns=max_turns, verbose=verbose)
    assert len(history) >= 1


def test_multiple_games(
    num_games: int,
    seed: int,
    max_turns: int = 60,
    verbose: bool = True,
) -> None:
    """Run several random-agent games in sequence and assert they finish cleanly."""
    completed = run_multiple_games(
        num_games=num_games,
        seed=seed,
        max_turns=max_turns,
        verbose=verbose,
    )
    assert completed == num_games


def test_single_game_smoke():
    """Basic smoke test to ensure a short random game runs without errors."""
    success, history = _play_random_game()
    assert success
    assert len(history) >= 1
