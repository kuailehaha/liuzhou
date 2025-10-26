import random
from typing import Dict, Union

from src.game_state import GameState, Player
from src.move_generator import generate_all_legal_moves, apply_move
from src.random_agent import RandomAgent

StatsDict = Dict[str, Union[int, float]]


def _play_turn(state: GameState, agent: RandomAgent) -> GameState:
    legal_moves = generate_all_legal_moves(state)
    if not legal_moves:
        return state
    move = agent.select_move(state)
    return apply_move(state, move, quiet=True)


def _simulate_random_game(agent: RandomAgent, game_seed: int, max_turns: int):
    """Simulate a single random-agent game and return outcome metadata."""
    random.seed(game_seed)
    state = GameState()
    turns_played = 0

    for turn in range(1, max_turns + 1):
        try:
            move = agent.select_move(state)
        except ValueError:
            winner = state.get_winner()
            return winner, turns_played, "no_legal_moves", None

        try:
            state = apply_move(state, move, quiet=True)
        except Exception as exc:  # noqa: BLE001 - escalate later via RuntimeError
            return None, turns_played, "exception", exc

        turns_played = turn
        winner = state.get_winner()
        if winner is not None:
            return winner, turns_played, "win", None

    return None, turns_played, "max_turns", None


def run_enhanced_tests(
    num_games: int,
    seed: int,
    verbose: bool = True,
    max_turns_per_game: int = 200,
) -> StatsDict:
    """Run large-scale random-agent simulations and report aggregate statistics."""
    if num_games <= 0:
        raise ValueError("num_games must be a positive integer")
    if max_turns_per_game <= 0:
        raise ValueError("max_turns_per_game must be a positive integer")

    agent = RandomAgent()
    stats: StatsDict = {
        "total_games": num_games,
        "completed_games": 0,
        "black_wins": 0,
        "white_wins": 0,
        "draws": 0,
        "max_turns_reached": 0,
        "stalled_games": 0,
        "total_turns": 0,
        "longest_game": 0,
        "average_turns": 0.0,
    }

    progress_interval = max(1, num_games // 10)

    for game_index in range(num_games):
        winner, turns_played, reason, error = _simulate_random_game(
            agent, seed + game_index, max_turns_per_game
        )

        if error is not None:
            raise RuntimeError(
                f"Random agent produced an invalid move in game {game_index} "
                f"(after {turns_played} turns): {error}"
            )

        stats["completed_games"] += 1
        stats["total_turns"] += turns_played
        stats["longest_game"] = max(stats["longest_game"], turns_played)

        if winner is None:
            stats["draws"] += 1
        elif winner == Player.BLACK:
            stats["black_wins"] += 1
        else:
            stats["white_wins"] += 1

        if reason == "max_turns":
            stats["max_turns_reached"] += 1
        elif reason == "no_legal_moves" and winner is None:
            stats["stalled_games"] += 1

        if verbose and (
            (game_index + 1) % progress_interval == 0
            or game_index == num_games - 1
        ):
            print(f"  Completed {game_index + 1}/{num_games} games...")

    if stats["completed_games"]:
        stats["average_turns"] = stats["total_turns"] / stats["completed_games"]

    if verbose:
        total = max(1, stats["completed_games"])
        print("\n---- Enhanced Random Agent Test Summary ----")
        print(
            f"Games played: {stats['completed_games']}/{stats['total_games']}"
        )
        print(
            f"Black wins : {stats['black_wins']} "
            f"({stats['black_wins'] / total:.2%})"
        )
        print(
            f"White wins : {stats['white_wins']} "
            f"({stats['white_wins'] / total:.2%})"
        )
        print(
            f"Draws      : {stats['draws']} "
            f"({stats['draws'] / total:.2%})"
        )
        if stats["stalled_games"]:
            print(f"Games ending with no legal moves: {stats['stalled_games']}")
        if stats["max_turns_reached"]:
            print(
                f"Games hitting the turn limit ({max_turns_per_game}): "
                f"{stats['max_turns_reached']}"
            )
        print(f"Average turns per game: {stats['average_turns']:.2f}")
        print(f"Longest game length   : {stats['longest_game']} turns")

    return stats


def test_random_agent_multiple_turns():
    """Ensure repeated random moves progress the game without raising errors."""
    random.seed(7)
    state = GameState()
    agent = RandomAgent()

    for _ in range(30):
        state = _play_turn(state, agent)

    assert isinstance(state, GameState)
