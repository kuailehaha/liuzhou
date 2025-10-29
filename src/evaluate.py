import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.neural_network import ChessNet # For type hinting
from src.mcts import MCTS
from src.random_agent import RandomAgent

@dataclass
class EvaluationStats:
    """Aggregate win/draw/loss information for a head-to-head evaluation."""
    wins: int
    losses: int
    draws: int
    total_games: int

    @property
    def win_rate(self) -> float:
        return self._safe_rate(self.wins)

    @property
    def loss_rate(self) -> float:
        return self._safe_rate(self.losses)

    @property
    def draw_rate(self) -> float:
        return self._safe_rate(self.draws)

    def _safe_rate(self, value: int) -> float:
        return 0.0 if self.total_games == 0 else value / self.total_games

class MCTSAgent:
    """Agent that selects moves via Monte Carlo Tree Search guided by a neural network."""

    def __init__(
        self,
        model: ChessNet,
        mcts_simulations: int,
        temperature: float = 0.05,
        device: str = "cpu",
        add_dirichlet_noise: bool = False,
        verbose: bool = False,
        mcts_verbose: Optional[bool] = None,
    ):
        self.model = model
        self.model.eval()  # Keep the model in inference mode during evaluation
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature  # Controls how deterministic the visit distribution is
        self.device = device
        self.verbose = verbose
        if mcts_verbose is None:
            mcts_verbose = verbose
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            exploration_weight=1.0,  # Exploration constant for UCT scoring
            temperature=self.temperature,
            device=self.device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=mcts_verbose,
        )

    def select_move(self, state: GameState) -> MoveType:
        """Use MCTS to choose a move for the given state."""
        if not generate_all_legal_moves(state):  # Fail fast if the state unexpectedly has no legal moves
            # print(f"MCTSAgent: No legal moves for state:\n{state}")
            raise ValueError("MCTSAgent: No legal moves available for the current state to select from.")

        moves, policy = self.mcts.search(state)
        if not moves:
            # print(f"MCTSAgent: MCTS search returned no moves for state:\n{state}")
            raise ValueError("MCTSAgent: MCTS search returned no moves.")
        
        # In evaluation we simply pick the move with the highest visit probability
        move_idx = np.argmax(policy)
        chosen_move = moves[move_idx]
        self.mcts.advance_root(chosen_move)
        return chosen_move

def play_single_game(
    agent_black,
    agent_white,
    initial_state: GameState = None,
    max_moves: int = 200,
    verbose: bool = False,
) -> float:
    """Play a single game and return the outcome from Black's perspective."""
    state = initial_state if initial_state else GameState()
    move_count = 0
    log = print if verbose else (lambda *args, **kwargs: None)

    while move_count < max_moves:
        current_player = state.current_player
        active_agent = agent_black if current_player == Player.BLACK else agent_white

        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            log(f"Game ended: Player {current_player} has no legal moves.")
            return -1.0 if current_player == Player.BLACK else 1.0

        try:
            selected_move = active_agent.select_move(state)
        except ValueError:
            log(f"Game ended: Agent for {current_player} could not select a move.")
            return -1.0 if current_player == Player.BLACK else 1.0

        state = apply_move(state, selected_move, quiet=True)
        move_count += 1

        winner = state.get_winner()
        if winner is not None:
            return 1.0 if winner == Player.BLACK else -1.0 if winner == Player.WHITE else 0.0

    return 0.0

def evaluate_against_agent(
    challenger_agent,
    opponent_agent,
    num_games: int,
    device: str,
    verbose: bool = False,
    game_verbose: Optional[bool] = None,
) -> EvaluationStats:
    """Pit the challenger against the opponent and return aggregated results."""
    log = print if verbose else (lambda *args, **kwargs: None)
    if num_games % 2 != 0:
        adjusted = max(2, (num_games // 2) * 2)
        if adjusted == 0 and num_games > 0:
            adjusted = 2
        log("Warning: num_games for evaluation should be even for fair comparison. Adjusting...")
        num_games = adjusted

    if game_verbose is None:
        game_verbose = verbose

    challenger_wins = 0
    challenger_losses = 0
    challenger_draws = 0

    for i in range(num_games):
        log(f"  Playing evaluation game {i + 1}/{num_games}...")
        if i < num_games / 2:
            result = play_single_game(challenger_agent, opponent_agent, verbose=game_verbose)
            if result == 1.0:
                challenger_wins += 1
            elif result == -1.0:
                challenger_losses += 1
            else:
                challenger_draws += 1
        else:
            result = play_single_game(opponent_agent, challenger_agent, verbose=game_verbose)
            if result == -1.0:
                challenger_wins += 1
            elif result == 1.0:
                challenger_losses += 1
            else:
                challenger_draws += 1

    return EvaluationStats(
        wins=challenger_wins,
        losses=challenger_losses,
        draws=challenger_draws,
        total_games=num_games
    )

if __name__ == '__main__':
    # Basic smoke test for the evaluation utilities
    print("Testing evaluation module...")
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create an untrained ChessNet model as a lightweight placeholder
    from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
    from src.game_state import GameState
    board_size = GameState.BOARD_SIZE
    dummy_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS).to(current_device)

    # Build the MCTS-based agent used for evaluation
    mcts_agent = MCTSAgent(dummy_model, mcts_simulations=20, device=current_device)  # Low simulation count keeps this quick
    
    # Random baseline agent for comparison
    random_agent = RandomAgent()

    num_eval_games = 4  # Even number so each agent gets both colors
    print(f"\nEvaluating MCTSAgent vs RandomAgent over {num_eval_games} games...")
    stats_vs_random = evaluate_against_agent(mcts_agent, random_agent, num_eval_games, current_device)
    print(f"MCTSAgent record against RandomAgent: {stats_vs_random.wins}-{stats_vs_random.losses}-{stats_vs_random.draws} "
          f"(win {stats_vs_random.win_rate:.2%} / loss {stats_vs_random.loss_rate:.2%} / draw {stats_vs_random.draw_rate:.2%})")

    # Example: evaluate two MCTS agents (e.g., new model vs. previous best)
    # Create another placeholder model to stand in for the previous best
    # old_best_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS).to(current_device)
    # best_model_agent = MCTSAgent(old_best_model, mcts_simulations=20, device=current_device)
    
    # print(f"\nEvaluating MCTSAgent vs Another MCTSAgent (dummy old_best_model) over {num_eval_games} games...")
    # win_rate_vs_best = evaluate_against_agent(mcts_agent, best_model_agent, num_eval_games, current_device)
    # print(f"MCTSAgent win rate against Another MCTSAgent: {win_rate_vs_best:.2%}")

    print("\nTesting play_single_game directly (Random vs Random for quick check)")
    res = play_single_game(RandomAgent(), RandomAgent())
    print(f"Random vs Random game result (1.0 Black win, -1.0 White win, 0.0 Draw): {res}") 
