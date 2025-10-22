import numpy as np
import torch
from typing import List, Optional, Tuple

from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.neural_network import ChessNet # For type hinting
from src.mcts import MCTS
from src.random_agent import RandomAgent

class MCTSAgent:
    """?? MCTS ????????????????"""

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
        self.model.eval()  # ??????????
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature  # ???????????????????
        self.device = device
        self.verbose = verbose
        if mcts_verbose is None:
            mcts_verbose = verbose
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            exploration_weight=1.0,  # MCTS????????
            temperature=self.temperature,
            device=self.device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=mcts_verbose,
        )

    def select_move(self, state: GameState) -> MoveType:
        """使用MCTS搜索选择一个动作。"""
        if not generate_all_legal_moves(state): # 提前检查，以防万一
            # print(f"MCTSAgent: No legal moves for state:\n{state}")
            raise ValueError("MCTSAgent: No legal moves available for the current state to select from.")

        moves, policy = self.mcts.search(state)
        if not moves:
            # print(f"MCTSAgent: MCTS search returned no moves for state:\n{state}")
            raise ValueError("MCTSAgent: MCTS search returned no moves.")
        
        # 在评估时，通常选择概率最高的动作
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
    """???????????????"""
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

        state = apply_move(state, selected_move)
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
) -> float:
    """?? challenger_agent ??? opponent_agent ????"""
    log = print if verbose else (lambda *args, **kwargs: None)
    if num_games % 2 != 0:
        adjusted = max(2, (num_games // 2) * 2)
        if adjusted == 0 and num_games > 0:
            adjusted = 2
        log("Warning: num_games for evaluation should be even for fair comparison. Adjusting...")
        num_games = adjusted

    if game_verbose is None:
        game_verbose = verbose

    challenger_wins = 0.0

    for i in range(num_games):
        log(f"  Playing evaluation game {i + 1}/{num_games}...")
        if i < num_games / 2:
            result = play_single_game(challenger_agent, opponent_agent, verbose=game_verbose)
            if result == 1.0:
                challenger_wins += 1
        else:
            result = play_single_game(opponent_agent, challenger_agent, verbose=game_verbose)
            if result == -1.0:
                challenger_wins += 1

    return 0.0 if num_games == 0 else challenger_wins / num_games

if __name__ == '__main__':
    # 简单测试评估流程
    print("Testing evaluation module...")
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建一个虚拟的ChessNet模型 (未训练)
    from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
    from src.game_state import GameState
    board_size = GameState.BOARD_SIZE
    dummy_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS).to(current_device)

    # 创建MCTS Agent
    mcts_agent = MCTSAgent(dummy_model, mcts_simulations=20, device=current_device) # 低模拟次数用于测试
    
    # 创建Random Agent
    random_agent = RandomAgent()

    num_eval_games = 4 # 必须是偶数
    print(f"\nEvaluating MCTSAgent vs RandomAgent over {num_eval_games} games...")
    win_rate_vs_random = evaluate_against_agent(mcts_agent, random_agent, num_eval_games, current_device)
    print(f"MCTSAgent win rate against RandomAgent: {win_rate_vs_random:.2%}")

    # 模拟评估两个MCTS Agent (例如，新模型 vs 旧的最佳模型)
    # 创建另一个虚拟模型代表旧的最佳模型
    # old_best_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS).to(current_device)
    # best_model_agent = MCTSAgent(old_best_model, mcts_simulations=20, device=current_device)
    
    # print(f"\nEvaluating MCTSAgent vs Another MCTSAgent (dummy old_best_model) over {num_eval_games} games...")
    # win_rate_vs_best = evaluate_against_agent(mcts_agent, best_model_agent, num_eval_games, current_device)
    # print(f"MCTSAgent win rate against Another MCTSAgent: {win_rate_vs_best:.2%}")

    print("\nTesting play_single_game directly (Random vs Random for quick check)")
    res = play_single_game(RandomAgent(), RandomAgent())
    print(f"Random vs Random game result (1.0 Black win, -1.0 White win, 0.0 Draw): {res}") 
