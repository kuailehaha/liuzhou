import numpy as np
import torch
from typing import List, Tuple

from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.neural_network import ChessNet # For type hinting
from src.mcts import MCTS
from src.random_agent import RandomAgent

class MCTSAgent:
    """使用 MCTS 和神经网络模型选择动作的智能体。"""
    def __init__(self, model: ChessNet, mcts_simulations: int, temperature: float = 0.1, device: str = 'cpu'):
        self.model = model
        self.model.eval() # 确保模型处于评估模式
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature # 评估时通常使用较低的温度以选择最佳动作
        self.device = device
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            exploration_weight=1.0, # MCTS中的标准探索权重
            temperature=self.temperature, # MCTS内部也使用温度，这里保持一致
            device=self.device
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
        return moves[move_idx]

def play_single_game(agent_black, agent_white, initial_state: GameState = None, max_moves: int = 200) -> float:
    """
    在两个智能体之间进行单局游戏。

    Args:
        agent_black: 控制黑方的智能体。
        agent_white: 控制白方的智能体。
        initial_state: 初始游戏状态，如果为None，则从默认状态开始。
        max_moves: 游戏最大步数，防止无限循环。

    Returns:
        1.0 如果黑方胜，-1.0 如果白方胜，0.0 如果平局。
    """
    state = initial_state if initial_state else GameState()
    move_count = 0

    while move_count < max_moves:
        current_player = state.current_player
        active_agent = agent_black if current_player == Player.BLACK else agent_white

        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            # 当前玩家没有合法走法，判负
            print(f"Game ended: Player {current_player} has no legal moves.")
            return -1.0 if current_player == Player.BLACK else 1.0

        try:
            selected_move = active_agent.select_move(state)
        except ValueError: # 如果智能体内部无法选择动作（例如 RandomAgent 在无棋可走时）
            print(f"Game ended: Agent for {current_player} could not select a move.")
            return -1.0 if current_player == Player.BLACK else 1.0
        
        # （可选）验证所选动作是否合法 - 通常智能体应该只选择合法动作
        # is_move_in_legal_list = any(selected_move == lm for lm in legal_moves)
        # if not is_move_in_legal_list:
        #     print(f"Error: Agent for {current_player} selected an ILLEGAL move!")
        #     print(f"Selected: {selected_move}")
        #     print(f"Legal moves: {legal_moves}")
        #     return -1.0 if current_player == Player.BLACK else 1.0 # 判非法操作方负

        state = apply_move(state, selected_move)
        move_count += 1

        winner = state.get_winner()
        if winner is not None:
            return 1.0 if winner == Player.BLACK else -1.0 if winner == Player.WHITE else 0.0
    
    # print(f"Game ended: Reached max moves ({max_moves}).")
    return 0.0 # 达到最大步数，平局

def evaluate_against_agent(
    challenger_agent, # 通常是新训练的模型 MCTS Agent
    opponent_agent,   # RandomAgent 或 BestModel MCTS Agent
    num_games: int,
    device: str # challenger_agent 可能需要device
) -> float:
    """
    评估 challenger_agent 相对于 opponent_agent 的胜率。
    双方轮流执黑。

    Args:
        challenger_agent: 挑战者智能体。
        opponent_agent: 对手智能体。
        num_games: 对弈的总局数 (必须是偶数，以保证双方执黑次数相同)。
        device: challenger_agent 可能需要的设备。

    Returns:
        challenger_agent 的胜率。
    """
    if num_games % 2 != 0:
        print("Warning: num_games for evaluation should be even for fair comparison. Adjusting...")
        num_games_original = num_games  # 保存原始值
        num_games = max(2, (num_games // 2) * 2) # 确保至少是2局偶数
        if num_games == 0 and num_games_original > 0 : num_games = 2 # Handle original 1 game case

    challenger_wins = 0.0

    for i in range(num_games):
        print(f"  Playing evaluation game {i + 1}/{num_games}...")
        if i < num_games / 2:
            # Challenger is Black
            # print(f"    Challenger (Black) vs Opponent (White)")
            result = play_single_game(challenger_agent, opponent_agent)
            if result == 1.0: # Challenger (Black) won
                challenger_wins += 1
        else:
            # Challenger is White
            # print(f"    Opponent (Black) vs Challenger (White)")
            result = play_single_game(opponent_agent, challenger_agent)
            if result == -1.0: # Challenger (White) won
                challenger_wins += 1
        # print(f"    Game {i+1} result for challenger: {result if i < num_games / 2 else -result}")
            
    if num_games == 0: return 0.0
    return challenger_wins / num_games

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