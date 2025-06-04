import random
from typing import List, Dict, Any
from src.game_state import GameState, Player
from src.move_generator import generate_all_legal_moves, apply_move, MoveType

class RandomAgent:
    """随机智能体，随机选择一个合法走法"""
    
    def select_move(self, state: GameState) -> MoveType:
        """从当前状态选择一个随机合法走法"""
        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            raise ValueError("没有合法走法可选")
        return random.choice(legal_moves)

def simulate_game(max_turns=1000):
    """模拟一局游戏，两方都使用随机智能体"""
    state = GameState()
    agent = RandomAgent()
    
    # 创建游戏历史记录
    history = [state.copy()]
    
    for turn in range(max_turns):
        # 获取当前玩家
        current_player = state.current_player
        
        # 选择走法
        try:
            move = agent.select_move(state)
        except ValueError as e:
            print(f"游戏结束：{e}")
            break
        
        # 应用走法
        try:
            state = apply_move(state, move)
            history.append(state.copy())
            
            # 打印当前状态
            print(f"回合 {turn+1}, {current_player.name} 执行: {move}")
            print(state)
            
            # 检查游戏是否结束
            if state.count_player_pieces(Player.BLACK) == 0 and turn > 36:
                print("游戏结束：白方获胜！")
                break
            elif state.count_player_pieces(Player.WHITE) == 0 and turn > 36:
                print("游戏结束：黑方获胜！")
                break
        except Exception as e:
            print(f"走法应用错误: {e}")
            break
    
    if turn == max_turns - 1:
        print(f"游戏达到最大回合数 {max_turns}，强制结束")
    
    return history

# 测试代码
if __name__ == "__main__":
    # 设置随机种子以便重现结果
    random.seed(42)
    
    # 模拟一局游戏
    game_history = simulate_game(max_turns=200)
    
    # 输出游戏长度
    print(f"游戏总共进行了 {len(game_history)-1} 个回合")
