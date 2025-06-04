import random
from typing import List, Dict, Any
from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.random_agent import RandomAgent

def test_single_game(seed: int, max_turns=200, verbose=True):
    """
    使用指定的随机种子测试一局随机智能体游戏并显示详细过程
    
    参数:
        seed: 随机种子
        max_turns: 最大回合数
        verbose: 是否打印详细信息
    """
    # 设置随机种子
    random.seed(seed)
    
    # 初始化游戏状态
    state = GameState()
    agent = RandomAgent()
    
    if verbose:
        print(f"=== 开始测试随机智能体游戏 (种子: {seed}) ===")
        print(f"初始状态:\n{state}\n")
    
    # 跟踪阶段变化
    current_phase = state.phase
    
    # 记录游戏历史
    history = [state.copy()]
    
    for turn in range(max_turns):
        # 获取当前玩家
        current_player = state.current_player
        
        # 检查阶段变化
        if state.phase != current_phase:
            if verbose:
                print(f"阶段变化: {current_phase.name} -> {state.phase.name}")
            current_phase = state.phase
        
        # 获取所有合法走法
        legal_moves = generate_all_legal_moves(state)
        
        if not legal_moves:
            if verbose:
                print(f"回合 {turn+1}: {current_player.name} 没有合法走法，游戏结束")
            break
        
        # 选择走法
        move = agent.select_move(state)
        
        # 打印走法信息
        if verbose:
            print(f"回合 {turn+1}: {current_player.name} 执行:", end=" ")
            
            if state.phase == Phase.PLACEMENT:
                pos = move['position']
                marks = move['mark_positions']
                mark_str = f", 标记 {marks}" if marks else ""
                print(f"落子 ({pos[0]},{pos[1]}){mark_str}")
            
            elif state.phase == Phase.REMOVAL:
                print(f"处理标记的棋子")
            
            elif state.phase == Phase.FORCED_REMOVAL:
                pos = move['position']
                print(f"强制移除对方棋子 ({pos[0]},{pos[1]})")
            
            elif state.phase == Phase.MOVEMENT:
                if move['action_type'] == 'move':
                    from_pos = move['from_position']
                    to_pos = move['to_position']
                    caps = move['capture_positions']
                    cap_str = f", 提吃 {caps}" if caps else ""
                    print(f"移动 ({from_pos[0]},{from_pos[1]}) -> ({to_pos[0]},{to_pos[1]}){cap_str}")
                elif move['action_type'] == 'no_moves_remove':
                    pos = move['position']
                    print(f"无子可动，移除对方棋子 ({pos[0]},{pos[1]})")
        
        # 应用走法
        try:
            state = apply_move(state, move)
            history.append(state.copy())
            
            if verbose and (turn < 5 or turn % 10 == 0 or turn > max_turns - 5):
                print(f"{state}\n")
            
            # 检查游戏是否结束
            if state.count_player_pieces(Player.BLACK) == 0 and turn > 36:
                if verbose:
                    print(f"游戏结束: 白方获胜！")
                break
            elif state.count_player_pieces(Player.WHITE) == 0 and turn > 36:
                if verbose:
                    print(f"游戏结束: 黑方获胜！")
                break
            
        except Exception as e:
            print(f"回合 {turn+1}: 走法应用错误: {e}")
            print(f"当前状态: {state}")
            print(f"尝试执行的走法: {move}")
            return False, history
    
    # 游戏结束
    if verbose:
        print(f"\n游戏结束，总共进行 {len(history)-1} 个回合")
        print(f"最终状态:\n{state}")
        
        # 打印棋子统计
        black_pieces = state.count_player_pieces(Player.BLACK)
        white_pieces = state.count_player_pieces(Player.WHITE)
        print(f"黑方棋子: {black_pieces}")
        print(f"白方棋子: {white_pieces}")
        
        if black_pieces == 0 and len(history) > 37:
            print("白方获胜!")
        elif white_pieces == 0 and len(history) > 37:
            print("黑方获胜!")
        else:
            print("游戏未决出胜负")
    
    return True, history

def test_multiple_games(num_games=5, start_seed=0):
    """测试多局游戏，每局打印简要信息"""
    print(f"=== 测试 {num_games} 局游戏 ===")
    
    for i in range(num_games):
        seed = start_seed + i
        print(f"\n游戏 {i+1}/{num_games} (种子: {seed}):")
        
        success, history = test_single_game(seed, verbose=False)
        
        # 打印简要结果
        turns = len(history) - 1
        final_state = history[-1]
        black_pieces = final_state.count_player_pieces(Player.BLACK)
        white_pieces = final_state.count_player_pieces(Player.WHITE)
        
        print(f"  回合数: {turns}")
        print(f"  黑方棋子: {black_pieces}")
        print(f"  白方棋子: {white_pieces}")
        
        if black_pieces == 0 and turns > 36:
            print("  结果: 白方获胜")
        elif white_pieces == 0 and turns > 36:
            print("  结果: 黑方获胜")
        else:
            print("  结果: 未决出胜负")
        
        if not success:
            print("  错误: 游戏运行过程中出现异常")

if __name__ == "__main__":
    import sys
    
    # 默认值
    mode = "single"
    num_games = 1
    seed = 42
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            if mode == "single":
                seed = int(sys.argv[2])
            else:
                num_games = int(sys.argv[2])
        except:
            pass
    
    if len(sys.argv) > 3 and mode == "multiple":
        try:
            seed = int(sys.argv[3])
        except:
            pass
    
    # 运行测试
    if mode == "single":
        test_single_game(seed)
    else:
        test_multiple_games(num_games, seed) 