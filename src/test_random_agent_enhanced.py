import random
import time
from typing import List, Dict, Any, Tuple, Counter
from tqdm import tqdm
from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.random_agent import RandomAgent

class GameStats:
    """游戏统计信息类"""
    def __init__(self):
        self.total_games = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0
        self.avg_game_length = 0.0 # Initialize as float
        self.max_game_length = 0
        self.min_game_length = float('inf')
        self.phase_transitions = {
            Phase.PLACEMENT: {Phase.REMOVAL: 0},
            Phase.REMOVAL: {Phase.FORCED_REMOVAL: 0, Phase.MOVEMENT: 0},
            Phase.FORCED_REMOVAL: {Phase.MOVEMENT: 0}
        }
        self.piece_diff_at_end = []  # 游戏结束时的棋子差异
        self.turn_distribution = Counter()  # 游戏回合数分布
    
    def add_game_result(self, game_length: int, winner: Player = None, phase_changes: Dict = None, 
                        black_pieces: int = 0, white_pieces: int = 0):
        """添加一局游戏结果"""
        self.total_games += 1
        
        # 记录获胜方
        if winner == Player.BLACK:
            self.black_wins += 1
        elif winner == Player.WHITE:
            self.white_wins += 1
        else:
            self.draws += 1
        
        # 更新游戏长度统计
        if self.total_games == 1: # First game
             self.avg_game_length = float(game_length)
        elif self.total_games > 1:
            self.avg_game_length = (self.avg_game_length * (self.total_games - 1) + game_length) / self.total_games

        self.max_game_length = max(self.max_game_length, game_length)
        self.min_game_length = min(self.min_game_length, game_length)
        self.turn_distribution[game_length] += 1
        
        # 更新阶段转换统计
        if phase_changes:
            for from_phase, to_phase in phase_changes:
                if from_phase in self.phase_transitions and to_phase in self.phase_transitions[from_phase]:
                    self.phase_transitions[from_phase][to_phase] += 1
        
        # 记录棋子差异
        self.piece_diff_at_end.append(black_pieces - white_pieces)
    
    def print_summary(self):
        """打印统计信息摘要"""
        print("\n" + "=" * 50)
        print("游戏统计摘要")
        print("=" * 50)
        
        total_games_val = self.total_games
        
        print(f"总游戏数: {total_games_val}")
        black_wins_percent = (self.black_wins / total_games_val * 100) if total_games_val > 0 else 0
        print(f"黑方获胜: {self.black_wins} ({black_wins_percent:.2f}%)")
        white_wins_percent = (self.white_wins / total_games_val * 100) if total_games_val > 0 else 0
        print(f"白方获胜: {self.white_wins} ({white_wins_percent:.2f}%)")
        draws_percent = (self.draws / total_games_val * 100) if total_games_val > 0 else 0
        print(f"平局: {self.draws} ({draws_percent:.2f}%)")
        
        print(f"\n游戏长度统计:")
        print(f"  平均回合数: {self.avg_game_length:.2f}")
        print(f"  最长回合数: {self.max_game_length if total_games_val > 0 else 0}")
        min_len = self.min_game_length if self.min_game_length != float('inf') else 0
        print(f"  最短回合数: {min_len}")
        
        print(f"\n阶段转换统计:")
        for from_phase, transitions in self.phase_transitions.items():
            for to_phase, count in transitions.items():
                if count > 0:
                    print(f"  {from_phase.name} -> {to_phase.name}: {count} 次")
        
        print(f"\n游戏结束时棋子差异 (黑-白):")
        if self.piece_diff_at_end:
            avg_diff = sum(self.piece_diff_at_end) / len(self.piece_diff_at_end)
            print(f"  平均差异: {avg_diff:.2f}")
            max_black_adv = max(self.piece_diff_at_end) if self.piece_diff_at_end else 0
            print(f"  最大黑方优势: {max_black_adv}")
            min_val_for_white_adv = [val for val in self.piece_diff_at_end if val < 0]
            max_white_adv = abs(min(min_val_for_white_adv)) if min_val_for_white_adv else 0
            print(f"  最大白方优势: {max_white_adv}")

        else:
            print("  无数据")
        
        print(f"\n最常见的游戏长度:")
        for turns, count in self.turn_distribution.most_common(5):
            turn_percent = (count / total_games_val * 100) if total_games_val > 0 else 0
            print(f"  {turns} 回合: {count} 次 ({turn_percent:.2f}%)")
        
        print("=" * 50)

def test_random_agent(seed: int, max_turns=200, quiet_mode: bool = False) -> Tuple[bool, Dict]:
    """
    使用指定的随机种子测试随机智能体
    
    参数:
        seed: 随机种子
        max_turns: 最大回合数
        quiet_mode: 是否为安静模式，抑制局间打印
        
    返回:
        (是否成功完成测试（无异常）, 游戏统计信息)
    """
    random.seed(seed)
    
    game_info = {
        'winner': None,
        'turns': 0,
        'phase_changes': [],
        'black_pieces': 0,
        'white_pieces': 0
    }
    
    try:
        state = GameState()
        agent = RandomAgent()
        current_phase = state.phase
        
        for turn in range(max_turns):
            current_player = state.current_player
            if state.phase != current_phase:
                game_info['phase_changes'].append((current_phase, state.phase))
                current_phase = state.phase
            
            try:
                move = agent.select_move(state)
            except ValueError:
                break
            
            try:
                state = apply_move(state, move, quiet=quiet_mode)
                game_info['turns'] = turn + 1
                
                if state.count_player_pieces(Player.BLACK) == 0 and turn > 36:
                    game_info['winner'] = Player.WHITE
                    break
                elif state.count_player_pieces(Player.WHITE) == 0 and turn > 36:
                    game_info['winner'] = Player.BLACK
                    break
            except Exception as e:
                if not quiet_mode:
                    print(f"种子 {seed}, 回合 {turn+1}: 走法应用错误: {e}")
                    print(f"当前状态: {state}")
                    print(f"尝试执行的走法: {move}")
                return False, game_info
        
        game_info['black_pieces'] = state.count_player_pieces(Player.BLACK)
        game_info['white_pieces'] = state.count_player_pieces(Player.WHITE)
        return True, game_info
    
    except Exception as e:
        if not quiet_mode:
            print(f"种子 {seed} 测试过程中发生异常: {e}")
        return False, game_info

def run_mass_tests(num_tests=10000, start_seed=0, verbose=True, max_turns_per_game=200):
    """
    运行大量随机种子测试
    
    参数:
        num_tests: 测试次数
        start_seed: 起始随机种子
        verbose: 是否打印详细进度信息 (True for text progress, False for tqdm)
        max_turns_per_game: 每局游戏的最大回合数
    
    返回:
        游戏统计信息对象
    """
    start_time = time.time()
    
    if verbose:
        print(f"开始运行 {num_tests} 个随机种子测试...")
    
    success_count = 0
    failure_seeds = []
    stats = GameStats()
    
    # tqdm is used when verbose is False (quiet mode)
    # disable=verbose means tqdm is disabled if verbose is True
    iterable = tqdm(range(num_tests), desc="进行测试", unit="局", disable=verbose)

    for i in iterable:
        seed = start_seed + i
        
        if verbose and ((i+1) % 100 == 0 or i == 0): # Text progress if verbose
            current_progress_percent = (i + 1) / num_tests * 100
            print(f"测试进度: {i+1}/{num_tests} ({current_progress_percent:.2f}%)")
        
        # quiet_mode for test_random_agent is True when verbose for run_mass_tests is False
        success, game_info = test_random_agent(seed, max_turns=max_turns_per_game, quiet_mode=(not verbose))
        
        if success:
            success_count += 1
            stats.add_game_result(
                game_length=game_info['turns'],
                winner=game_info['winner'],
                phase_changes=game_info['phase_changes'],
                black_pieces=game_info['black_pieces'],
                white_pieces=game_info['white_pieces']
            )
        else:
            failure_seeds.append(seed)
    
    elapsed_time = time.time() - start_time
    
    # Always print the summary
    print("\n" + "=" * 50)
    success_percent = (success_count / num_tests * 100) if num_tests > 0 else 0
    print(f"测试完成: 成功 {success_count}/{num_tests} ({success_percent:.2f}%)")
    print(f"总运行时间: {elapsed_time:.2f} 秒")
    
    if failure_seeds:
        print(f"失败的种子: {failure_seeds[:10]}{'...' if len(failure_seeds) > 10 else ''}")
    elif num_tests > 0 and success_count == num_tests: # Ensure tests were run and all succeeded
        print("所有测试均成功通过!")
        
    stats.print_summary()
    
    return stats

if __name__ == "__main__":
    import sys
    
    num_tests = 10000
    start_seed = 0
    verbose_flag = True # Default to verbose (text progress, no tqdm)
    
    if len(sys.argv) > 1:
        try:
            num_tests = int(sys.argv[1])
        except ValueError:
            print(f"无法解析测试次数: {sys.argv[1]}。使用默认值 {num_tests}。")
    
    if len(sys.argv) > 2:
        try:
            start_seed = int(sys.argv[2])
        except ValueError:
            print(f"无法解析起始种子: {sys.argv[2]}。使用默认值 {start_seed}。")
            
    # Check for a quiet flag if run_tests.py passes it (e.g. from -q)
    # This part assumes run_tests.py would modify sys.argv or pass verbose differently
    # For direct execution of this file, verbose is True by default.
    # If run_tests.py calls this with verbose=False, then tqdm will be used.

    run_mass_tests(num_tests, start_seed, verbose=verbose_flag) 