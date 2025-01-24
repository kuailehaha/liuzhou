from src.game_state import GameState, Phase, Player
from src.rule_engine import generate_legal_moves_phase1, apply_move_phase1

def main():
    state = GameState()  # 空盘, 阶段=PLACEMENT, current_player=BLACK
    moves = generate_legal_moves_phase1(state)
    print("合法落子数:", len(moves))  # 36

    # 演示在(0,0) 下第一子(黑)
    state = apply_move_phase1(state, (0,0))
    print("下完第一手后:\n", state)

    # 再给白方在(0,1)落子
    state = apply_move_phase1(state, (0,1))
    print("下完第二手后:\n", state)

    # ... 可以继续模拟直到棋盘填满
    # 或编写测试用例检查标记逻辑是否正确

if __name__ == "__main__":
    main()