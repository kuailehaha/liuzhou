# file: src/rule_engine.py

from typing import List, Tuple
from src.game_state import GameState, Phase, Player

def generate_legal_moves_phase1(state: GameState) -> List[Tuple[int, int]]:
    """
    生成第一阶段可落子的位置(空位)。
    """
    legal_moves = []
    if state.phase != Phase.PLACEMENT:
        return legal_moves

    for r in range(state.BOARD_SIZE):
        for c in range(state.BOARD_SIZE):
            if state.board[r][c] == 0:  # 空位
                legal_moves.append((r, c))

    return legal_moves

def apply_move_phase1(state: GameState, move: Tuple[int, int]) -> GameState:
    """
    在第一阶段执行一手落子:
    1. 在空位落子
    2. 若该子未被标记过，则检查是否形成方/洲; 如有则标记对方若干子
    3. 如果棋盘满，进入第二阶段，否则切换玩家
    """
    r, c = move
    new_state = state.copy()

    if new_state.board[r][c] != 0 or new_state.phase != Phase.PLACEMENT:
        # 非法情况，直接返回原state(也可抛异常)
        return new_state

    current_player = new_state.current_player
    # 下子
    new_state.board[r][c] = current_player.value

    # ---【关键】判断：该子是否已经被对手标记过？---
    # 如果是黑方下子，则( r, c )是否在 marked_black？
    # 如果是白方下子，则( r, c )是否在 marked_white？
    # 若已标记 => 不能再触发新标记
    if current_player == Player.BLACK:
        # 检查我方新下的子是否在 black标记集中(其实黑方自己不会标记黑子；这里要看规则的含义)
        # 根据规则：“如果某颗棋子已经被对手标记，则不能再触发新的标记”；
        #   => 即黑方落子的 (r, c) 是否在 state.marked_black 里。
        #   但是 'marked_black' 的含义是“被标记的黑方棋子”，所以要检查 (r,c) in new_state.marked_black
        already_marked = (r, c) in new_state.marked_black
    else:
        already_marked = (r, c) in new_state.marked_white

    # 若没有被标记过，就进行形状检测
    if not already_marked:
        shape = detect_shape_formed(new_state.board, r, c, current_player.value)
        if shape == "line":
            # 标记对手2颗棋子
            mark_opponent_pieces(new_state, count=2)
        elif shape == "square":
            # 标记对手1颗棋子
            mark_opponent_pieces(new_state, count=1)
        # else: "none" 不做任何事

    # 若棋盘已经满，进入Phase 2；否则切换玩家
    if new_state.is_board_full():
        new_state.phase = Phase.REMOVAL
    else:
        new_state.switch_player()

    return new_state




def check_squares(board: List[List[int]], r: int, c: int, player_value: int) -> bool:
    """
    判断以 (r, c) 这步新落子为中心，是否形成了至少一个2x2的方块(同色).
    只要找到一个即可返回True，否则False。
    """
    # 棋盘大小
    size = len(board)
    # 可能形成2x2方块的左上角坐标范围
    # (r,c)有可能是方块的四个角之一，因此要检查与(r,c)相邻的区域
    for dr in [0, -1]:
        for dc in [0, -1]:
            rr = r + dr
            cc = c + dc
            # 确保 2x2 方块在棋盘范围内
            if 0 <= rr < size-1 and 0 <= cc < size-1:
                if (board[rr][cc] == player_value and
                    board[rr][cc+1] == player_value and
                    board[rr+1][cc] == player_value and
                    board[rr+1][cc+1] == player_value):
                    return True
    return False

def check_lines(board: List[List[int]], r: int, c: int, player_value: int) -> bool:
    """
    检查是否形成“6连线”（洲），仅考虑水平或垂直连续6子，不考虑斜线。
    只要找到一条满足条件的就返回 True，否则 False。
    """
    size = len(board)

    # 行、列
    row = r
    col = c

    # ===== 检查所在行是否有连续6枚 player_value =====
    # 为了便于处理，我们先找到该行所有同色棋子的最大连续段。
    # 具体做法：以 (row,col) 为中心，向左/向右扩展统计。
    count_in_row = 1  # (row, col) 本身算1
    # 向左扩展
    cc = col - 1
    while cc >= 0 and board[row][cc] == player_value:
        count_in_row += 1
        cc -= 1
    # 向右扩展
    cc = col + 1
    while cc < size and board[row][cc] == player_value:
        count_in_row += 1
        cc += 1
    if count_in_row >= 6:
        return True

    # ===== 检查所在列是否有连续6枚 player_value =====
    count_in_col = 1  # (row, col) 本身算1
    # 向上扩展
    rr = row - 1
    while rr >= 0 and board[rr][col] == player_value:
        count_in_col += 1
        rr -= 1
    # 向下扩展
    rr = row + 1
    while rr < size and board[rr][col] == player_value:
        count_in_col += 1
        rr += 1
    if count_in_col >= 6:
        return True

    return False

def detect_shape_formed(board: List[List[int]], r: int, c: int, player_value: int) -> str:
    """
    综合检测本次落子后是否至少形成一个"洲"或"方"。
    优先返回 'line' 表示洲，其次 'square' 表示方，若都无则返回 'none'。
    """
    # 先判定是否形成洲(6连线)
    if check_lines(board, r, c, player_value):
        return "line"

    # 再判定是否形成方(2x2)
    if check_squares(board, r, c, player_value):
        return "square"

    return "none"

def mark_opponent_pieces(state: GameState, count: int):
    """
    简单示例：自动从左上往右下寻找对手棋子，挑count颗未被标记的来标记。
    若不足count颗，则尽量标记全部可标记的。
    """
    current_player = state.current_player
    opponent_value = current_player.opponent().value

    # 我方若是 BLACK=1，则要标记对方(WHITE=-1)的子 => state.marked_white
    # 反之亦然
    if current_player == Player.BLACK:
        marked_set = state.marked_white
    else:
        marked_set = state.marked_black

    # 扫描棋盘，找到对方棋子
    marked = 0
    for r in range(state.BOARD_SIZE):
        for c in range(state.BOARD_SIZE):
            if marked >= count:
                break
            if state.board[r][c] == opponent_value:
                # 还未被标记则标记
                if current_player == Player.BLACK:
                    if (r, c) not in state.marked_white:
                        marked_set.add((r, c))
                        marked += 1
                else:
                    if (r, c) not in state.marked_black:
                        marked_set.add((r, c))
                        marked += 1

        if marked >= count:
            break