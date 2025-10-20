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

def apply_move_phase1(state: GameState, move: Tuple[int, int], mark_positions: List[Tuple[int, int]] = None) -> GameState:
    """
    在第一阶段执行一手落子:
    1. 在空位落子
    2. 若该子未被标记过，则检查是否形成方/洲; 如有则标记对方若干子
    3. 如果棋盘满，进入第二阶段，否则切换玩家
    """
    r, c = move
    new_state = state.copy()
    current_player = new_state.current_player  # 获取当前玩家

    # 检查是否在空位落子
    if new_state.board[r][c] != 0:
        raise ValueError(f"位置 ({r}, {c}) 已有棋子")
    
    # 检查是否在被标记的位置落子
    if current_player == Player.BLACK:
        if (r, c) in new_state.marked_white:
            raise ValueError(f"位置 ({r}, {c}) 已被标记，不能落子")
    else:
        if (r, c) in new_state.marked_black:
            raise ValueError(f"位置 ({r}, {c}) 已被标记，不能落子")

    if new_state.phase != Phase.PLACEMENT:
        raise ValueError("当前不是落子阶段")

    # 下子
    new_state.board[r][c] = current_player.value

    # ---【关键】判断：该子是否已经被对手标记过？---
    if current_player == Player.BLACK:
        already_marked = (r, c) in new_state.marked_black
        opponent_marked = new_state.marked_white
    else:
        already_marked = (r, c) in new_state.marked_white
        opponent_marked = new_state.marked_black

    # 若没有被标记过，就进行形状检测
    if not already_marked:
        shape = detect_shape_formed(
            new_state.board, r, c, current_player.value,
            new_state.marked_black if current_player == Player.BLACK else new_state.marked_white
        )
        if shape == "line":
            # 形成洲，标记对方2颗棋子
            if mark_positions is None or len(mark_positions) != 2:
                raise ValueError("形成洲后必须指定标记对方2颗棋子")
            for mr, mc in mark_positions:
                if new_state.board[mr][mc] != current_player.opponent().value or (mr, mc) in opponent_marked or is_piece_in_shape(new_state.board, mr, mc, current_player.opponent().value, opponent_marked):
                    raise ValueError(f"标记位置 ({mr}, {mc}) 不合法")
                if current_player == Player.BLACK:
                    new_state.marked_white.add((mr, mc))
                else:
                    new_state.marked_black.add((mr, mc))
        elif shape == "square":
            # 形成方，标记对方1颗棋子
            if mark_positions is None or len(mark_positions) != 1:
                raise ValueError("形成方后必须指定标记对方1颗棋子")
            mr, mc = mark_positions[0]
            if new_state.board[mr][mc] != current_player.opponent().value or (mr, mc) in opponent_marked or is_piece_in_shape(new_state.board, mr, mc, current_player.opponent().value, opponent_marked):
                raise ValueError(f"标记位置 ({mr}, {mc}) 不合法")
            if current_player == Player.BLACK:
                new_state.marked_white.add((mr, mc))
            else:
                new_state.marked_black.add((mr, mc))
        elif mark_positions is not None:
            # 如果没有形成方或洲，但传入了mark_positions参数，报错
            raise ValueError("未形成方或洲时不能标记对方棋子")
        # else: "none" 不做任何事

    # 若棋盘已经满，进入Phase 2；否则切换玩家
    if new_state.is_board_full():
        new_state.phase = Phase.REMOVAL
    else:
        new_state.switch_player()

    return new_state




def check_squares(board: List[List[int]], r: int, c: int, player_value: int, marked_set: set) -> bool:
    """
    判断以 (r, c) 这步新落子为中心，是否形成了至少一个2x2的方块(同色且未被标记).
    只要找到一个即可返回True，否则False。
    """
    size = len(board)
    for dr in [0, -1]:
        for dc in [0, -1]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < size-1 and 0 <= cc < size-1:
                cells = [(rr, cc), (rr, cc+1), (rr+1, cc), (rr+1, cc+1)]
                if all(board[x][y] == player_value and (x, y) not in marked_set for x, y in cells):
                    return True
    return False

def check_lines(board: List[List[int]], r: int, c: int, player_value: int, marked_set: set) -> bool:
    """
    检查是否形成"6连线"（洲），仅考虑水平或垂直连续6子，不考虑斜线。
    连线中的棋子不能被标记。
    """
    size = len(board)
    row = r
    col = c
    # 行
    count_in_row = 1 if (row, col) not in marked_set else 0
    cc = col - 1
    while cc >= 0 and board[row][cc] == player_value and (row, cc) not in marked_set:
        count_in_row += 1
        cc -= 1
    cc = col + 1
    while cc < size and board[row][cc] == player_value and (row, cc) not in marked_set:
        count_in_row += 1
        cc += 1
    if count_in_row >= 6:
        return True
    # 列
    count_in_col = 1 if (row, col) not in marked_set else 0
    rr = row - 1
    while rr >= 0 and board[rr][col] == player_value and (rr, col) not in marked_set:
        count_in_col += 1
        rr -= 1
    rr = row + 1
    while rr < size and board[rr][col] == player_value and (rr, col) not in marked_set:
        count_in_col += 1
        rr += 1
    if count_in_col >= 6:
        return True
    return False

def detect_shape_formed(board: List[List[int]], r: int, c: int, player_value: int, marked_set: set) -> str:
    """
    综合检测本次落子后是否至少形成一个"洲"或"方"。
    优先返回 'line' 表示洲，其次 'square' 表示方，若都无则返回 'none'。
    检测时，已被标记的棋子不参与判断。
    """
    # 检查是否形成洲
    size = len(board)
    # 横向检查
    for start_c in range(max(0, c-5), min(c+1, size-5)):
        if all(board[r][start_c+i] == player_value and (r, start_c+i) not in marked_set for i in range(6)):
            return "line"
    # 纵向检查
    for start_r in range(max(0, r-5), min(r+1, size-5)):
        if all(board[start_r+i][c] == player_value and (start_r+i, c) not in marked_set for i in range(6)):
            return "line"
    
    # 检查是否形成方
    if check_squares(board, r, c, player_value, marked_set):
        return "square"
    
    return "none"

def mark_opponent_pieces(state: GameState, count: int):
    """
    标记对方棋子时，不能标记正处于"方"或"洲"结构中的棋子。
    """
    current_player = state.current_player
    opponent_value = current_player.opponent().value

    if current_player == Player.BLACK:
        marked_set = state.marked_white
    else:
        marked_set = state.marked_black

    marked = 0
    for r in range(state.BOARD_SIZE):
        for c in range(state.BOARD_SIZE):
            if marked >= count:
                break
            if state.board[r][c] == opponent_value:
                # 还未被标记且不在特殊结构中才可标记
                if current_player == Player.BLACK:
                    if (r, c) not in state.marked_white and not is_piece_in_shape(state.board, r, c, opponent_value, state.marked_white):
                        marked_set.add((r, c))
                        marked += 1
                else:
                    if (r, c) not in state.marked_black and not is_piece_in_shape(state.board, r, c, opponent_value, state.marked_black):
                        marked_set.add((r, c))
                        marked += 1
        if marked >= count:
            break

def is_piece_in_shape(board, r, c, player_value, marked_set):
    """
    判断棋子(r, c)是否正处于某个"方"或"洲"结构中。
    注意：被标记的棋子不能作为"方"或"洲"结构的一部分。
    """
    size = len(board)
    # 检查是否在2x2方块中
    for dr in [0, -1]:
        for dc in [0, -1]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < size-1 and 0 <= cc < size-1:
                cells = [(rr, cc), (rr, cc+1), (rr+1, cc), (rr+1, cc+1)]
                # 构成方的所有棋子都必须是player_value且未被标记
                if all(board[x][y] == player_value and (x,y) not in marked_set for x, y in cells):
                    # 检查(r,c)是否是这个方的一部分
                    if (r,c) in cells:
                        return True
                        
    # 检查是否在6连线（洲）中
    # 横向
    for start_c in range(max(0, c-5), min(size-5, c+1)):
        # 构成洲的所有棋子都必须是player_value且未被标记
        if all(board[r][start_c+i] == player_value and (r, start_c+i) not in marked_set for i in range(6)):
            # 检查(r,c)是否是这个洲的一部分
            if start_c <= c < start_c + 6:
                return True
    # 纵向
    for start_r in range(max(0, r-5), min(size-5, r+1)):
        # 构成洲的所有棋子都必须是player_value且未被标记
        if all(board[start_r+i][c] == player_value and (start_r+i, c) not in marked_set for i in range(6)):
            # 检查(r,c)是否是这个洲的一部分
            if start_r <= r < start_r + 6:
                return True
    return False

def process_phase2_removals(state: GameState) -> GameState:
    """
    处理第二阶段的棋子移除逻辑：
    1. 如果双方都没有标记棋子，则进入强制互相移除阶段，由白方先行指定移除对方棋子。
    2. 如果有一方或双方标记了棋子，则移除所有被标记的棋子。
    3. 清空标记集合。
    4. 如果进行了标记棋子的移除，则进入第三阶段，由白方先行。
    """
    new_state = state.copy()

    if not new_state.marked_black and not new_state.marked_white:
        # 情况1：没有棋子被标记，进入强制移除阶段
        new_state.phase = Phase.FORCED_REMOVAL
        new_state.current_player = Player.WHITE # 白方先指定要移除的对方棋子
        new_state.forced_removals_done = 0 # 确保计数器从0开始
    else:
        # 情况2：有棋子被标记，移除它们
        removed_count = 0
        for r_idx, row in enumerate(new_state.board):
            for c_idx, _ in enumerate(row):
                if (r_idx, c_idx) in new_state.marked_black:
                    new_state.board[r_idx][c_idx] = 0 # 移除黑棋
                    removed_count += 1
                elif (r_idx, c_idx) in new_state.marked_white:
                    new_state.board[r_idx][c_idx] = 0 # 移除白棋
                    removed_count += 1
        
        new_state.marked_black.clear()
        new_state.marked_white.clear()
        
        # 只有在真正移除了标记棋子后才进入MOVEMENT阶段
        if removed_count > 0:
            new_state.phase = Phase.MOVEMENT
            new_state.current_player = Player.WHITE # 第三阶段由白方先行
        # 如果 marked_black 和 marked_white 存在，但里面的坐标在棋盘上已经变空（不太可能，除非外部逻辑错误）
        # 这种情况下，我们依然清空标记，但阶段维持不变或按原逻辑（如果棋盘满了去REMOVAL）
        # 但根据当前函数被调用的前提（棋盘满，从PLACEMENT到REMOVAL），这里应该总是进入MOVEMENT或FORCED_REMOVAL

    return new_state

def apply_forced_removal(state: GameState, piece_to_remove: Tuple[int, int]) -> GameState:
    """
    处理强制移除阶段的单步移除操作：
    当前玩家指定移除对方的一个棋子。
    1. 检查是否为 FORCED_REMOVAL 阶段。
    2. 根据 forced_removals_done 决定是哪一方在操作以及要移除哪一方的棋子。
       - forced_removals_done == 0: 白方操作，移除黑子。
       - forced_removals_done == 1: 黑方操作，移除白子。
    3. 检查指定位置的棋子是否属于对方。
    4. 检查该棋子是否不构成"方"或"洲"的一部分。
    5. 从棋盘上移除该棋子，增加 forced_removals_done 计数。
    6. 如果 forced_removals_done == 1，切换到黑方操作。
    7. 如果 forced_removals_done == 2，进入 MOVEMENT 阶段，轮到白方。
    """
    new_state = state.copy()
    r, c = piece_to_remove

    if new_state.phase != Phase.FORCED_REMOVAL:
        raise ValueError("当前不是强制移除阶段")

    if new_state.forced_removals_done == 0: # 白方移除黑子
        if new_state.current_player != Player.WHITE:
            raise ValueError("强制移除顺序错误：应为白方先移除黑子")
        if new_state.board[r][c] != Player.BLACK.value:
            raise ValueError(f"位置 ({r}, {c}) 不是对方 (黑方) 的棋子")
        
        opponent_player_value = Player.BLACK.value
        # 检查棋子是否在方或洲中 (被移除的是黑子，所以用黑子的视角检查)
        if is_piece_in_shape(new_state.board, r, c, opponent_player_value, set()): 
            raise ValueError(f"黑棋子 ({r}, {c}) 构成方或洲，不能被强制移除")

        new_state.board[r][c] = 0
        new_state.forced_removals_done = 1
        new_state.current_player = Player.BLACK # 轮到黑方移除白子

    elif new_state.forced_removals_done == 1: # 黑方移除白子
        if new_state.current_player != Player.BLACK:
            raise ValueError("强制移除顺序错误：应为黑方移除白子")
        if new_state.board[r][c] != Player.WHITE.value:
            raise ValueError(f"位置 ({r}, {c}) 不是对方 (白方) 的棋子")

        opponent_player_value = Player.WHITE.value
        # 检查棋子是否在方或洲中 (被移除的是白子，所以用白子的视角检查)
        if is_piece_in_shape(new_state.board, r, c, opponent_player_value, set()): 
            raise ValueError(f"白棋子 ({r}, {c}) 构成方或洲，不能被强制移除")

        new_state.board[r][c] = 0
        new_state.forced_removals_done = 2
        new_state.phase = Phase.MOVEMENT
        new_state.current_player = Player.WHITE # 第三阶段由白方开始
    else:
        # forced_removals_done 应该是0或1进入此函数
        raise RuntimeError(f"强制移除状态错误 (forced_removals_done={new_state.forced_removals_done})！")

    return new_state

def generate_legal_moves_phase3(state: GameState) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    生成第三阶段（走子阶段）所有合法的走子操作。
    一个走子操作由 (起始坐标, 目标坐标) 表示。
    """
    legal_moves = []
    if state.phase != Phase.MOVEMENT:
        return legal_moves

    player_value = state.current_player.value
    player_pieces = state.get_player_pieces(state.current_player)

    for r_from, c_from in player_pieces:
        # 定义四个可能的移动方向：上, 下, 左, 右
        possible_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in possible_deltas:
            r_to, c_to = r_from + dr, c_from + dc

            # 检查目标位置是否在棋盘内
            if 0 <= r_to < GameState.BOARD_SIZE and 0 <= c_to < GameState.BOARD_SIZE:
                # 检查目标位置是否为空
                if state.board[r_to][c_to] == 0:
                    legal_moves.append(((r_from, c_from), (r_to, c_to)))
    
    return legal_moves

def apply_move_phase3(
    state: GameState, 
    move: Tuple[Tuple[int, int], Tuple[int, int]], 
    capture_positions: List[Tuple[int, int]] = None,
    quiet: bool = False
) -> GameState:
    """
    在第三阶段执行一手走子和可能的提吃:
    1. 验证阶段和移动的有效性。
    2. 移动棋子。
    3. 检测移动后是否形成"方"或"洲"。
    4. 如果形成，根据 capture_positions 提吃对方棋子。
       - 提吃优先提吃对方不在"方"或"洲"中的普通棋子。
       - 若对方普通棋子不足时，可以提吃方或洲中的棋子。
    5. 检查是否有一方棋子被吃光，判定胜负。
    6. 如果游戏未结束，切换玩家。
    """
    new_state = state.copy()
    (r_from, c_from), (r_to, c_to) = move

    if new_state.phase != Phase.MOVEMENT:
        raise ValueError("当前不是走子阶段")

    current_player_value = new_state.current_player.value
    opponent_player_value = new_state.current_player.opponent().value

    # 验证移动是否合法
    if not (0 <= r_from < GameState.BOARD_SIZE and 0 <= c_from < GameState.BOARD_SIZE and \
            0 <= r_to < GameState.BOARD_SIZE and 0 <= c_to < GameState.BOARD_SIZE):
        raise ValueError("移动坐标超出棋盘范围")

    if new_state.board[r_from][c_from] != current_player_value:
        raise ValueError(f"起始位置 ({r_from}, {c_from}) 不是当前玩家的棋子")
    
    if new_state.board[r_to][c_to] != 0:
        raise ValueError(f"目标位置 ({r_to}, {c_to}) 非空，不能移动")

    # 校验是否为单步水平或垂直移动
    if not ((abs(r_from - r_to) == 1 and c_from == c_to) or \
            (abs(c_from - c_to) == 1 and r_from == r_to)):
        raise ValueError("棋子只能水平或垂直移动一格")

    # 执行移动
    new_state.board[r_to][c_to] = current_player_value
    new_state.board[r_from][c_from] = 0

    # 检测移动后的棋子 (r_to, c_to) 是否形成了方或洲
    # 在第三阶段，没有"标记棋子"的概念，所以 is_piece_in_shape 和 detect_shape_formed 的 marked_set 参数都传空 set
    shape_formed = detect_shape_formed(new_state.board, r_to, c_to, current_player_value, set())

    expected_captures = 0
    if shape_formed == "line":
        expected_captures = 2
    elif shape_formed == "square":
        expected_captures = 1

    if expected_captures > 0:
        if capture_positions is None or len(capture_positions) != expected_captures:
            raise ValueError(f"形成 {shape_formed} 后应指定提吃 {expected_captures} 颗对方棋子，但指定了 {len(capture_positions) if capture_positions else 0} 颗")
        
        # 检查每个提吃位置的合法性
        for cap_r, cap_c in capture_positions:
            if not (0 <= cap_r < GameState.BOARD_SIZE and 0 <= cap_c < GameState.BOARD_SIZE):
                raise ValueError(f"提吃位置 ({cap_r}, {cap_c}) 超出棋盘范围")
            if new_state.board[cap_r][cap_c] != opponent_player_value:
                raise ValueError(f"提吃位置 ({cap_r}, {cap_c}) 不是对方棋子")

            # 执行提吃
            new_state.board[cap_r][cap_c] = 0
                
    elif capture_positions is not None and len(capture_positions) > 0:
        raise ValueError("未形成方或洲，不能提吃棋子")

    # 检查胜负条件：对方棋子是否被吃光
    if new_state.count_player_pieces(new_state.current_player.opponent()) == 0:
        if not quiet:
            print(f"游戏结束！玩家 {new_state.current_player.name} 获胜！")
        # 可以在这里将 phase 改为 GAME_OVER 或类似状态，或者由调用方处理
        # 为了简单起见，我们先只打印信息。实际游戏中可能需要更完善的结束处理。
        return new_state # 游戏结束，不再切换玩家

    # 切换玩家
    new_state.switch_player()

    return new_state

def has_legal_moves_phase3(state: GameState) -> bool:
    """
    判断当前玩家在第三阶段是否有合法移动
    """
    if state.phase != Phase.MOVEMENT:
        raise ValueError("当前不是走子阶段")
    
    return len(generate_legal_moves_phase3(state)) > 0

def handle_no_moves_phase3(
    state: GameState,
    stucked_player_removes: Tuple[int, int],
    quiet: bool = False
) -> GameState:
    """
    处理第三阶段无子可动的情况：
    1. 当前玩家（无子可动方）移除对方一枚棋子。
       - 优先移除对方不在"方"或"洲"中的普通棋子。
       - 若对方没有普通棋子，可以移除方或洲中的棋子。
    2. 进入 COUNTER_REMOVAL 阶段，让对方反制移除。
    """
    new_state = state.copy()

    # 调用者（如 move_generator）应确保此时玩家确实无子可动
    # if has_legal_moves_phase3(new_state):
    #     raise ValueError("当前玩家有合法移动，不能触发此函数")

    r, c = stucked_player_removes
    current_player = new_state.current_player
    opponent_player = current_player.opponent()

    # 验证要移除的是对方棋子
    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError(f"移除位置 ({r}, {c}) 超出棋盘范围")

    if new_state.board[r][c] != opponent_player.value:
        raise ValueError(f"位置 ({r}, {c}) 不是对方棋子")

    # 获取对方所有普通棋子（不在方或洲中的棋子）
    opponent_normal_pieces = []
    for i in range(GameState.BOARD_SIZE):
        for j in range(GameState.BOARD_SIZE):
            if new_state.board[i][j] == opponent_player.value:
                if not is_piece_in_shape(new_state.board, i, j, opponent_player.value, set()):
                    opponent_normal_pieces.append((i, j))

    # 验证：如果对方有普通棋子，不能移除对方"方"或"洲"中的棋子
    is_in_shape = is_piece_in_shape(new_state.board, r, c, opponent_player.value, set())
    if is_in_shape and len(opponent_normal_pieces) > 0:
        raise ValueError(f"当对方有普通棋子时，不能移除对方在 ({r}, {c}) 处构成方或洲的棋子")

    # 执行移除
    new_state.board[r][c] = 0

    # 检查是否获胜
    if new_state.count_player_pieces(opponent_player) == 0:
        if not quiet:
            print(f"游戏结束！玩家 {current_player.name} 获胜！")
        return new_state # 游戏结束

    # 进入反制移除阶段，并切换玩家
    new_state.phase = Phase.COUNTER_REMOVAL
    new_state.switch_player()

    return new_state


def apply_counter_removal_phase3(
    state: GameState,
    opponent_removes: Tuple[int, int],
    quiet: bool = False
) -> GameState:
    """
    处理 COUNTER_REMOVAL 阶段：
    1. 验证当前是 COUNTER_REMOVAL 阶段。
    2. 当前玩家(remover)移除对方(stuck_player)的一枚棋子。
    3. 验证移除的合法性（是对方棋子，优先普通子等）。
    4. 执行移除。
    5. 检查胜负。
    6. 将阶段切换回 MOVEMENT。
    7. 将回合交还给对方(stuck_player)。
    """
    new_state = state.copy()
    r, c = opponent_removes

    if new_state.phase != Phase.COUNTER_REMOVAL:
        raise ValueError("当前不是反制移除阶段")

    remover_player = new_state.current_player
    stuck_player = remover_player.opponent()

    # 验证要移除的是对方（被困住方）的棋子
    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError(f"移除位置 ({r}, {c}) 超出棋盘范围")

    if new_state.board[r][c] != stuck_player.value:
        raise ValueError(f"位置 ({r}, {c}) 不是被困住玩家 ({stuck_player.name}) 的棋子")

    # 获取被困住方的所有普通棋子
    stuck_player_normal_pieces = []
    for i in range(GameState.BOARD_SIZE):
        for j in range(GameState.BOARD_SIZE):
            if new_state.board[i][j] == stuck_player.value:
                if not is_piece_in_shape(new_state.board, i, j, stuck_player.value, set()):
                    stuck_player_normal_pieces.append((i, j))

    # 验证：如果对方有普通棋子，不能移除对方"方"或"洲"中的棋子
    is_in_shape = is_piece_in_shape(new_state.board, r, c, stuck_player.value, set())
    if is_in_shape and len(stuck_player_normal_pieces) > 0:
        raise ValueError(f"当被困住的玩家有普通棋子时，不能移除其在 ({r}, {c}) 处构成方或洲的棋子")

    # 执行移除
    new_state.board[r][c] = 0

    # 检查胜负
    if new_state.count_player_pieces(stuck_player) == 0:
        if not quiet:
            print(f"游戏结束！玩家 {remover_player.name} 获胜！")
        return new_state  # 游戏结束

    # 切换回走子阶段
    new_state.phase = Phase.MOVEMENT
    # 将回合交还给被困住方
    new_state.switch_player()

    return new_state