from typing import List, Tuple

from src.game_state import GameState, Phase, Player


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def generate_placement_positions(state: GameState) -> List[Tuple[int, int]]:
    if state.phase != Phase.PLACEMENT:
        return []
    positions: List[Tuple[int, int]] = []
    for r in range(state.BOARD_SIZE):
        for c in range(state.BOARD_SIZE):
            if state.board[r][c] == 0:
                positions.append((r, c))
    return positions


def apply_placement_move(state: GameState, position: Tuple[int, int]) -> GameState:
    r, c = position
    new_state = state.copy()

    if new_state.phase != Phase.PLACEMENT:
        raise ValueError("当前不是落子阶段")

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("落子位置超出棋盘范围")

    if new_state.board[r][c] != 0:
        raise ValueError(f"位置 ({r}, {c}) 已有棋子")

    current_player = new_state.current_player
    opponent_marked = (
        new_state.marked_white if current_player == Player.BLACK else new_state.marked_black
    )
    if (r, c) in opponent_marked:
        raise ValueError(f"位置 ({r}, {c}) 已被对方标记，不能落子")

    new_state.board[r][c] = current_player.value

    # 检测是否形成形状
    own_marked = (
        new_state.marked_black if current_player == Player.BLACK else new_state.marked_white
    )
    already_marked = (r, c) in own_marked
    if not already_marked:
        shape = detect_shape_formed(
            new_state.board,
            r,
            c,
            current_player.value,
            own_marked,
        )
        if shape == "line":
            _set_pending_marks(new_state, required=2)
            new_state.phase = Phase.MARK_SELECTION
            return new_state
        if shape == "square":
            _set_pending_marks(new_state, required=1)
            new_state.phase = Phase.MARK_SELECTION
            return new_state

    _clear_pending_marks(new_state)

    if new_state.is_board_full():
        new_state.phase = Phase.REMOVAL
    else:
        new_state.switch_player()
        new_state.phase = Phase.PLACEMENT

    return new_state


def generate_mark_targets(state: GameState) -> List[Tuple[int, int]]:
    if state.phase != Phase.MARK_SELECTION:
        return []
    if state.pending_marks_remaining <= 0:
        return []

    current_player = state.current_player
    opponent = current_player.opponent()
    opponent_value = opponent.value
    opponent_marked = state.marked_white if opponent == Player.WHITE else state.marked_black

    opponent_pieces: List[Tuple[int, int]] = []
    opponent_normal_pieces: List[Tuple[int, int]] = []
    for r in range(state.BOARD_SIZE):
        for c in range(state.BOARD_SIZE):
            if state.board[r][c] == opponent_value:
                pos = (r, c)
                opponent_pieces.append(pos)
                if not is_piece_in_shape(state.board, r, c, opponent_value, opponent_marked):
                    opponent_normal_pieces.append(pos)

    pool = (
        [pos for pos in opponent_normal_pieces if pos not in opponent_marked]
        if opponent_normal_pieces
        else [pos for pos in opponent_pieces if pos not in opponent_marked]
    )
    return pool


def apply_mark_selection(state: GameState, position: Tuple[int, int]) -> GameState:
    new_state = state.copy()
    if new_state.phase != Phase.MARK_SELECTION:
        raise ValueError("当前不是标记选择阶段")
    if new_state.pending_marks_remaining <= 0:
        raise ValueError("没有待完成的标记任务")

    r, c = position
    opponent = new_state.current_player.opponent()
    opponent_value = opponent.value
    opponent_marked = new_state.marked_white if opponent == Player.WHITE else new_state.marked_black

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("标记位置超出棋盘范围")
    if new_state.board[r][c] != opponent_value:
        raise ValueError("只能标记对方棋子")
    if (r, c) in opponent_marked:
        raise ValueError("该棋子已经被标记")

    # 如果对方还有未构成形状的普通棋子，则不能标记形状中的棋子
    opponent_normal_pieces = [
        (rr, cc)
        for rr in range(GameState.BOARD_SIZE)
        for cc in range(GameState.BOARD_SIZE)
        if new_state.board[rr][cc] == opponent_value
        and not is_piece_in_shape(new_state.board, rr, cc, opponent_value, opponent_marked)
    ]
    if opponent_normal_pieces and is_piece_in_shape(
        new_state.board, r, c, opponent_value, opponent_marked
    ):
        raise ValueError("对方存在普通棋子时，不能标记其关键结构中的棋子")

    if opponent == Player.WHITE:
        new_state.marked_white.add((r, c))
    else:
        new_state.marked_black.add((r, c))

    new_state.pending_marks_remaining -= 1
    if new_state.pending_marks_remaining > 0:
        return new_state

    _clear_pending_marks(new_state)

    if new_state.is_board_full():
        new_state.phase = Phase.REMOVAL
    else:
        new_state.switch_player()
        new_state.phase = Phase.PLACEMENT

    return new_state


def process_phase2_removals(state: GameState) -> GameState:
    new_state = state.copy()
    if new_state.phase != Phase.REMOVAL:
        raise ValueError("当前不是移除阶段")

    if not new_state.marked_black and not new_state.marked_white:
        new_state.phase = Phase.FORCED_REMOVAL
        new_state.current_player = Player.WHITE
        new_state.forced_removals_done = 0
        return new_state

    removed = 0
    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            pos = (r, c)
            if pos in new_state.marked_black:
                new_state.board[r][c] = 0
                removed += 1
            elif pos in new_state.marked_white:
                new_state.board[r][c] = 0
                removed += 1

    new_state.marked_black.clear()
    new_state.marked_white.clear()

    if removed > 0:
        new_state.phase = Phase.MOVEMENT
        new_state.current_player = Player.WHITE

    return new_state


def generate_movement_moves(state: GameState) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if state.phase != Phase.MOVEMENT:
        return []

    player_value = state.current_player.value
    moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] != player_value:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GameState.BOARD_SIZE and 0 <= nc < GameState.BOARD_SIZE:
                    if state.board[nr][nc] == 0:
                        moves.append(((r, c), (nr, nc)))
    return moves


def has_legal_movement_moves(state: GameState) -> bool:
    if state.phase != Phase.MOVEMENT:
        raise ValueError("当前不是走子阶段")
    return bool(generate_movement_moves(state))


def apply_movement_move(
    state: GameState,
    move: Tuple[Tuple[int, int], Tuple[int, int]],
    quiet: bool = False,
) -> GameState:
    (r_from, c_from), (r_to, c_to) = move
    new_state = state.copy()
    if new_state.phase != Phase.MOVEMENT:
        raise ValueError("当前不是走子阶段")

    if new_state.board[r_from][c_from] != new_state.current_player.value:
        raise ValueError("起始位置不是当前玩家的棋子")
    if new_state.board[r_to][c_to] != 0:
        raise ValueError("目标位置不是空位")
    if not ((abs(r_from - r_to) == 1 and c_from == c_to) or (abs(c_from - c_to) == 1 and r_from == r_to)):
        raise ValueError("只能水平或垂直移动一格")

    new_state.board[r_to][c_to] = new_state.board[r_from][c_from]
    new_state.board[r_from][c_from] = 0

    current_player = new_state.current_player
    shape = detect_shape_formed(
        new_state.board,
        r_to,
        c_to,
        current_player.value,
        set(),
    )

    if shape == "line":
        _set_pending_captures(new_state, required=2)
        new_state.phase = Phase.CAPTURE_SELECTION
        return new_state
    if shape == "square":
        _set_pending_captures(new_state, required=1)
        new_state.phase = Phase.CAPTURE_SELECTION
        return new_state

    _clear_pending_captures(new_state)
    new_state.switch_player()
    return new_state


def generate_capture_targets(state: GameState) -> List[Tuple[int, int]]:
    if state.phase != Phase.CAPTURE_SELECTION:
        return []
    if state.pending_captures_remaining <= 0:
        return []

    opponent = state.current_player.opponent()
    opponent_value = opponent.value
    opponent_marked = state.marked_white if opponent == Player.WHITE else state.marked_black

    opponent_pieces: List[Tuple[int, int]] = []
    opponent_normal_pieces: List[Tuple[int, int]] = []
    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == opponent_value:
                pos = (r, c)
                opponent_pieces.append(pos)
                if not is_piece_in_shape(state.board, r, c, opponent_value, opponent_marked):
                    opponent_normal_pieces.append(pos)

    pool = opponent_normal_pieces if opponent_normal_pieces else opponent_pieces
    return pool


def apply_capture_selection(
    state: GameState,
    position: Tuple[int, int],
    quiet: bool = False,
) -> GameState:
    new_state = state.copy()
    if new_state.phase != Phase.CAPTURE_SELECTION:
        raise ValueError("当前不是提子阶段")
    if new_state.pending_captures_remaining <= 0:
        raise ValueError("没有待完成的提子任务")

    r, c = position
    opponent = new_state.current_player.opponent()
    opponent_value = opponent.value
    opponent_marked = new_state.marked_white if opponent == Player.WHITE else new_state.marked_black

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("位置超出棋盘范围")
    if new_state.board[r][c] != opponent_value:
        raise ValueError("只能提掉对方棋子")

    opponent_normal_pieces = [
        (rr, cc)
        for rr in range(GameState.BOARD_SIZE)
        for cc in range(GameState.BOARD_SIZE)
        if new_state.board[rr][cc] == opponent_value
        and not is_piece_in_shape(new_state.board, rr, cc, opponent_value, opponent_marked)
    ]
    if opponent_normal_pieces and is_piece_in_shape(
        new_state.board, r, c, opponent_value, opponent_marked
    ):
        raise ValueError("对方还有普通棋子时，不能提关键结构中的棋子")

    new_state.board[r][c] = 0
    new_state.pending_captures_remaining -= 1

    if new_state.count_player_pieces(opponent) == 0:
        if not quiet:
            print(f"游戏结束！玩家 {new_state.current_player.name} 获胜！")
        return new_state

    if new_state.pending_captures_remaining > 0:
        return new_state

    _clear_pending_captures(new_state)
    new_state.switch_player()
    new_state.phase = Phase.MOVEMENT
    return new_state


def apply_forced_removal(state: GameState, piece_to_remove: Tuple[int, int]) -> GameState:
    new_state = state.copy()
    r, c = piece_to_remove

    if new_state.phase != Phase.FORCED_REMOVAL:
        raise ValueError("当前不是强制移除阶段")

    if new_state.forced_removals_done == 0:
        if new_state.current_player != Player.WHITE:
            raise ValueError("强制移除顺序错误：应由白方先移除黑子")
        if new_state.board[r][c] != Player.BLACK.value:
            raise ValueError("必须移除黑方棋子")
        if is_piece_in_shape(new_state.board, r, c, Player.BLACK.value, set()):
            raise ValueError("构成方或洲的棋子不能被强制移除")
        new_state.board[r][c] = 0
        new_state.forced_removals_done = 1
        new_state.current_player = Player.BLACK
    elif new_state.forced_removals_done == 1:
        if new_state.current_player != Player.BLACK:
            raise ValueError("强制移除顺序错误：应由黑方移除白子")
        if new_state.board[r][c] != Player.WHITE.value:
            raise ValueError("必须移除白方棋子")
        if is_piece_in_shape(new_state.board, r, c, Player.WHITE.value, set()):
            raise ValueError("构成方或洲的棋子不能被强制移除")
        new_state.board[r][c] = 0
        new_state.forced_removals_done = 2
        new_state.phase = Phase.MOVEMENT
        new_state.current_player = Player.WHITE
    else:
        raise RuntimeError("强制移除状态异常")

    return new_state


def handle_no_moves_phase3(
    state: GameState,
    stucked_player_removes: Tuple[int, int],
    quiet: bool = False,
) -> GameState:
    new_state = state.copy()
    if new_state.phase != Phase.MOVEMENT:
        raise ValueError("无子可动处理只能在走子阶段触发")

    r, c = stucked_player_removes
    current_player = new_state.current_player
    opponent = current_player.opponent()

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("位置超出棋盘范围")
    if new_state.board[r][c] != opponent.value:
        raise ValueError("只能移除对方棋子")

    opponent_normal_pieces = [
        (rr, cc)
        for rr in range(GameState.BOARD_SIZE)
        for cc in range(GameState.BOARD_SIZE)
        if new_state.board[rr][cc] == opponent.value
        and not is_piece_in_shape(new_state.board, rr, cc, opponent.value, set())
    ]
    if opponent_normal_pieces and is_piece_in_shape(
        new_state.board, r, c, opponent.value, set()
    ):
        raise ValueError("对方尚有普通棋子，不能移除结构中的棋子")

    new_state.board[r][c] = 0

    if new_state.count_player_pieces(opponent) == 0:
        if not quiet:
            print(f"游戏结束！玩家 {current_player.name} 获胜！")
        return new_state

    new_state.phase = Phase.COUNTER_REMOVAL
    new_state.switch_player()
    return new_state


def apply_counter_removal_phase3(
    state: GameState,
    opponent_removes: Tuple[int, int],
    quiet: bool = False,
) -> GameState:
    new_state = state.copy()
    r, c = opponent_removes

    if new_state.phase != Phase.COUNTER_REMOVAL:
        raise ValueError("当前不是反制移除阶段")

    remover = new_state.current_player
    stuck_player = remover.opponent()

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("位置超出棋盘范围")
    if new_state.board[r][c] != stuck_player.value:
        raise ValueError("只能移除被困住玩家的棋子")

    stuck_player_normal_pieces = [
        (rr, cc)
        for rr in range(GameState.BOARD_SIZE)
        for cc in range(GameState.BOARD_SIZE)
        if new_state.board[rr][cc] == stuck_player.value
        and not is_piece_in_shape(new_state.board, rr, cc, stuck_player.value, set())
    ]
    if stuck_player_normal_pieces and is_piece_in_shape(
        new_state.board, r, c, stuck_player.value, set()
    ):
        raise ValueError("对方尚有普通棋子，不能移除结构中的棋子")

    new_state.board[r][c] = 0

    if new_state.count_player_pieces(stuck_player) == 0:
        if not quiet:
            print(f"游戏结束！玩家 {remover.name} 获胜！")
        return new_state

    new_state.phase = Phase.MOVEMENT
    new_state.switch_player()
    return new_state


# ---------------------------------------------------------------------------
# Shape detection utilities (copied from previous implementation)
# ---------------------------------------------------------------------------


def detect_shape_formed(
    board: List[List[int]],
    r: int,
    c: int,
    player_value: int,
    marked_set: set,
) -> str:
    if check_squares(board, r, c, player_value, marked_set):
        return "square"
    if check_lines(board, r, c, player_value, marked_set):
        return "line"
    return "none"


def check_squares(
    board: List[List[int]],
    r: int,
    c: int,
    player_value: int,
    marked_set: set,
) -> bool:
    size = len(board)
    for dr in [0, -1]:
        for dc in [0, -1]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < size - 1 and 0 <= cc < size - 1:
                cells = [(rr, cc), (rr, cc + 1), (rr + 1, cc), (rr + 1, cc + 1)]
                if all(board[x][y] == player_value and (x, y) not in marked_set for x, y in cells):
                    return True
    return False


def check_lines(
    board: List[List[int]],
    r: int,
    c: int,
    player_value: int,
    marked_set: set,
) -> bool:
    size = len(board)

    # 水平检查
    count = 1
    for dc in range(c - 1, -1, -1):
        if board[r][dc] == player_value and (r, dc) not in marked_set:
            count += 1
        else:
            break
    for dc in range(c + 1, size):
        if board[r][dc] == player_value and (r, dc) not in marked_set:
            count += 1
        else:
            break
    if count >= 6:
        return True

    # 垂直检查
    count = 1
    for dr in range(r - 1, -1, -1):
        if board[dr][c] == player_value and (dr, c) not in marked_set:
            count += 1
        else:
            break
    for dr in range(r + 1, size):
        if board[dr][c] == player_value and (dr, c) not in marked_set:
            count += 1
        else:
            break
    return count >= 6


def is_piece_in_shape(
    board: List[List[int]],
    r: int,
    c: int,
    player_value: int,
    marked_set: set,
) -> bool:
    if board[r][c] != player_value:
        return False
    return check_squares(board, r, c, player_value, marked_set) or check_lines(
        board, r, c, player_value, marked_set
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_pending_marks(state: GameState, required: int):
    state.pending_marks_required = required
    state.pending_marks_remaining = required


def _clear_pending_marks(state: GameState):
    state.pending_marks_required = 0
    state.pending_marks_remaining = 0


def _set_pending_captures(state: GameState, required: int):
    state.pending_captures_required = required
    state.pending_captures_remaining = required


def _clear_pending_captures(state: GameState):
    state.pending_captures_required = 0
    state.pending_captures_remaining = 0


# ---------------------------------------------------------------------------
# Backward compatibility wrappers
# ---------------------------------------------------------------------------


def generate_legal_moves_phase1(state: GameState) -> List[Tuple[int, int]]:
    return generate_placement_positions(state)


def apply_move_phase1(
    state: GameState,
    move: Tuple[int, int],
    mark_positions: List[Tuple[int, int]] = None,
) -> GameState:
    new_state = apply_placement_move(state, move)
    if mark_positions:
        if new_state.phase != Phase.MARK_SELECTION:
            raise ValueError("当前状态不需要标记，但传入了 mark_positions")
        for pos in mark_positions:
            new_state = apply_mark_selection(new_state, pos)
    return new_state


def generate_legal_moves_phase3(
    state: GameState,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    return generate_movement_moves(state)


def has_legal_moves_phase3(state: GameState) -> bool:
    return has_legal_movement_moves(state)


def apply_move_phase3(
    state: GameState,
    move: Tuple[Tuple[int, int], Tuple[int, int]],
    capture_positions: List[Tuple[int, int]] = None,
    quiet: bool = False,
) -> GameState:
    new_state = apply_movement_move(state, move, quiet=quiet)
    if capture_positions:
        if new_state.phase != Phase.CAPTURE_SELECTION:
            raise ValueError("当前状态不需要提子，但传入了 capture_positions")
        for pos in capture_positions:
            new_state = apply_capture_selection(new_state, pos, quiet=quiet)
    return new_state
