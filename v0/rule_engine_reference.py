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
        raise ValueError("褰撳墠涓嶆槸钀藉瓙闃舵")

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("钀藉瓙浣嶇疆瓒呭嚭妫嬬洏鑼冨洿")

    if new_state.board[r][c] != 0:
        raise ValueError(f"浣嶇疆 ({r}, {c}) 宸叉湁妫嬪瓙")

    current_player = new_state.current_player
    opponent_marked = (
        new_state.marked_white if current_player == Player.BLACK else new_state.marked_black
    )
    if (r, c) in opponent_marked:
        raise ValueError(f"浣嶇疆 ({r}, {c}) 宸茶瀵规柟鏍囪锛屼笉鑳借惤")

    new_state.board[r][c] = current_player.value

    # 妫€娴嬫槸鍚﹀舰鎴愬舰"    
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
    if not pool:
        # All normal pieces have been marked already; fall back to any remaining piece.
        pool = [pos for pos in opponent_pieces if pos not in opponent_marked]
    return pool


def apply_mark_selection(state: GameState, position: Tuple[int, int]) -> GameState:
    new_state = state.copy()
    if new_state.phase != Phase.MARK_SELECTION:
        raise ValueError("褰撳墠涓嶆槸鏍囪閫夋嫨闃舵")
    if new_state.pending_marks_remaining <= 0:
        raise ValueError("娌℃湁寰呭畬鎴愮殑鏍囪浠诲姟")

    r, c = position
    opponent = new_state.current_player.opponent()
    opponent_value = opponent.value
    opponent_marked = new_state.marked_white if opponent == Player.WHITE else new_state.marked_black

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("鏍囪浣嶇疆瓒呭嚭妫嬬洏鑼冨洿")
    if new_state.board[r][c] != opponent_value:
        raise ValueError("鍙兘鏍囪瀵规柟妫嬪瓙")
    if (r, c) in opponent_marked:
        raise ValueError("璇ユ瀛愬凡缁忚鏍囪")

    # 濡傛灉瀵规柟杩樻湁鏈瀯鎴愬舰鐘剁殑鏅€氭瀛愶紝鍒欎笉鑳芥爣璁板舰鐘朵腑鐨勬瀛?
    opponent_normal_pieces = [
        (rr, cc)
        for rr in range(GameState.BOARD_SIZE)
        for cc in range(GameState.BOARD_SIZE)
        if new_state.board[rr][cc] == opponent_value
        and not is_piece_in_shape(new_state.board, rr, cc, opponent_value, opponent_marked)
    ]
    opponent_normal_unmarked = [
        pos for pos in opponent_normal_pieces if pos not in opponent_marked
    ]
    if opponent_normal_unmarked and is_piece_in_shape(
        new_state.board, r, c, opponent_value, opponent_marked
    ):
        raise ValueError("瀵规柟瀛樺湪鏅€氭瀛愭椂锛屼笉鑳芥爣璁板叾鍏抽敭缁撴瀯涓殑妫嬪瓙")

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
        raise ValueError("褰撳墠涓嶆槸绉婚櫎闃舵")

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
        raise ValueError("褰撳墠涓嶆槸璧板瓙闃舵")
    return bool(generate_movement_moves(state))


def apply_movement_move(
    state: GameState,
    move: Tuple[Tuple[int, int], Tuple[int, int]],
    quiet: bool = False,
) -> GameState:
    (r_from, c_from), (r_to, c_to) = move
    new_state = state.copy()
    if new_state.phase != Phase.MOVEMENT:
        raise ValueError("褰撳墠涓嶆槸璧板瓙闃舵")

    if new_state.board[r_from][c_from] != new_state.current_player.value:
        raise ValueError("璧峰浣嶇疆涓嶆槸褰撳墠鐜╁鐨勬")
    if new_state.board[r_to][c_to] != 0:
        raise ValueError("鐩爣浣嶇疆涓嶆槸绌轰綅")
    if not ((abs(r_from - r_to) == 1 and c_from == c_to) or (abs(c_from - c_to) == 1 and r_from == r_to)):
        raise ValueError("鍙兘姘村钩鎴栧瀭鐩寸Щ鍔ㄤ竴")

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
        raise ValueError("褰撳墠涓嶆槸鎻愬瓙闃舵")
    if new_state.pending_captures_remaining <= 0:
        raise ValueError("娌℃湁寰呭畬鎴愮殑鎻愬瓙浠诲姟")

    r, c = position
    opponent = new_state.current_player.opponent()
    opponent_value = opponent.value
    opponent_marked = new_state.marked_white if opponent == Player.WHITE else new_state.marked_black

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("浣嶇疆瓒呭嚭妫嬬洏鑼冨洿")
    if new_state.board[r][c] != opponent_value:
        raise ValueError("鍙兘鎻愭帀瀵规柟妫嬪瓙")

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
        raise ValueError("瀵规柟杩樻湁鏅€氭瀛愭椂锛屼笉鑳芥彁鍏抽敭缁撴瀯涓殑妫嬪瓙")

    new_state.board[r][c] = 0
    new_state.pending_captures_remaining -= 1

    if new_state.count_player_pieces(opponent) < GameState.LOSE_PIECE_THRESHOLD:
        if not quiet:
            print(f"Game over! Player {new_state.current_player.name} wins")
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
        raise ValueError("褰撳墠涓嶆槸寮哄埗绉婚櫎闃舵")

    if new_state.forced_removals_done == 0:
        if new_state.current_player != Player.WHITE:
            raise ValueError("寮哄埗绉婚櫎椤哄簭閿欒锛氬簲鐢辩櫧鏂瑰厛绉婚櫎榛戝瓙")
        if new_state.board[r][c] != Player.BLACK.value:
            raise ValueError("蹇呴』绉婚櫎榛戞柟妫嬪瓙")
        if is_piece_in_shape(new_state.board, r, c, Player.BLACK.value, set()):
            raise ValueError("鏋勬垚鏂规垨娲茬殑妫嬪瓙涓嶈兘琚己鍒剁Щ")
        new_state.board[r][c] = 0
        new_state.forced_removals_done = 1
        new_state.current_player = Player.BLACK
    elif new_state.forced_removals_done == 1:
        if new_state.current_player != Player.BLACK:
            raise ValueError("寮哄埗绉婚櫎椤哄簭閿欒锛氬簲鐢遍粦鏂圭Щ闄ょ櫧")
        if new_state.board[r][c] != Player.WHITE.value:
            raise ValueError("蹇呴』绉婚櫎鐧芥柟妫嬪瓙")
        if is_piece_in_shape(new_state.board, r, c, Player.WHITE.value, set()):
            raise ValueError("鏋勬垚鏂规垨娲茬殑妫嬪瓙涓嶈兘琚己鍒剁Щ")
        new_state.board[r][c] = 0
        new_state.forced_removals_done = 2
        new_state.phase = Phase.MOVEMENT
        new_state.current_player = Player.WHITE
    else:
        raise RuntimeError("寮哄埗绉婚櫎鐘舵€佸紓")

    return new_state


def handle_no_moves_phase3(
    state: GameState,
    stucked_player_removes: Tuple[int, int],
    quiet: bool = False,
) -> GameState:
    new_state = state.copy()
    if new_state.phase != Phase.MOVEMENT:
        raise ValueError("鏃犲瓙鍙姩澶勭悊鍙兘鍦ㄨ蛋瀛愰樁娈佃Е")

    r, c = stucked_player_removes
    current_player = new_state.current_player
    opponent = current_player.opponent()

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("浣嶇疆瓒呭嚭妫嬬洏鑼冨洿")
    if new_state.board[r][c] != opponent.value:
        raise ValueError("鍙兘绉婚櫎瀵规柟妫嬪瓙")

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
        raise ValueError("瀵规柟灏氭湁鏅€氭瀛愶紝涓嶈兘绉婚櫎缁撴瀯涓殑妫嬪瓙")

    new_state.board[r][c] = 0

    if new_state.count_player_pieces(opponent) < GameState.LOSE_PIECE_THRESHOLD:
        if not quiet:
            print(f"Game over! Player {current_player.name} wins")
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
        raise ValueError("褰撳墠涓嶆槸鍙嶅埗绉婚櫎闃舵")

    remover = new_state.current_player
    stuck_player = remover.opponent()

    if not (0 <= r < GameState.BOARD_SIZE and 0 <= c < GameState.BOARD_SIZE):
        raise ValueError("浣嶇疆瓒呭嚭妫嬬洏鑼冨洿")
    if new_state.board[r][c] != stuck_player.value:
        raise ValueError("鍙兘绉婚櫎琚洶浣忕帺瀹剁殑妫嬪瓙")

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
        raise ValueError("瀵规柟灏氭湁鏅€氭瀛愶紝涓嶈兘绉婚櫎缁撴瀯涓殑妫嬪瓙")

    new_state.board[r][c] = 0

    if new_state.count_player_pieces(stuck_player) < GameState.LOSE_PIECE_THRESHOLD:
        if not quiet:
            print(f"Game over! Player {remover.name} wins")
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
    found_square = check_squares(board, r, c, player_value, marked_set)
    found_line = check_lines(board, r, c, player_value, marked_set)

    if found_line:
        return "line"
    if found_square:
        return "square"
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

    # 姘村钩妫€"    count = 1
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

    # 鍨傜洿妫€"    count = 1
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
            raise ValueError("褰撳墠鐘舵€佷笉闇€瑕佹爣璁帮紝浣嗕紶鍏ヤ簡 mark_positions")
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
            raise ValueError("褰撳墠鐘舵€佷笉闇€瑕佹彁瀛愶紝浣嗕紶鍏ヤ簡 capture_positions")
        for pos in capture_positions:
            new_state = apply_capture_selection(new_state, pos, quiet=quiet)
    return new_state

