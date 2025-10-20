from typing import List, Tuple, Dict, Any, Union
from src.game_state import GameState, Phase, Player
from src.rule_engine import (
    generate_legal_moves_phase1,
    detect_shape_formed,
    is_piece_in_shape,
    generate_legal_moves_phase3,
    has_legal_moves_phase3,
    apply_move_phase1,
    process_phase2_removals,
    apply_forced_removal,
    apply_move_phase3,
    handle_no_moves_phase3,
    apply_counter_removal_phase3,
)

# 定义动作类型
MoveType = Dict[str, Any]


def generate_all_legal_moves(state: GameState) -> List[MoveType]:
    """
    为当前游戏状态生成所有合法走法

    返回值是一个字典列表，每个字典代表一个合法走法，格式为：

    第一阶段(PLACEMENT):
    {
        'phase': Phase.PLACEMENT,
        'action_type': 'place',
        'position': (r, c),  # 落子位置
        'mark_positions': [(r1, c1), (r2, c2), ...] or None  # 标记位置，如有
    }

    第二阶段(REMOVAL):
    {
        'phase': Phase.REMOVAL,
        'action_type': 'process_removal',
    }

    强制移除阶段(FORCED_REMOVAL):
    {
        'phase': Phase.FORCED_REMOVAL,
        'action_type': 'remove',
        'position': (r, c),  # 要移除的对方棋子位置
    }

    第三阶段(MOVEMENT):
    {
        'phase': Phase.MOVEMENT,
        'action_type': 'move',
        'from_position': (r_from, c_from),  # 起始位置
        'to_position': (r_to, c_to),  # 目标位置
        'capture_positions': [(r1, c1), (r2, c2), ...] or None  # 提吃位置，如有
    }

    第三阶段无子可动时:
    {
        'phase': Phase.MOVEMENT,
        'action_type': 'no_moves_remove',
        'position': (r, c),  # 要移除的对方棋子位置
    }
    
    反制移除阶段(COUNTER_REMOVAL):
    {
        'phase': Phase.COUNTER_REMOVAL,
        'action_type': 'counter_remove',
        'position': (r, c), # 要移除的被困住方的棋子位置
    }

    注：对于标记和提吃位置，如果有多种可能的组合，会生成多个不同的走法
    """
    # 如果游戏已经结束，直接返回空列表
    if state.is_game_over():
        return []

    if state.phase == Phase.PLACEMENT:
        return _generate_moves_phase1(state)
    elif state.phase == Phase.REMOVAL:
        return _generate_moves_phase2(state)
    elif state.phase == Phase.FORCED_REMOVAL:
        return _generate_moves_forced_removal(state)
    elif state.phase == Phase.MOVEMENT:
        # 检查当前玩家是否有合法移动
        if has_legal_moves_phase3(state):
            return _generate_moves_phase3(state)
        else:
            return _generate_moves_no_moves(state)
    elif state.phase == Phase.COUNTER_REMOVAL:
        return _generate_moves_counter_removal(state)
    else:
        return []


def _generate_moves_phase1(state: GameState) -> List[MoveType]:
    """第一阶段合法走法生成"""
    legal_moves = []

    # 获取所有可落子位置
    empty_positions = generate_legal_moves_phase1(state)

    for position in empty_positions:
        r, c = position

        # 创建临时状态来测试是否形成方/洲
        temp_state = state.copy()
        temp_state.board[r][c] = temp_state.current_player.value

        # 检测是否形成方或洲
        # 被标记的棋子不参与形成方或洲
        marked_set = (
            temp_state.marked_black
            if temp_state.current_player == Player.BLACK
            else temp_state.marked_white
        )
        shape = detect_shape_formed(
            temp_state.board, r, c, temp_state.current_player.value, marked_set
        )

        if shape == "none":
            # 没有形成方或洲，只是普通落子
            legal_moves.append(
                {
                    "phase": Phase.PLACEMENT,
                    "action_type": "place",
                    "position": position,
                    "mark_positions": None,
                }
            )
        else:
            # 形成了方或洲，需要标记对方棋子
            opponent_value = temp_state.current_player.opponent().value
            opponent_pieces = []

            # 收集所有可标记的对方棋子
            for row in range(GameState.BOARD_SIZE):
                for col in range(GameState.BOARD_SIZE):
                    if temp_state.board[row][col] == opponent_value:
                        # 检查是否已被标记
                        is_marked = False
                        if temp_state.current_player == Player.BLACK:
                            is_marked = (row, col) in temp_state.marked_white
                        else:
                            is_marked = (row, col) in temp_state.marked_black

                        # 检查是否在方或洲中
                        opponent_marked = (
                            temp_state.marked_white
                            if temp_state.current_player == Player.BLACK
                            else temp_state.marked_black
                        )
                        is_in_shape = is_piece_in_shape(
                            temp_state.board, row, col, opponent_value, opponent_marked
                        )

                        # 如果没有被标记且不在方或洲中，则可以标记
                        if not is_marked and not is_in_shape:
                            opponent_pieces.append((row, col))

            # 根据形成的形状，生成所有可能的标记组合
            if shape == "square":  # 标记1颗
                if not opponent_pieces: # 如果没有可标记的棋子
                     legal_moves.append(
                        {
                            "phase": Phase.PLACEMENT,
                            "action_type": "place",
                            "position": position,
                            "mark_positions": None,
                        }
                    )
                else:
                    for piece in opponent_pieces:
                        legal_moves.append(
                            {
                                "phase": Phase.PLACEMENT,
                                "action_type": "place",
                                "position": position,
                                "mark_positions": [piece],
                            }
                        )
            elif shape == "line":  # 标记2颗
                if len(opponent_pieces) < 2: # 如果没有足够可标记的棋子
                    legal_moves.append(
                        {
                            "phase": Phase.PLACEMENT,
                            "action_type": "place",
                            "position": position,
                            "mark_positions": None,
                        }
                    )
                else:
                    for i in range(len(opponent_pieces)):
                        for j in range(i + 1, len(opponent_pieces)):
                            legal_moves.append(
                                {
                                    "phase": Phase.PLACEMENT,
                                    "action_type": "place",
                                    "position": position,
                                    "mark_positions": [
                                        opponent_pieces[i],
                                        opponent_pieces[j],
                                    ],
                                }
                            )

    return legal_moves


def _generate_moves_phase2(state: GameState) -> List[MoveType]:
    """第二阶段合法走法生成"""
    # 第二阶段只有一个操作：移除标记的棋子并进入下一阶段
    return [
        {
            "phase": Phase.REMOVAL,
            "action_type": "process_removal",
        }
    ]


def _generate_moves_forced_removal(state: GameState) -> List[MoveType]:
    """强制移除阶段合法走法生成"""
    legal_moves = []

    # 当前玩家
    current_player = state.current_player
    # 对方
    opponent_player = current_player.opponent()
    opponent_value = opponent_player.value

    # 收集对方的所有棋子
    opponent_pieces = []
    opponent_normal_pieces = []

    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == opponent_value:
                opponent_pieces.append((r, c))
                # 检查是否在方或洲中
                if not is_piece_in_shape(state.board, r, c, opponent_value, set()):
                    opponent_normal_pieces.append((r, c))

    # 优先移除不在方或洲中的普通棋子
    target_pieces = opponent_normal_pieces if opponent_normal_pieces else opponent_pieces
    
    for piece in target_pieces:
        legal_moves.append(
            {
                "phase": Phase.FORCED_REMOVAL,
                "action_type": "remove",
                "position": piece,
            }
        )

    return legal_moves


def _generate_moves_phase3(state: GameState) -> List[MoveType]:
    """第三阶段合法走法生成"""
    legal_moves = []

    # 获取所有可能的移动
    basic_moves = generate_legal_moves_phase3(state)

    for (r_from, c_from), (r_to, c_to) in basic_moves:
        # 创建临时状态来测试移动后是否形成方/洲
        temp_state = state.copy()
        temp_state.board[r_to][c_to] = temp_state.current_player.value
        temp_state.board[r_from][c_from] = 0

        # 检测是否形成方或洲
        shape = detect_shape_formed(
            temp_state.board, r_to, c_to, temp_state.current_player.value, set()
        )

        if shape == "none":
            # 没有形成方或洲，只是普通移动
            legal_moves.append(
                {
                    "phase": Phase.MOVEMENT,
                    "action_type": "move",
                    "from_position": (r_from, c_from),
                    "to_position": (r_to, c_to),
                    "capture_positions": None,
                }
            )
        else:
            # 形成了方或洲，需要提吃对方棋子
            opponent_value = temp_state.current_player.opponent().value
            opponent_pieces = []
            for r in range(GameState.BOARD_SIZE):
                for c in range(GameState.BOARD_SIZE):
                    if temp_state.board[r][c] == opponent_value:
                        opponent_pieces.append((r, c))

            # 根据形成的形状，生成所有可能的提吃组合
            if shape == "square":  # 提吃1颗
                if not opponent_pieces: # 如果没有可提吃的棋子
                    legal_moves.append(
                        {
                            "phase": Phase.MOVEMENT,
                            "action_type": "move",
                            "from_position": (r_from, c_from),
                            "to_position": (r_to, c_to),
                            "capture_positions": None,
                        }
                    )
                else:
                    for piece in opponent_pieces:
                        legal_moves.append(
                            {
                                "phase": Phase.MOVEMENT,
                                "action_type": "move",
                                "from_position": (r_from, c_from),
                                "to_position": (r_to, c_to),
                                "capture_positions": [piece],
                            }
                        )
            elif shape == "line":  # 提吃2颗
                if len(opponent_pieces) < 2: # 如果没有足够可提吃的棋子
                    legal_moves.append(
                        {
                            "phase": Phase.MOVEMENT,
                            "action_type": "move",
                            "from_position": (r_from, c_from),
                            "to_position": (r_to, c_to),
                            "capture_positions": None,
                        }
                    )
                else:
                    for i in range(len(opponent_pieces)):
                        for j in range(i + 1, len(opponent_pieces)):
                            legal_moves.append(
                                {
                                    "phase": Phase.MOVEMENT,
                                    "action_type": "move",
                                    "from_position": (r_from, c_from),
                                    "to_position": (r_to, c_to),
                                    "capture_positions": [
                                        opponent_pieces[i],
                                        opponent_pieces[j],
                                    ],
                                }
                            )

    return legal_moves


def _generate_moves_no_moves(state: GameState) -> List[MoveType]:
    """第三阶段无子可动时的合法走法生成"""
    legal_moves = []

    # 当前玩家(无法移动的玩家)
    current_player = state.current_player
    # 对方
    opponent_player = current_player.opponent()
    opponent_value = opponent_player.value

    # 收集对方的所有棋子
    opponent_pieces = []
    opponent_normal_pieces = []

    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == opponent_value:
                opponent_pieces.append((r, c))
                # 检查是否在方或洲中
                if not is_piece_in_shape(state.board, r, c, opponent_value, set()):
                    opponent_normal_pieces.append((r, c))

    # 优先移除不在方或洲中的普通棋子
    target_pieces = opponent_normal_pieces if opponent_normal_pieces else opponent_pieces
    
    for piece in target_pieces:
        legal_moves.append(
            {
                "phase": Phase.MOVEMENT,
                "action_type": "no_moves_remove",
                "position": piece,
            }
        )

    return legal_moves

def _generate_moves_counter_removal(state: GameState) -> List[MoveType]:
    """反制移除阶段合法走法生成"""
    legal_moves = []

    # 当前玩家是反制方 (remover)
    # 对方是被困住方 (stuck_player)
    remover_player = state.current_player
    stuck_player = remover_player.opponent()
    stuck_player_value = stuck_player.value

    # 收集被困住方的所有棋子
    stuck_player_pieces = []
    stuck_player_normal_pieces = []

    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == stuck_player_value:
                stuck_player_pieces.append((r, c))
                # 检查是否在方或洲中
                if not is_piece_in_shape(state.board, r, c, stuck_player_value, set()):
                    stuck_player_normal_pieces.append((r, c))

    # 优先移除不在方或洲中的普通棋子
    target_pieces = stuck_player_normal_pieces if stuck_player_normal_pieces else stuck_player_pieces

    for piece in target_pieces:
        legal_moves.append(
            {
                "phase": Phase.COUNTER_REMOVAL,
                "action_type": "counter_remove",
                "position": piece,
            }
        )
    return legal_moves


def apply_move(state: GameState, move: MoveType, quiet: bool = False) -> GameState:
    """
    应用一个走法到游戏状态

    参数:
        state: 当前游戏状态
        move: 要应用的走法，格式见 generate_all_legal_moves 函数的文档
        quiet: 是否安静模式，默认为 False

    返回:
        应用走法后的新游戏状态
    """
    if move["phase"] != state.phase:
        raise ValueError(
            f"走法阶段 {move['phase']} 与当前游戏阶段 {state.phase} 不匹配"
        )

    if state.phase == Phase.PLACEMENT:
        # 第一阶段：落子
        if move["action_type"] != "place":
            raise ValueError(
                f"在第一阶段只能执行 'place' 操作，而不是 {move['action_type']}"
            )

        new_state = apply_move_phase1(state, move["position"], move["mark_positions"])
        new_state.move_count = state.move_count + 1
        return new_state

    elif state.phase == Phase.REMOVAL:
        # 第二阶段：处理标记的棋子
        if move["action_type"] != "process_removal":
            raise ValueError(
                f"在第二阶段只能执行 'process_removal' 操作，而不是 {move['action_type']}"
            )

        new_state = process_phase2_removals(state)
        new_state.move_count = state.move_count + 1
        return new_state

    elif state.phase == Phase.FORCED_REMOVAL:
        # 强制移除阶段
        if move["action_type"] != "remove":
            raise ValueError(
                f"在强制移除阶段只能执行 'remove' 操作，而不是 {move['action_type']}"
            )

        new_state = apply_forced_removal(state, move["position"])
        new_state.move_count = state.move_count + 1
        return new_state
    
    elif state.phase == Phase.COUNTER_REMOVAL:
        # 反制移除阶段
        if move["action_type"] != "counter_remove":
            raise ValueError(
                f"在反制移除阶段只能执行 'counter_remove' 操作，而不是 {move['action_type']}"
            )
        new_state = apply_counter_removal_phase3(state, move["position"], quiet=quiet)
        new_state.move_count = state.move_count + 1
        return new_state

    elif state.phase == Phase.MOVEMENT:
        # 第三阶段：走子或处理无子可动
        if move["action_type"] == "move":
            new_state = apply_move_phase3(
                state,
                (move["from_position"], move["to_position"]),
                move["capture_positions"],
                quiet=quiet,
            )
            new_state.move_count = state.move_count + 1
            return new_state
        elif move["action_type"] == "no_moves_remove":
            # 无子可动时的移除
            new_state = handle_no_moves_phase3(
                state, move["position"], quiet=quiet
            )
            # The state is now COUNTER_REMOVAL, turn switched.
            new_state.move_count = state.move_count + 1
            return new_state
        else:
            raise ValueError(f"在第三阶段不支持操作 {move['action_type']}")

    else:
        raise ValueError(f"不支持的游戏阶段: {state.phase}")