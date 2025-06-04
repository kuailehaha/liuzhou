from typing import List, Tuple, Dict, Any, Union
from src.game_state import GameState, Phase, Player
from src.rule_engine import (
    generate_legal_moves_phase1, detect_shape_formed, is_piece_in_shape,
    generate_legal_moves_phase3, has_legal_moves_phase3,
    apply_move_phase1, process_phase2_removals, apply_forced_removal,
    apply_move_phase3, handle_no_moves_phase3, apply_counter_removal_phase3
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
    
    注：对于标记和提吃位置，如果有多种可能的组合，会生成多个不同的走法
    """
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
        marked_set = temp_state.marked_black if temp_state.current_player == Player.BLACK else temp_state.marked_white
        shape = detect_shape_formed(temp_state.board, r, c, temp_state.current_player.value, marked_set)
        
        if shape == "none":
            # 没有形成方或洲，只是普通落子
            legal_moves.append({
                'phase': Phase.PLACEMENT,
                'action_type': 'place',
                'position': position,
                'mark_positions': None
            })
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
                        opponent_marked = temp_state.marked_white if temp_state.current_player == Player.BLACK else temp_state.marked_black
                        is_in_shape = is_piece_in_shape(temp_state.board, row, col, opponent_value, opponent_marked)
                        
                        # 如果没有被标记且不在方或洲中，则可以标记
                        if not is_marked and not is_in_shape:
                            opponent_pieces.append((row, col))
            
            # 根据形成的形状，生成所有可能的标记组合
            if shape == "square":  # 标记1颗
                for piece in opponent_pieces:
                    legal_moves.append({
                        'phase': Phase.PLACEMENT,
                        'action_type': 'place',
                        'position': position,
                        'mark_positions': [piece]
                    })
            elif shape == "line":  # 标记2颗
                if len(opponent_pieces) >= 2:
                    for i in range(len(opponent_pieces)):
                        for j in range(i+1, len(opponent_pieces)):
                            legal_moves.append({
                                'phase': Phase.PLACEMENT,
                                'action_type': 'place',
                                'position': position,
                                'mark_positions': [opponent_pieces[i], opponent_pieces[j]]
                            })
    
    return legal_moves

def _generate_moves_phase2(state: GameState) -> List[MoveType]:
    """第二阶段合法走法生成"""
    # 第二阶段只有一个操作：移除标记的棋子并进入下一阶段
    return [{
        'phase': Phase.REMOVAL,
        'action_type': 'process_removal',
    }]

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
    if opponent_normal_pieces:
        for piece in opponent_normal_pieces:
            legal_moves.append({
                'phase': Phase.FORCED_REMOVAL,
                'action_type': 'remove',
                'position': piece
            })
    else:
        # 如果没有普通棋子，可以移除在方或洲中的棋子
        for piece in opponent_pieces:
            legal_moves.append({
                'phase': Phase.FORCED_REMOVAL,
                'action_type': 'remove',
                'position': piece
            })
    
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
        shape = detect_shape_formed(temp_state.board, r_to, c_to, temp_state.current_player.value, set())
        
        if shape == "none":
            # 没有形成方或洲，只是普通移动
            legal_moves.append({
                'phase': Phase.MOVEMENT,
                'action_type': 'move',
                'from_position': (r_from, c_from),
                'to_position': (r_to, c_to),
                'capture_positions': None
            })
        else:
            # 形成了方或洲，需要提吃对方棋子
            opponent_value = temp_state.current_player.opponent().value
            opponent_pieces = []
            opponent_normal_pieces = []
            opponent_shape_pieces = []
            
            # 收集所有可提吃的对方棋子
            for row in range(GameState.BOARD_SIZE):
                for col in range(GameState.BOARD_SIZE):
                    if temp_state.board[row][col] == opponent_value:
                        opponent_pieces.append((row, col))
                        # 检查是否在方或洲中
                        if is_piece_in_shape(temp_state.board, row, col, opponent_value, set()):
                            opponent_shape_pieces.append((row, col))
                        else:
                            opponent_normal_pieces.append((row, col))
            
            # 根据形状和对方棋子情况，生成所有可能的提吃组合
            if shape == "square":  # 提吃1颗
                if opponent_normal_pieces:
                    # 有普通棋子，只能提吃普通棋子
                    for piece in opponent_normal_pieces:
                        legal_moves.append({
                            'phase': Phase.MOVEMENT,
                            'action_type': 'move',
                            'from_position': (r_from, c_from),
                            'to_position': (r_to, c_to),
                            'capture_positions': [piece]
                        })
                elif opponent_shape_pieces:
                    # 没有普通棋子，可以提吃方/洲中的棋子
                    for piece in opponent_shape_pieces:
                        legal_moves.append({
                            'phase': Phase.MOVEMENT,
                            'action_type': 'move',
                            'from_position': (r_from, c_from),
                            'to_position': (r_to, c_to),
                            'capture_positions': [piece]
                        })
            
            elif shape == "line":  # 提吃2颗
                if len(opponent_normal_pieces) >= 2:
                    # 有足够的普通棋子，只能提吃普通棋子
                    for i in range(len(opponent_normal_pieces)):
                        for j in range(i+1, len(opponent_normal_pieces)):
                            legal_moves.append({
                                'phase': Phase.MOVEMENT,
                                'action_type': 'move',
                                'from_position': (r_from, c_from),
                                'to_position': (r_to, c_to),
                                'capture_positions': [opponent_normal_pieces[i], opponent_normal_pieces[j]]
                            })
                
                elif len(opponent_normal_pieces) == 1 and opponent_shape_pieces:
                    # 有1颗普通棋子，需要提吃这颗普通棋子和1颗方/洲中的棋子
                    normal_piece = opponent_normal_pieces[0]
                    for shape_piece in opponent_shape_pieces:
                        legal_moves.append({
                            'phase': Phase.MOVEMENT,
                            'action_type': 'move',
                            'from_position': (r_from, c_from),
                            'to_position': (r_to, c_to),
                            'capture_positions': [normal_piece, shape_piece]
                        })
                
                elif len(opponent_normal_pieces) == 0 and len(opponent_shape_pieces) >= 2:
                    # 没有普通棋子，可以提吃2颗方/洲中的棋子
                    for i in range(len(opponent_shape_pieces)):
                        for j in range(i+1, len(opponent_shape_pieces)):
                            legal_moves.append({
                                'phase': Phase.MOVEMENT,
                                'action_type': 'move',
                                'from_position': (r_from, c_from),
                                'to_position': (r_to, c_to),
                                'capture_positions': [opponent_shape_pieces[i], opponent_shape_pieces[j]]
                            })
    
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
    if opponent_normal_pieces:
        for piece in opponent_normal_pieces:
            legal_moves.append({
                'phase': Phase.MOVEMENT,
                'action_type': 'no_moves_remove',
                'position': piece
            })
    else:
        # 如果没有普通棋子，可以移除在方或洲中的棋子
        for piece in opponent_pieces:
            legal_moves.append({
                'phase': Phase.MOVEMENT,
                'action_type': 'no_moves_remove',
                'position': piece
            })
    
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
    if move['phase'] != state.phase:
        raise ValueError(f"走法阶段 {move['phase']} 与当前游戏阶段 {state.phase} 不匹配")
    
    if state.phase == Phase.PLACEMENT:
        # 第一阶段：落子
        if move['action_type'] != 'place':
            raise ValueError(f"在第一阶段只能执行 'place' 操作，而不是 {move['action_type']}")
        
        return apply_move_phase1(state, move['position'], move['mark_positions'])
    
    elif state.phase == Phase.REMOVAL:
        # 第二阶段：处理标记的棋子
        if move['action_type'] != 'process_removal':
            raise ValueError(f"在第二阶段只能执行 'process_removal' 操作，而不是 {move['action_type']}")
        
        return process_phase2_removals(state)
    
    elif state.phase == Phase.FORCED_REMOVAL:
        # 强制移除阶段
        if move['action_type'] != 'remove':
            raise ValueError(f"在强制移除阶段只能执行 'remove' 操作，而不是 {move['action_type']}")
        
        return apply_forced_removal(state, move['position'])
    
    elif state.phase == Phase.MOVEMENT:
        # 第三阶段：走子或处理无子可动
        if move['action_type'] == 'move':
            return apply_move_phase3(
                state, 
                (move['from_position'], move['to_position']), 
                move['capture_positions'],
                quiet=quiet
            )
        elif move['action_type'] == 'no_moves_remove':
            # 无子可动时的移除
            state_after_remove = handle_no_moves_phase3(state, move['position'], quiet=quiet)
            
            # 模拟对方的反制移除
            # 这里我们需要知道对方会选择移除哪个棋子
            # 在真实游戏中，这应该由对方玩家决定
            # 在自动模拟中，我们可以选择一个合法的反制移除
            opponent_player = state.current_player.opponent()
            
            # 为了简化，我们选择第一个合法的反制移除
            current_player_pieces = []
            current_player_normal_pieces = []
            current_player_value = state.current_player.value
            
            for r in range(GameState.BOARD_SIZE):
                for c in range(GameState.BOARD_SIZE):
                    if state_after_remove.board[r][c] == current_player_value:
                        current_player_pieces.append((r, c))
                        if not is_piece_in_shape(state_after_remove.board, r, c, current_player_value, set()):
                            current_player_normal_pieces.append((r, c))
            
            # 选择要反制移除的棋子
            counter_remove_position = None
            if current_player_normal_pieces:
                counter_remove_position = current_player_normal_pieces[0]
            elif current_player_pieces:
                counter_remove_position = current_player_pieces[0]
            
            if counter_remove_position:
                return apply_counter_removal_phase3(state_after_remove, counter_remove_position, quiet=quiet)
            else:
                # 没有棋子可以反制移除，对方胜利
                return state_after_remove
        else:
            raise ValueError(f"在第三阶段不支持操作 {move['action_type']}")
    
    else:
        raise ValueError(f"不支持的游戏阶段: {state.phase}")
