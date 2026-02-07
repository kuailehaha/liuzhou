from typing import Any, Dict, List, Tuple

from src.game_state import GameState, Phase, Player
from src.rule_engine import (
    apply_capture_selection,
    apply_counter_removal_phase3,
    apply_forced_removal,
    apply_mark_selection,
    apply_movement_move,
    apply_placement_move,
    generate_capture_targets,
    generate_mark_targets,
    generate_movement_moves,
    generate_placement_positions,
    handle_no_moves_phase3,
    has_legal_movement_moves,
    is_piece_in_shape,
    process_phase2_removals,
)

MoveType = Dict[str, Any]


def generate_all_legal_moves(state: GameState) -> List[MoveType]:
    if state.is_game_over():
        return []

    phase = state.phase

    if phase == Phase.PLACEMENT:
        return [
            {"phase": Phase.PLACEMENT, "action_type": "place", "position": pos}
            for pos in generate_placement_positions(state)
        ]

    if phase == Phase.MARK_SELECTION:
        return [
            {"phase": Phase.MARK_SELECTION, "action_type": "mark", "position": pos}
            for pos in generate_mark_targets(state)
        ]

    if phase == Phase.REMOVAL:
        return [{"phase": Phase.REMOVAL, "action_type": "process_removal"}]

    if phase == Phase.FORCED_REMOVAL:
        return _generate_moves_forced_removal(state)

    if phase == Phase.MOVEMENT:
        if has_legal_movement_moves(state):
            return [
                {
                    "phase": Phase.MOVEMENT,
                    "action_type": "move",
                    "from_position": src,
                    "to_position": dst,
                }
                for src, dst in generate_movement_moves(state)
            ]
        return _generate_moves_no_moves(state)

    if phase == Phase.CAPTURE_SELECTION:
        return [
            {"phase": Phase.CAPTURE_SELECTION, "action_type": "capture", "position": pos}
            for pos in generate_capture_targets(state)
        ]

    if phase == Phase.COUNTER_REMOVAL:
        return _generate_moves_counter_removal(state)

    return []


def apply_move(state: GameState, move: MoveType, quiet: bool = False) -> GameState:
    if move["phase"] != state.phase:
        raise ValueError(f"走法阶段 {move['phase']} 与当前状态 {state.phase} 不一致")

    if state.phase == Phase.PLACEMENT:
        if move["action_type"] != "place":
            raise ValueError("落子阶段仅允许 'place' 动作")
        new_state = apply_placement_move(state, move["position"])

    elif state.phase == Phase.MARK_SELECTION:
        if move["action_type"] != "mark":
            raise ValueError("标记阶段仅允许 'mark' 动作")
        new_state = apply_mark_selection(state, move["position"])

    elif state.phase == Phase.REMOVAL:
        if move["action_type"] != "process_removal":
            raise ValueError("移除阶段仅允许 'process_removal' 动作")
        new_state = process_phase2_removals(state)

    elif state.phase == Phase.FORCED_REMOVAL:
        if move["action_type"] != "remove":
            raise ValueError("强制移除阶段仅允许 'remove' 动作")
        new_state = apply_forced_removal(state, move["position"])

    elif state.phase == Phase.MOVEMENT:
        if move["action_type"] == "move":
            new_state = apply_movement_move(
                state,
                (move["from_position"], move["to_position"]),
                quiet=quiet,
            )
        elif move["action_type"] == "no_moves_remove":
            new_state = handle_no_moves_phase3(state, move["position"], quiet=quiet)
        else:
            raise ValueError(f"未知的走子阶段动作: {move['action_type']}")

    elif state.phase == Phase.CAPTURE_SELECTION:
        if move["action_type"] != "capture":
            raise ValueError("提子阶段仅允许 'capture' 动作")
        new_state = apply_capture_selection(state, move["position"], quiet=quiet)

    elif state.phase == Phase.COUNTER_REMOVAL:
        if move["action_type"] != "counter_remove":
            raise ValueError("反制移除阶段仅允许 'counter_remove' 动作")
        new_state = apply_counter_removal_phase3(state, move["position"], quiet=quiet)

    else:
        raise ValueError(f"不支持的阶段: {state.phase}")

    new_state.move_count = state.move_count + 1

    # Track "no capture" for draw detection (类似中国象棋的无吃子判和规则)
    if state.phase in (Phase.PLACEMENT, Phase.MARK_SELECTION):
        # 落子阶段不跟踪
        new_state.moves_since_capture = 0
    else:
        # 比较走子前后棋子总数，若有棋子被移除则重置计数
        old_total = (state.count_player_pieces(Player.BLACK)
                     + state.count_player_pieces(Player.WHITE))
        new_total = (new_state.count_player_pieces(Player.BLACK)
                     + new_state.count_player_pieces(Player.WHITE))
        if new_total < old_total:
            new_state.moves_since_capture = 0
        else:
            new_state.moves_since_capture = state.moves_since_capture + 1

    return new_state


# ---------------------------------------------------------------------------
# Internal helpers for special phases
# ---------------------------------------------------------------------------


def _generate_moves_forced_removal(state: GameState) -> List[MoveType]:
    legal_moves: List[MoveType] = []
    if state.phase != Phase.FORCED_REMOVAL:
        return legal_moves

    if state.forced_removals_done == 0:
        target_player = Player.BLACK
    elif state.forced_removals_done == 1:
        target_player = Player.WHITE
    else:
        return legal_moves

    target_value = target_player.value
    candidates: List[Tuple[int, int]] = []
    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == target_value and not is_piece_in_shape(
                state.board, r, c, target_value, set()
            ):
                candidates.append((r, c))

    for pos in candidates:
        legal_moves.append(
            {"phase": Phase.FORCED_REMOVAL, "action_type": "remove", "position": pos}
        )
    return legal_moves


def _generate_moves_no_moves(state: GameState) -> List[MoveType]:
    legal_moves: List[MoveType] = []
    current_player = state.current_player
    opponent = current_player.opponent()
    opponent_value = opponent.value

    opponent_pieces: List[Tuple[int, int]] = []
    opponent_normal: List[Tuple[int, int]] = []
    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == opponent_value:
                pos = (r, c)
                opponent_pieces.append(pos)
                if not is_piece_in_shape(state.board, r, c, opponent_value, set()):
                    opponent_normal.append(pos)

    targets = opponent_normal if opponent_normal else opponent_pieces
    for pos in targets:
        legal_moves.append(
            {"phase": Phase.MOVEMENT, "action_type": "no_moves_remove", "position": pos}
        )
    return legal_moves


def _generate_moves_counter_removal(state: GameState) -> List[MoveType]:
    legal_moves: List[MoveType] = []
    remover = state.current_player
    stuck_player = remover.opponent()
    stuck_value = stuck_player.value

    stuck_pieces: List[Tuple[int, int]] = []
    stuck_normal: List[Tuple[int, int]] = []
    for r in range(GameState.BOARD_SIZE):
        for c in range(GameState.BOARD_SIZE):
            if state.board[r][c] == stuck_value:
                pos = (r, c)
                stuck_pieces.append(pos)
                if not is_piece_in_shape(state.board, r, c, stuck_value, set()):
                    stuck_normal.append(pos)

    targets = stuck_normal if stuck_normal else stuck_pieces
    for pos in targets:
        legal_moves.append(
            {"phase": Phase.COUNTER_REMOVAL, "action_type": "counter_remove", "position": pos}
        )
    return legal_moves
