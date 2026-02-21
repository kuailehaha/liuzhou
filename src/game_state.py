"""
Core game state representation and helpers.
"""

from enum import Enum
from typing import List, Optional, Set, Tuple


class Phase(Enum):
    PLACEMENT = 1  # 落子
    MARK_SELECTION = 2  # 逐个选择要标记的对方棋子
    REMOVAL = 3  # 已标记棋子的实际移除
    MOVEMENT = 4  # 行棋阶段，仅移动棋子
    CAPTURE_SELECTION = 5  # 逐个选择要提掉的对方棋子
    FORCED_REMOVAL = 6  # 强制移除阶段
    COUNTER_REMOVAL = 7  # 无子可动后的反制移除阶段


class Player(Enum):
    BLACK = 1
    WHITE = -1

    def opponent(self) -> "Player":
        return Player.WHITE if self == Player.BLACK else Player.BLACK


class GameState:
    BOARD_SIZE = 6
    MAX_MOVE_COUNT = 144
    LOSE_PIECE_THRESHOLD = 4
    NO_CAPTURE_DRAW_LIMIT = 36  # 18回合无吃子判和（每回合双方各一步 = 36 actions）

    def __init__(
        self,
        board: Optional[List[List[int]]] = None,
        phase: Phase = Phase.PLACEMENT,
        current_player: Player = Player.BLACK,
        marked_black: Optional[Set[Tuple[int, int]]] = None,
        marked_white: Optional[Set[Tuple[int, int]]] = None,
        forced_removals_done: int = 0,
        move_count: int = 0,
        pending_marks_required: int = 0,
        pending_marks_remaining: int = 0,
        pending_captures_required: int = 0,
        pending_captures_remaining: int = 0,
        moves_since_capture: int = 0,
    ):
        if board is None:
            board = [[0] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]

        self.board = board
        self.phase = phase
        self.current_player = current_player

        self.marked_black = marked_black if marked_black is not None else set()
        self.marked_white = marked_white if marked_white is not None else set()
        self.forced_removals_done = forced_removals_done
        self.move_count = move_count
        self.moves_since_capture = moves_since_capture

        # Pending tasks created when某一步形成方/洲或触发提子。
        self.pending_marks_required = pending_marks_required
        self.pending_marks_remaining = pending_marks_remaining
        self.pending_captures_required = pending_captures_required
        self.pending_captures_remaining = pending_captures_remaining

    def copy(self) -> "GameState":
        new_board = [row[:] for row in self.board]
        return GameState(
            board=new_board,
            phase=self.phase,
            current_player=self.current_player,
            marked_black=self.marked_black.copy(),
            marked_white=self.marked_white.copy(),
            forced_removals_done=self.forced_removals_done,
            move_count=self.move_count,
            pending_marks_required=self.pending_marks_required,
            pending_marks_remaining=self.pending_marks_remaining,
            pending_captures_required=self.pending_captures_required,
            pending_captures_remaining=self.pending_captures_remaining,
            moves_since_capture=self.moves_since_capture,
        )

    def switch_player(self):
        self.current_player = self.current_player.opponent()

    def has_reached_move_limit(self) -> bool:
        return (self.move_count >= self.MAX_MOVE_COUNT or
                self.moves_since_capture >= self.NO_CAPTURE_DRAW_LIMIT)

    def is_board_full(self) -> bool:
        return all(cell != 0 for row in self.board for cell in row)

    def get_player_pieces(self, player: Player) -> List[Tuple[int, int]]:
        """获取指定玩家在棋盘上所有棋子的坐标列表。"""
        player_value = player.value
        return [
            (r, c)
            for r in range(self.BOARD_SIZE)
            for c in range(self.BOARD_SIZE)
            if self.board[r][c] == player_value
        ]

    def count_player_pieces(self, player: Player) -> int:
        """统计指定玩家棋子数量。"""
        player_value = player.value
        return sum(
            1
            for r in range(self.BOARD_SIZE)
            for c in range(self.BOARD_SIZE)
            if self.board[r][c] == player_value
        )

    def clear_pending_marks(self):
        self.pending_marks_required = 0
        self.pending_marks_remaining = 0

    def clear_pending_captures(self):
        self.pending_captures_required = 0
        self.pending_captures_remaining = 0

    def __str__(self):
        header = "    " + " ".join(str(c) for c in range(self.BOARD_SIZE))
        border = "   +" + "-" * (2 * self.BOARD_SIZE - 1) + "+"
        rows = [header, border]
        for r in range(self.BOARD_SIZE):
            row_str = f" {r} |"
            for c in range(self.BOARD_SIZE):
                if (r, c) in self.marked_black:
                    cell = "B"  # 黑方被标记
                elif (r, c) in self.marked_white:
                    cell = "W"  # 白方被标记
                else:
                    val = self.board[r][c]
                    if val == Player.BLACK.value:
                        cell = "●"
                    elif val == Player.WHITE.value:
                        cell = "○"
                    else:
                        cell = "·"
                row_str += cell + " "
            rows.append(row_str.rstrip() + "|")
        rows.append(border)
        rows.append(f"Phase: {self.phase}, Current Player: {self.current_player}")
        rows.append(f"Marked Black: {self.marked_black}")
        rows.append(f"Marked White: {self.marked_white}")
        rows.append(f"Forced Removals Done: {self.forced_removals_done}")
        rows.append(
            f"Pending Marks: {self.pending_marks_remaining}/{self.pending_marks_required}"
        )
        rows.append(
            f"Pending Captures: {self.pending_captures_remaining}/{self.pending_captures_required}"
        )
        rows.append(f"Move Count: {self.move_count}")
        rows.append(f"Moves Since Capture: {self.moves_since_capture}")
        return "\n".join(rows)

    def get_winner(self) -> Optional[Player]:
        """若有一方棋子被提光，返回获胜方，否则返回 None。"""
        if self.phase == Phase.PLACEMENT:
            # 落子阶段不判胜负
            return None

        black_pieces = self.count_player_pieces(Player.BLACK)
        white_pieces = self.count_player_pieces(Player.WHITE)

        if black_pieces < self.LOSE_PIECE_THRESHOLD:
            return Player.WHITE
        if white_pieces < self.LOSE_PIECE_THRESHOLD:
            return Player.BLACK
        return None

    def is_game_over(self) -> bool:
        return self.get_winner() is not None or self.has_reached_move_limit()
