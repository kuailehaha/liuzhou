# file: src/game_state.py

from enum import Enum
from typing import List, Optional, Tuple

class Phase(Enum):
    PLACEMENT = 1   # 第一阶段：落子
    REMOVAL = 2     # 第二阶段：标记/强制移除
    MOVEMENT = 3    # 第三阶段：走子与提吃

class Player(Enum):
    BLACK = 1
    WHITE = -1

    def opponent(self) -> "Player":
        return Player.WHITE if self == Player.BLACK else Player.BLACK

class GameState:
    BOARD_SIZE = 6

    def __init__(
        self,
        board: Optional[List[List[int]]] = None,
        phase: Phase = Phase.PLACEMENT,
        current_player: Player = Player.BLACK,
        marked_black: Optional[set] = None,  # 存储被标记的黑棋子坐标
        marked_white: Optional[set] = None   # 存储被标记的白棋子坐标
    ):
        if board is None:
            board = [[0]*self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]

        self.board = board
        self.phase = phase
        self.current_player = current_player

        self.marked_black = marked_black if marked_black is not None else set()
        self.marked_white = marked_white if marked_white is not None else set()

    def copy(self) -> "GameState":
        new_board = [row[:] for row in self.board]
        new_state = GameState(
            board=new_board,
            phase=self.phase,
            current_player=self.current_player,
            marked_black=self.marked_black.copy(),
            marked_white=self.marked_white.copy()
        )
        return new_state

    def switch_player(self):
        self.current_player = self.current_player.opponent()

    def is_board_full(self) -> bool:
        for row in self.board:
            for cell in row:
                if cell == 0:
                    return False
        return True

    def __str__(self):
        rows = []
        for r in range(self.BOARD_SIZE):
            row_str = " ".join(f"{self.board[r][c]:2d}" for c in range(self.BOARD_SIZE))
            rows.append(row_str)
        return (
            "\n".join(rows)
            + f"\nPhase: {self.phase}, Current Player: {self.current_player}\n"
            + f"Marked Black: {self.marked_black}\nMarked White: {self.marked_white}"
        )
