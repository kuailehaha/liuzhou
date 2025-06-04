# file: src/game_state.py

from enum import Enum
from typing import List, Optional, Tuple

class Phase(Enum):
    PLACEMENT = 1   # 第一阶段：落子
    REMOVAL = 2     # 第二阶段：标记/强制移除
    MOVEMENT = 3    # 第三阶段：走子与提吃
    FORCED_REMOVAL = 4 # 新增：强制移除阶段

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
        marked_white: Optional[set] = None,   # 存储被标记的白棋子坐标
        forced_removals_done: int = 0 # 新增：记录强制移除阶段完成的次数
    ):
        if board is None:
            board = [[0]*self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]

        self.board = board
        self.phase = phase
        self.current_player = current_player

        self.marked_black = marked_black if marked_black is not None else set()
        self.marked_white = marked_white if marked_white is not None else set()
        self.forced_removals_done = forced_removals_done # 初始化新属性

    def copy(self) -> "GameState":
        new_board = [row[:] for row in self.board]
        new_state = GameState(
            board=new_board,
            phase=self.phase,
            current_player=self.current_player,
            marked_black=self.marked_black.copy(),
            marked_white=self.marked_white.copy(),
            forced_removals_done=self.forced_removals_done # 复制新属性
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

    def get_player_pieces(self, player: Player) -> List[Tuple[int, int]]:
        """获取指定玩家在棋盘上所有棋子的坐标列表"""
        pieces = []
        player_value = player.value
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == player_value:
                    pieces.append((r, c))
        return pieces

    def count_player_pieces(self, player: Player) -> int:
        """获取指定玩家在棋盘上的棋子总数"""
        count = 0
        player_value = player.value
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == player_value:
                    count += 1
        return count

    def __str__(self):
        # 构建棋盘字符串，带坐标和边框，被标记的棋子特殊显示
        header = "    " + " ".join(str(c) for c in range(self.BOARD_SIZE))
        border = "   +" + "-" * (2 * self.BOARD_SIZE - 1) + "+"
        rows = [header, border]
        for r in range(self.BOARD_SIZE):
            row_str = f" {r} |"
            for c in range(self.BOARD_SIZE):
                if (r, c) in self.marked_black:
                    cell = "◎"  # 被标记黑棋
                elif (r, c) in self.marked_white:
                    cell = "◉"  # 被标记白棋
                else:
                    val = self.board[r][c]
                    if val == 1:
                        cell = "○"  # 黑棋
                    elif val == -1:
                        cell = "●"  # 白棋
                    else:
                        cell = "·"  # 空位
                row_str += cell + " "
            row_str = row_str.rstrip() + "|"
            rows.append(row_str)
        rows.append(border)
        return (
            "\n".join(rows)
            + f"\nPhase: {self.phase}, Current Player: {self.current_player}\n"
            + f"Marked Black: {self.marked_black}\nMarked White: {self.marked_white}\n"
            + f"Forced Removals Done: {self.forced_removals_done}"  # 在打印信息中加入
        )
