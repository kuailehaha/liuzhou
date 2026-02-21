#include "v0/rule_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace v0 {
namespace {

constexpr std::array<std::pair<int, int>, 4> kDirections = {{
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1},
}};

enum class ShapeType {
    kNone,
    kLine,
    kSquare,
};

bool InBounds(int r, int c) {
    return r >= 0 && r < kBoardSize && c >= 0 && c < kBoardSize;
}

const MarkSet& MarksForPlayer(const GameState& state, Player player) {
    return state.Marks(player);
}

MarkSet& MarksForPlayer(GameState& state, Player player) {
    return state.Marks(player);
}

void SetPendingMarks(GameState& state, int required) {
    state.pending_marks_required = required;
    state.pending_marks_remaining = required;
}

void ClearPendingMarks(GameState& state) {
    state.pending_marks_required = 0;
    state.pending_marks_remaining = 0;
}

void SetPendingCaptures(GameState& state, int required) {
    state.pending_captures_required = required;
    state.pending_captures_remaining = required;
}

void ClearPendingCaptures(GameState& state) {
    state.pending_captures_required = 0;
    state.pending_captures_remaining = 0;
}

bool CheckSquares(
    const GameState& state,
    int r,
    int c,
    int player_value,
    const MarkSet& marked_set) {
    for (int dr : {0, -1}) {
        for (int dc : {0, -1}) {
            int rr = r + dr;
            int cc = c + dc;
            if (rr >= 0 && rr < kBoardSize - 1 && cc >= 0 && cc < kBoardSize - 1) {
                bool ok = true;
                std::array<std::pair<int, int>, 4> cells = {{
                    {rr, cc},
                    {rr, cc + 1},
                    {rr + 1, cc},
                    {rr + 1, cc + 1},
                }};
                for (const auto& cell : cells) {
                    if (state.BoardAt(cell.first, cell.second) != player_value ||
                        marked_set.Contains(cell)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool CheckLines(
    const GameState& state,
    int r,
    int c,
    int player_value,
    const MarkSet& marked_set) {
    int count = 1;
    // Horizontal left
    for (int dc = c - 1; dc >= 0; --dc) {
        if (state.BoardAt(r, dc) == player_value && !marked_set.Contains(r, dc)) {
            ++count;
        } else {
            break;
        }
    }
    // Horizontal right
    for (int dc = c + 1; dc < kBoardSize; ++dc) {
        if (state.BoardAt(r, dc) == player_value && !marked_set.Contains(r, dc)) {
            ++count;
        } else {
            break;
        }
    }
    if (count >= 6) {
        return true;
    }

    count = 1;
    // Vertical up
    for (int dr = r - 1; dr >= 0; --dr) {
        if (state.BoardAt(dr, c) == player_value && !marked_set.Contains(dr, c)) {
            ++count;
        } else {
            break;
        }
    }
    // Vertical down
    for (int dr = r + 1; dr < kBoardSize; ++dr) {
        if (state.BoardAt(dr, c) == player_value && !marked_set.Contains(dr, c)) {
            ++count;
        } else {
            break;
        }
    }
    return count >= 6;
}

ShapeType DetectShapeFormed(
    const GameState& state,
    int r,
    int c,
    int player_value,
    const MarkSet& marked_set) {
    bool found_square = CheckSquares(state, r, c, player_value, marked_set);
    bool found_line = CheckLines(state, r, c, player_value, marked_set);

    if (found_line) {
        return ShapeType::kLine;
    }
    if (found_square) {
        return ShapeType::kSquare;
    }
    return ShapeType::kNone;
}

std::vector<Coord> CollectOpponentPieces(
    const GameState& state,
    Player opponent,
    const MarkSet& opponent_marked,
    std::vector<Coord>* normal_out = nullptr) {
    std::vector<Coord> pieces;
    pieces.reserve(kCellCount);
    int target = PlayerValue(opponent);
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) == target) {
                Coord pos{r, c};
                pieces.push_back(pos);
                if (normal_out &&
                    !IsPieceInShape(state, r, c, target, opponent_marked)) {
                    normal_out->push_back(pos);
                }
            }
        }
    }
    return pieces;
}

std::vector<Coord> FilterUnmarked(
    const std::vector<Coord>& coords,
    const MarkSet& marked) {
    std::vector<Coord> result;
    result.reserve(coords.size());
    for (const auto& coord : coords) {
        if (!marked.Contains(coord)) {
            result.push_back(coord);
        }
    }
    return result;
}

}  // namespace

bool IsPieceInShape(
    const GameState& state,
    int r,
    int c,
    int player_value,
    const MarkSet& marked_set) {
    if (!InBounds(r, c)) {
        return false;
    }
    if (state.BoardAt(r, c) != player_value) {
        return false;
    }
    return CheckSquares(state, r, c, player_value, marked_set) ||
        CheckLines(state, r, c, player_value, marked_set);
}

std::vector<Coord> GeneratePlacementPositions(const GameState& state) {
    if (state.phase != Phase::kPlacement) {
        return {};
    }
    std::vector<Coord> positions;
    positions.reserve(kCellCount);
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) == 0) {
                positions.emplace_back(r, c);
            }
        }
    }
    return positions;
}

GameState ApplyPlacementMove(const GameState& state, const Coord& position) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kPlacement) {
        throw std::runtime_error("当前不是落子阶段");
    }

    const int r = position.first;
    const int c = position.second;
    if (!InBounds(r, c)) {
        throw std::runtime_error("落子位置超出棋盘范围");
    }
    if (new_state.BoardAt(r, c) != 0) {
        throw std::runtime_error("该位置已有棋子");
    }

    Player current = new_state.current_player;
    Player opponent = Opponent(current);
    const auto& opponent_marked = MarksForPlayer(new_state, opponent);
    if (opponent_marked.Contains(position)) {
        throw std::runtime_error("该位置已被对方标记");
    }

    new_state.SetBoard(r, c, static_cast<int8_t>(PlayerValue(current)));

    auto& own_marked = MarksForPlayer(new_state, current);
    if (!own_marked.Contains(position)) {
        ShapeType shape = DetectShapeFormed(
            new_state,
            r,
            c,
            PlayerValue(current),
            own_marked);
        if (shape == ShapeType::kLine) {
            SetPendingMarks(new_state, 2);
            new_state.phase = Phase::kMarkSelection;
            return new_state;
        }
        if (shape == ShapeType::kSquare) {
            SetPendingMarks(new_state, 1);
            new_state.phase = Phase::kMarkSelection;
            return new_state;
        }
    }

    ClearPendingMarks(new_state);

    if (new_state.IsBoardFull()) {
        new_state.phase = Phase::kRemoval;
    } else {
        new_state.SwitchPlayer();
        new_state.phase = Phase::kPlacement;
    }
    return new_state;
}

std::vector<Coord> GenerateMarkTargets(const GameState& state) {
    if (state.phase != Phase::kMarkSelection || state.pending_marks_remaining <= 0) {
        return {};
    }

    Player current = state.current_player;
    Player opponent = Opponent(current);
    const auto& opponent_marked = MarksForPlayer(state, opponent);

    std::vector<Coord> opponent_normal;
    std::vector<Coord> opponent_pieces = CollectOpponentPieces(
        state,
        opponent,
        opponent_marked,
        &opponent_normal);

    std::vector<Coord> pool;
    if (!opponent_normal.empty()) {
        pool = FilterUnmarked(opponent_normal, opponent_marked);
    } else {
        pool = FilterUnmarked(opponent_pieces, opponent_marked);
    }

    if (pool.empty()) {
        pool = FilterUnmarked(opponent_pieces, opponent_marked);
    }
    return pool;
}

GameState ApplyMarkSelection(const GameState& state, const Coord& position) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kMarkSelection) {
        throw std::runtime_error("当前不是标记选择阶段");
    }
    if (new_state.pending_marks_remaining <= 0) {
        throw std::runtime_error("没有待完成的标记任务");
    }

    const int r = position.first;
    const int c = position.second;
    if (!InBounds(r, c)) {
        throw std::runtime_error("标记位置超出棋盘范围");
    }

    Player opponent = Opponent(new_state.current_player);
    int opponent_value = PlayerValue(opponent);
    const auto& opponent_marked = MarksForPlayer(new_state, opponent);

    if (new_state.BoardAt(r, c) != opponent_value) {
        throw std::runtime_error("只能标记对方棋子");
    }
    if (opponent_marked.Contains(position)) {
        throw std::runtime_error("该棋子已经被标记");
    }

    std::vector<Coord> opponent_normal;
    CollectOpponentPieces(new_state, opponent, opponent_marked, &opponent_normal);
    std::vector<Coord> opponent_normal_unmarked = FilterUnmarked(opponent_normal, opponent_marked);
    if (!opponent_normal_unmarked.empty() &&
        IsPieceInShape(new_state, r, c, opponent_value, opponent_marked)) {
        throw std::runtime_error("对方存在普通棋子时，不能标记其关键结构中的棋子");
    }

    MarksForPlayer(new_state, opponent).Add(position);
    new_state.pending_marks_remaining -= 1;
    if (new_state.pending_marks_remaining > 0) {
        return new_state;
    }

    ClearPendingMarks(new_state);
    if (new_state.IsBoardFull()) {
        new_state.phase = Phase::kRemoval;
    } else {
        new_state.SwitchPlayer();
        new_state.phase = Phase::kPlacement;
    }
    return new_state;
}

GameState ProcessPhase2Removals(const GameState& state) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kRemoval) {
        throw std::runtime_error("当前不是移除阶段");
    }

    if (new_state.marked_black.Empty() && new_state.marked_white.Empty()) {
        new_state.phase = Phase::kForcedRemoval;
        new_state.current_player = Player::kWhite;
        new_state.forced_removals_done = 0;
        return new_state;
    }

    int removed = 0;
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            Coord pos{r, c};
            if (new_state.marked_black.Contains(pos) || new_state.marked_white.Contains(pos)) {
                if (new_state.BoardAt(r, c) != 0) {
                    new_state.SetBoard(r, c, 0);
                    ++removed;
                }
            }
        }
    }

    new_state.marked_black.Clear();
    new_state.marked_white.Clear();

    if (removed > 0) {
        new_state.phase = Phase::kMovement;
        new_state.current_player = Player::kWhite;
    }

    return new_state;
}

std::vector<Move> GenerateMovementMoves(const GameState& state) {
    if (state.phase != Phase::kMovement) {
        return {};
    }
    std::vector<Move> moves;
    moves.reserve(kCellCount * 4);
    int player_value = PlayerValue(state.current_player);
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) != player_value) {
                continue;
            }
            for (const auto& dir : kDirections) {
                int nr = r + dir.first;
                int nc = c + dir.second;
                if (InBounds(nr, nc) && state.BoardAt(nr, nc) == 0) {
                    moves.push_back({{r, c}, {nr, nc}});
                }
            }
        }
    }
    return moves;
}

bool HasLegalMovementMoves(const GameState& state) {
    if (state.phase != Phase::kMovement) {
        throw std::runtime_error("当前不是走子阶段");
    }
    auto moves = GenerateMovementMoves(state);
    return !moves.empty();
}

GameState ApplyMovementMove(const GameState& state, const Move& move, bool /*quiet*/) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kMovement) {
        throw std::runtime_error("当前不是走子阶段");
    }
    const Coord& from = move.first;
    const Coord& to = move.second;

    if (!InBounds(from.first, from.second) || !InBounds(to.first, to.second)) {
        throw std::runtime_error("移动超出棋盘范围");
    }
    if (new_state.BoardAt(from.first, from.second) != PlayerValue(new_state.current_player)) {
        throw std::runtime_error("起始位置不是当前玩家棋子");
    }
    if (new_state.BoardAt(to.first, to.second) != 0) {
        throw std::runtime_error("目标位置不是空位");
    }

    bool valid_step =
        (std::abs(from.first - to.first) == 1 && from.second == to.second) ||
        (std::abs(from.second - to.second) == 1 && from.first == to.first);
    if (!valid_step) {
        throw std::runtime_error("只能水平或垂直移动一格");
    }

    new_state.SetBoard(to.first, to.second, new_state.BoardAt(from.first, from.second));
    new_state.SetBoard(from.first, from.second, 0);

    MarkSet empty_marks;
    ShapeType shape = DetectShapeFormed(
        new_state,
        to.first,
        to.second,
        PlayerValue(new_state.current_player),
        empty_marks);

    if (shape == ShapeType::kLine) {
        SetPendingCaptures(new_state, 2);
        new_state.phase = Phase::kCaptureSelection;
        return new_state;
    }
    if (shape == ShapeType::kSquare) {
        SetPendingCaptures(new_state, 1);
        new_state.phase = Phase::kCaptureSelection;
        return new_state;
    }

    ClearPendingCaptures(new_state);
    new_state.SwitchPlayer();
    return new_state;
}

std::vector<Coord> GenerateCaptureTargets(const GameState& state) {
    if (state.phase != Phase::kCaptureSelection || state.pending_captures_remaining <= 0) {
        return {};
    }

    Player opponent = Opponent(state.current_player);
    const auto& opponent_marked = MarksForPlayer(state, opponent);
    std::vector<Coord> opponent_normal;
    std::vector<Coord> opponent_pieces = CollectOpponentPieces(
        state,
        opponent,
        opponent_marked,
        &opponent_normal);

    if (!opponent_normal.empty()) {
        return opponent_normal;
    }
    return opponent_pieces;
}

GameState ApplyCaptureSelection(const GameState& state, const Coord& position, bool quiet) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kCaptureSelection) {
        throw std::runtime_error("当前不是提子阶段");
    }
    if (new_state.pending_captures_remaining <= 0) {
        throw std::runtime_error("没有待完成的提子任务");
    }

    if (!InBounds(position.first, position.second)) {
        throw std::runtime_error("位置超出棋盘范围");
    }

    Player opponent = Opponent(new_state.current_player);
    int opponent_value = PlayerValue(opponent);
    const auto& opponent_marked = MarksForPlayer(new_state, opponent);

    if (new_state.BoardAt(position.first, position.second) != opponent_value) {
        throw std::runtime_error("只能提掉对方棋子");
    }

    std::vector<Coord> opponent_normal;
    CollectOpponentPieces(new_state, opponent, opponent_marked, &opponent_normal);
    if (!opponent_normal.empty() &&
        IsPieceInShape(new_state, position.first, position.second, opponent_value, opponent_marked)) {
        throw std::runtime_error("对方还有普通棋子时，不能提关键结构中的棋子");
    }

    new_state.SetBoard(position.first, position.second, 0);
    new_state.pending_captures_remaining -= 1;

    if (new_state.CountPlayerPieces(opponent) < kLosePieceThreshold) {
        if (!quiet) {
            // Placeholder for optional logging hook.
        }
        return new_state;
    }

    if (new_state.pending_captures_remaining > 0) {
        return new_state;
    }

    ClearPendingCaptures(new_state);
    new_state.SwitchPlayer();
    new_state.phase = Phase::kMovement;
    return new_state;
}

GameState ApplyForcedRemoval(const GameState& state, const Coord& piece_to_remove) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kForcedRemoval) {
        throw std::runtime_error("当前不是强制移除阶段");
    }

    if (!InBounds(piece_to_remove.first, piece_to_remove.second)) {
        throw std::runtime_error("位置超出棋盘范围");
    }

    int r = piece_to_remove.first;
    int c = piece_to_remove.second;
    if (new_state.forced_removals_done == 0) {
        if (new_state.current_player != Player::kWhite) {
            throw std::runtime_error("强制移除顺序错误：应由白方先手");
        }
        if (new_state.BoardAt(r, c) != PlayerValue(Player::kBlack)) {
            throw std::runtime_error("必须移除黑方棋子");
        }
        MarkSet empty_marked;
        if (IsPieceInShape(new_state, r, c, PlayerValue(Player::kBlack), empty_marked)) {
            throw std::runtime_error("构成方或洲的棋子不能被强制移除");
        }
        new_state.SetBoard(r, c, 0);
        new_state.forced_removals_done = 1;
        new_state.current_player = Player::kBlack;
    } else if (new_state.forced_removals_done == 1) {
        if (new_state.current_player != Player::kBlack) {
            throw std::runtime_error("强制移除顺序错误：应由黑方执行");
        }
        if (new_state.BoardAt(r, c) != PlayerValue(Player::kWhite)) {
            throw std::runtime_error("必须移除白方棋子");
        }
        MarkSet empty_marked;
        if (IsPieceInShape(new_state, r, c, PlayerValue(Player::kWhite), empty_marked)) {
            throw std::runtime_error("构成方或洲的棋子不能被强制移除");
        }
        new_state.SetBoard(r, c, 0);
        new_state.forced_removals_done = 2;
        new_state.phase = Phase::kMovement;
        new_state.current_player = Player::kWhite;
    } else {
        throw std::runtime_error("强制移除状态异常");
    }

    return new_state;
}

GameState HandleNoMovesPhase3(const GameState& state, const Coord& stucked_player_removes, bool quiet) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kMovement) {
        throw std::runtime_error("无子可动处理只能在走子阶段触发");
    }

    Player current = new_state.current_player;
    Player opponent = Opponent(current);
    if (!InBounds(stucked_player_removes.first, stucked_player_removes.second)) {
        throw std::runtime_error("位置超出棋盘范围");
    }
    if (new_state.BoardAt(stucked_player_removes.first, stucked_player_removes.second) !=
        PlayerValue(opponent)) {
        throw std::runtime_error("只能移除对方棋子");
    }

    MarkSet empty_marked;
    std::vector<Coord> opponent_normal;
    CollectOpponentPieces(new_state, opponent, empty_marked, &opponent_normal);
    if (!opponent_normal.empty() &&
        IsPieceInShape(
            new_state,
            stucked_player_removes.first,
            stucked_player_removes.second,
            PlayerValue(opponent),
            empty_marked)) {
        throw std::runtime_error("对方尚有普通棋子，不能移除结构中的棋子");
    }

    new_state.SetBoard(stucked_player_removes.first, stucked_player_removes.second, 0);
    if (new_state.CountPlayerPieces(opponent) < kLosePieceThreshold) {
        if (!quiet) {
            // Optional logging hook.
        }
        return new_state;
    }

    new_state.phase = Phase::kCounterRemoval;
    new_state.SwitchPlayer();
    return new_state;
}

GameState ApplyCounterRemovalPhase3(const GameState& state, const Coord& opponent_removes, bool quiet) {
    GameState new_state = state.Copy();
    if (new_state.phase != Phase::kCounterRemoval) {
        throw std::runtime_error("当前不是反制移除阶段");
    }

    Player remover = new_state.current_player;
    Player stuck_player = Opponent(remover);

    if (!InBounds(opponent_removes.first, opponent_removes.second)) {
        throw std::runtime_error("位置超出棋盘范围");
    }
    if (new_state.BoardAt(opponent_removes.first, opponent_removes.second) !=
        PlayerValue(stuck_player)) {
        throw std::runtime_error("只能移除被困住玩家的棋子");
    }

    MarkSet empty_marked;
    std::vector<Coord> stuck_normal;
    CollectOpponentPieces(new_state, stuck_player, empty_marked, &stuck_normal);
    if (!stuck_normal.empty() &&
        IsPieceInShape(
            new_state,
            opponent_removes.first,
            opponent_removes.second,
            PlayerValue(stuck_player),
            empty_marked)) {
        throw std::runtime_error("对方尚有普通棋子，不能移除结构中的棋子");
    }

    new_state.SetBoard(opponent_removes.first, opponent_removes.second, 0);
    if (new_state.CountPlayerPieces(stuck_player) < kLosePieceThreshold) {
        if (!quiet) {
            // Optional logging hook.
        }
        return new_state;
    }

    new_state.phase = Phase::kMovement;
    new_state.SwitchPlayer();
    return new_state;
}

std::vector<Coord> GenerateLegalMovesPhase1(const GameState& state) {
    return GeneratePlacementPositions(state);
}

GameState ApplyMovePhase1(
    const GameState& state,
    const Coord& move,
    const std::vector<Coord>& mark_positions) {
    GameState new_state = ApplyPlacementMove(state, move);
    if (!mark_positions.empty()) {
        if (new_state.phase != Phase::kMarkSelection) {
            throw std::runtime_error("当前状态不需要标记，但传入了 mark_positions");
        }
        for (const auto& pos : mark_positions) {
            new_state = ApplyMarkSelection(new_state, pos);
        }
    }
    return new_state;
}

std::vector<Move> GenerateLegalMovesPhase3(const GameState& state) {
    return GenerateMovementMoves(state);
}

bool HasLegalMovesPhase3(const GameState& state) {
    return HasLegalMovementMoves(state);
}

GameState ApplyMovePhase3(
    const GameState& state,
    const Move& move,
    const std::vector<Coord>& capture_positions,
    bool quiet) {
    GameState new_state = ApplyMovementMove(state, move, quiet);
    if (!capture_positions.empty()) {
        if (new_state.phase != Phase::kCaptureSelection) {
            throw std::runtime_error("当前状态不需要提子，但传入了 capture_positions");
        }
        for (const auto& pos : capture_positions) {
            new_state = ApplyCaptureSelection(new_state, pos, quiet);
        }
    }
    return new_state;
}

}  // namespace v0
