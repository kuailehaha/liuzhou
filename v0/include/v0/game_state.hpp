#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace v0 {

constexpr int kBoardSize = 6;
constexpr int kCellCount = kBoardSize * kBoardSize;
constexpr int kMaxMoveCount = 200;

inline int CellIndex(int row, int col) {
    return row * kBoardSize + col;
}

using Coord = std::pair<int, int>;

enum class Phase : int32_t {
    kPlacement = 1,
    kMarkSelection = 2,
    kRemoval = 3,
    kMovement = 4,
    kCaptureSelection = 5,
    kForcedRemoval = 6,
    kCounterRemoval = 7,
};

enum class Player : int32_t {
    kBlack = 1,
    kWhite = -1,
};

inline Player Opponent(Player p) {
    return p == Player::kBlack ? Player::kWhite : Player::kBlack;
}

inline int PlayerValue(Player p) {
    return static_cast<int>(p == Player::kBlack ? 1 : -1);
}

struct MarkSet {
    std::bitset<kCellCount> bits;

    bool ContainsIndex(int idx) const {
        return bits.test(idx);
    }

    bool Contains(int row, int col) const {
        return ContainsIndex(CellIndex(row, col));
    }

    bool Contains(const std::pair<int, int>& coord) const {
        return Contains(coord.first, coord.second);
    }

    void Add(int row, int col) {
        bits.set(CellIndex(row, col));
    }

    void Add(const std::pair<int, int>& coord) {
        Add(coord.first, coord.second);
    }

    void AddIndex(int idx) {
        bits.set(idx);
    }

    void Remove(int row, int col) {
        bits.reset(CellIndex(row, col));
    }

    void Remove(const std::pair<int, int>& coord) {
        Remove(coord.first, coord.second);
    }

    void Clear() {
        bits.reset();
    }

    bool Empty() const {
        return bits.none();
    }

    std::vector<Coord> ToVector() const;
};

class GameState {
   public:
    using BoardArray = std::array<int8_t, kCellCount>;

    GameState();

    BoardArray board{};
    Phase phase{Phase::kPlacement};
    Player current_player{Player::kBlack};
    MarkSet marked_black{};
    MarkSet marked_white{};
    int32_t forced_removals_done{0};
    int32_t move_count{0};
    int32_t pending_marks_required{0};
    int32_t pending_marks_remaining{0};
    int32_t pending_captures_required{0};
    int32_t pending_captures_remaining{0};

    int8_t BoardAt(int row, int col) const {
        return board[CellIndex(row, col)];
    }

    void SetBoard(int row, int col, int8_t value) {
        board[CellIndex(row, col)] = value;
    }

    GameState Copy() const {
        return *this;
    }

    void SwitchPlayer() {
        current_player = Opponent(current_player);
    }

    bool IsBoardFull() const;
    bool HasReachedMoveLimit() const {
        return move_count >= kMaxMoveCount;
    }

    int CountPlayerPieces(Player player) const;
    std::vector<Coord> GetPlayerPieces(Player player) const;

    const MarkSet& Marks(Player player) const;
    MarkSet& Marks(Player player);

    void ClearMarks() {
        marked_black.Clear();
        marked_white.Clear();
    }

    std::optional<Player> GetWinner() const;
    bool IsGameOver() const;
};

}  // namespace v0
