#include "v0/game_state.hpp"

#include <algorithm>

namespace v0 {

std::vector<std::pair<int, int>> MarkSet::ToVector() const {
    std::vector<std::pair<int, int>> coords;
    coords.reserve(bits.count());
    for (int idx = 0; idx < kCellCount; ++idx) {
        if (bits.test(idx)) {
            int row = idx / kBoardSize;
            int col = idx % kBoardSize;
            coords.emplace_back(row, col);
        }
    }
    return coords;
}

GameState::GameState() {
    board.fill(0);
}

bool GameState::IsBoardFull() const {
    return std::all_of(board.begin(), board.end(), [](int8_t cell) { return cell != 0; });
}

int GameState::CountPlayerPieces(Player player) const {
    int target = PlayerValue(player);
    int count = 0;
    for (int8_t cell : board) {
        if (cell == target) {
            ++count;
        }
    }
    return count;
}

std::vector<std::pair<int, int>> GameState::GetPlayerPieces(Player player) const {
    std::vector<std::pair<int, int>> pieces;
    pieces.reserve(kCellCount);
    int target = PlayerValue(player);
    for (int idx = 0; idx < kCellCount; ++idx) {
        if (board[idx] == target) {
            pieces.emplace_back(idx / kBoardSize, idx % kBoardSize);
        }
    }
    return pieces;
}

const MarkSet& GameState::Marks(Player player) const {
    return player == Player::kBlack ? marked_black : marked_white;
}

MarkSet& GameState::Marks(Player player) {
    return player == Player::kBlack ? marked_black : marked_white;
}

}  // namespace v0
