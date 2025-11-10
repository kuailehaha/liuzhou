#include "v0/game_state.hpp"

#include <algorithm>

namespace v0 {

std::vector<Coord> MarkSet::ToVector() const {
    std::vector<Coord> coords;
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

std::vector<Coord> GameState::GetPlayerPieces(Player player) const {
    std::vector<Coord> pieces;
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

std::optional<Player> GameState::GetWinner() const {
    if (phase == Phase::kPlacement) {
        return std::nullopt;
    }
    int black_pieces = CountPlayerPieces(Player::kBlack);
    int white_pieces = CountPlayerPieces(Player::kWhite);
    if (black_pieces == 0) {
        return Player::kWhite;
    }
    if (white_pieces == 0) {
        return Player::kBlack;
    }
    return std::nullopt;
}

bool GameState::IsGameOver() const {
    return GetWinner().has_value() || HasReachedMoveLimit();
}

}  // namespace v0
