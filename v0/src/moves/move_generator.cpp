#include "v0/move_generator.hpp"

#include <stdexcept>

namespace v0 {
namespace {

constexpr int32_t kActionPlacement = 1;
constexpr int32_t kActionMovement = 2;
constexpr int32_t kActionMarkSelection = 3;
constexpr int32_t kActionCaptureSelection = 4;
constexpr int32_t kActionForcedRemovalSelection = 5;
constexpr int32_t kActionCounterRemovalSelection = 6;
constexpr int32_t kActionNoMovesRemovalSelection = 7;
constexpr int32_t kActionProcessRemoval = 8;

inline bool IsValidCoord(const Coord& coord) {
    return coord.first >= 0 && coord.first < kBoardSize && coord.second >= 0 &&
        coord.second < kBoardSize;
}

inline int CoordToCell(const Coord& coord) {
    return CellIndex(coord.first, coord.second);
}

MoveRecord MakeSimpleMove(Phase phase, ActionType type, const Coord& pos) {
    MoveRecord move;
    move.phase = phase;
    move.action_type = type;
    move.primary = pos;
    return move;
}

std::vector<Coord> CollectPieces(const GameState& state, Player player) {
    std::vector<Coord> pieces;
    pieces.reserve(kCellCount);
    int value = PlayerValue(player);
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) == value) {
                pieces.emplace_back(r, c);
            }
        }
    }
    return pieces;
}

}  // namespace

const char* ActionTypeToString(ActionType type) {
    switch (type) {
        case ActionType::kPlace:
            return "place";
        case ActionType::kMove:
            return "move";
        case ActionType::kMark:
            return "mark";
        case ActionType::kCapture:
            return "capture";
        case ActionType::kForcedRemoval:
            return "remove";
        case ActionType::kCounterRemoval:
            return "counter_remove";
        case ActionType::kNoMovesRemoval:
            return "no_moves_remove";
        case ActionType::kProcessRemoval:
            return "process_removal";
        default:
            return "unknown";
    }
}

MoveRecord MoveRecord::Placement(const Coord& pos) {
    return MakeSimpleMove(Phase::kPlacement, ActionType::kPlace, pos);
}

MoveRecord MoveRecord::Mark(const Coord& pos) {
    return MakeSimpleMove(Phase::kMarkSelection, ActionType::kMark, pos);
}

MoveRecord MoveRecord::Capture(const Coord& pos) {
    return MakeSimpleMove(Phase::kCaptureSelection, ActionType::kCapture, pos);
}

MoveRecord MoveRecord::ForcedRemoval(const Coord& pos) {
    return MakeSimpleMove(Phase::kForcedRemoval, ActionType::kForcedRemoval, pos);
}

MoveRecord MoveRecord::CounterRemoval(const Coord& pos) {
    return MakeSimpleMove(Phase::kCounterRemoval, ActionType::kCounterRemoval, pos);
}

MoveRecord MoveRecord::NoMovesRemoval(const Coord& pos) {
    return MakeSimpleMove(Phase::kMovement, ActionType::kNoMovesRemoval, pos);
}

MoveRecord MoveRecord::ProcessRemoval() {
    MoveRecord move;
    move.phase = Phase::kRemoval;
    move.action_type = ActionType::kProcessRemoval;
    move.primary = {-1, -1};
    move.secondary = {-1, -1};
    return move;
}

MoveRecord MoveRecord::Movement(const Coord& from, const Coord& to) {
    MoveRecord move;
    move.phase = Phase::kMovement;
    move.action_type = ActionType::kMove;
    move.primary = from;
    move.secondary = to;
    return move;
}

bool MoveRecord::HasPosition() const {
    switch (action_type) {
        case ActionType::kPlace:
        case ActionType::kMark:
        case ActionType::kCapture:
        case ActionType::kForcedRemoval:
        case ActionType::kCounterRemoval:
        case ActionType::kNoMovesRemoval:
            return IsValidCoord(primary);
        default:
            return false;
    }
}

Coord MoveRecord::Position() const {
    return primary;
}

bool MoveRecord::HasFrom() const {
    return action_type == ActionType::kMove && IsValidCoord(primary);
}

Coord MoveRecord::From() const {
    return primary;
}

bool MoveRecord::HasTo() const {
    return action_type == ActionType::kMove && IsValidCoord(secondary);
}

Coord MoveRecord::To() const {
    return secondary;
}

std::vector<MoveRecord> GenerateForcedRemovalMoves(const GameState& state) {
    std::vector<MoveRecord> moves;
    if (state.phase != Phase::kForcedRemoval) {
        return moves;
    }

    Player target_player;
    if (state.forced_removals_done == 0) {
        target_player = Player::kBlack;
    } else if (state.forced_removals_done == 1) {
        target_player = Player::kWhite;
    } else {
        return moves;
    }

    MarkSet empty_marked;
    int value = PlayerValue(target_player);
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) == value &&
                !IsPieceInShape(state, r, c, value, empty_marked)) {
                moves.push_back(MoveRecord::ForcedRemoval({r, c}));
            }
        }
    }
    return moves;
}

std::vector<MoveRecord> GenerateNoMovesOptions(const GameState& state) {
    std::vector<MoveRecord> moves;
    if (state.phase != Phase::kMovement) {
        return moves;
    }
    Player opponent = Opponent(state.current_player);
    int opponent_value = PlayerValue(opponent);
    MarkSet empty_marked;

    std::vector<Coord> opponent_pieces;
    std::vector<Coord> opponent_normal;
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) == opponent_value) {
                Coord pos{r, c};
                opponent_pieces.push_back(pos);
                if (!IsPieceInShape(state, r, c, opponent_value, empty_marked)) {
                    opponent_normal.push_back(pos);
                }
            }
        }
    }

    const std::vector<Coord>& targets =
        !opponent_normal.empty() ? opponent_normal : opponent_pieces;
    moves.reserve(targets.size());
    for (const auto& pos : targets) {
        moves.push_back(MoveRecord::NoMovesRemoval(pos));
    }
    return moves;
}

std::vector<MoveRecord> GenerateCounterRemovalMoves(const GameState& state) {
    std::vector<MoveRecord> moves;
    if (state.phase != Phase::kCounterRemoval) {
        return moves;
    }
    Player remover = state.current_player;
    Player stuck_player = Opponent(remover);
    int stuck_value = PlayerValue(stuck_player);
    MarkSet empty_marked;

    std::vector<Coord> stuck_pieces;
    std::vector<Coord> stuck_normal;
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            if (state.BoardAt(r, c) == stuck_value) {
                Coord pos{r, c};
                stuck_pieces.push_back(pos);
                if (!IsPieceInShape(state, r, c, stuck_value, empty_marked)) {
                    stuck_normal.push_back(pos);
                }
            }
        }
    }

    const std::vector<Coord>& targets =
        !stuck_normal.empty() ? stuck_normal : stuck_pieces;
    moves.reserve(targets.size());
    for (const auto& pos : targets) {
        moves.push_back(MoveRecord::CounterRemoval(pos));
    }
    return moves;
}

std::vector<MoveRecord> GenerateAllLegalMoves(const GameState& state) {
    if (state.IsGameOver()) {
        return {};
    }

    switch (state.phase) {
        case Phase::kPlacement: {
            auto positions = GeneratePlacementPositions(state);
            std::vector<MoveRecord> moves;
            moves.reserve(positions.size());
            for (const auto& pos : positions) {
                moves.push_back(MoveRecord::Placement(pos));
            }
            return moves;
        }
        case Phase::kMarkSelection: {
            auto targets = GenerateMarkTargets(state);
            std::vector<MoveRecord> moves;
            moves.reserve(targets.size());
            for (const auto& pos : targets) {
                moves.push_back(MoveRecord::Mark(pos));
            }
            return moves;
        }
        case Phase::kRemoval: {
            return {MoveRecord::ProcessRemoval()};
        }
        case Phase::kForcedRemoval:
            return GenerateForcedRemovalMoves(state);
        case Phase::kMovement: {
            if (HasLegalMovementMoves(state)) {
                auto movement_pairs = GenerateMovementMoves(state);
                std::vector<MoveRecord> moves;
                moves.reserve(movement_pairs.size());
                for (const auto& mv : movement_pairs) {
                    moves.push_back(MoveRecord::Movement(mv.first, mv.second));
                }
                return moves;
            }
            return GenerateNoMovesOptions(state);
        }
        case Phase::kCaptureSelection: {
            auto targets = GenerateCaptureTargets(state);
            std::vector<MoveRecord> moves;
            moves.reserve(targets.size());
            for (const auto& pos : targets) {
                moves.push_back(MoveRecord::Capture(pos));
            }
            return moves;
        }
        case Phase::kCounterRemoval:
            return GenerateCounterRemovalMoves(state);
        default:
            return {};
    }
}

ActionCode EncodeAction(const MoveRecord& move) {
    ActionCode code;
    code.extra = 0;

    switch (move.action_type) {
        case ActionType::kPlace:
            code.kind = kActionPlacement;
            code.primary = CoordToCell(move.Position());
            code.secondary = 0;
            break;
        case ActionType::kMove:
            code.kind = kActionMovement;
            code.primary = CoordToCell(move.From());
            code.secondary = CoordToCell(move.To());
            break;
        case ActionType::kMark:
            code.kind = kActionMarkSelection;
            code.primary = CoordToCell(move.Position());
            code.secondary = 0;
            break;
        case ActionType::kCapture:
            code.kind = kActionCaptureSelection;
            code.primary = CoordToCell(move.Position());
            code.secondary = 0;
            break;
        case ActionType::kForcedRemoval:
            code.kind = kActionForcedRemovalSelection;
            code.primary = CoordToCell(move.Position());
            code.secondary = 0;
            break;
        case ActionType::kCounterRemoval:
            code.kind = kActionCounterRemovalSelection;
            code.primary = CoordToCell(move.Position());
            code.secondary = 0;
            break;
        case ActionType::kNoMovesRemoval:
            code.kind = kActionNoMovesRemovalSelection;
            code.primary = CoordToCell(move.Position());
            code.secondary = 0;
            break;
        case ActionType::kProcessRemoval:
            code.kind = kActionProcessRemoval;
            code.primary = 0;
            code.secondary = 0;
            break;
        default:
            throw std::runtime_error("Unknown action type.");
    }

    return code;
}

std::vector<ActionCode> EncodeActions(const std::vector<MoveRecord>& moves) {
    std::vector<ActionCode> codes;
    codes.reserve(moves.size());
    for (const auto& move : moves) {
        codes.push_back(EncodeAction(move));
    }
    return codes;
}

GameState ApplyMove(const GameState& state, const MoveRecord& move, bool quiet) {
    if (move.phase != state.phase) {
        throw std::runtime_error("Move phase does not match state phase.");
    }
    GameState new_state;
    switch (state.phase) {
        case Phase::kPlacement:
            if (move.action_type != ActionType::kPlace) {
                throw std::runtime_error("Placement phase only allows 'place'.");
            }
            new_state = ApplyPlacementMove(state, move.Position());
            break;
        case Phase::kMarkSelection:
            if (move.action_type != ActionType::kMark) {
                throw std::runtime_error("Mark phase only allows 'mark'.");
            }
            new_state = ApplyMarkSelection(state, move.Position());
            break;
        case Phase::kRemoval:
            if (move.action_type != ActionType::kProcessRemoval) {
                throw std::runtime_error("Removal phase only allows 'process_removal'.");
            }
            new_state = ProcessPhase2Removals(state);
            break;
        case Phase::kForcedRemoval:
            if (move.action_type != ActionType::kForcedRemoval) {
                throw std::runtime_error("Forced removal phase only allows 'remove'.");
            }
            new_state = ApplyForcedRemoval(state, move.Position());
            break;
        case Phase::kMovement:
            if (move.action_type == ActionType::kMove) {
                Move pair{move.From(), move.To()};
                new_state = ApplyMovementMove(state, pair, quiet);
            } else if (move.action_type == ActionType::kNoMovesRemoval) {
                new_state = HandleNoMovesPhase3(state, move.Position(), quiet);
            } else {
                throw std::runtime_error("Unknown action for movement phase.");
            }
            break;
        case Phase::kCaptureSelection:
            if (move.action_type != ActionType::kCapture) {
                throw std::runtime_error("Capture phase only allows 'capture'.");
            }
            new_state = ApplyCaptureSelection(state, move.Position(), quiet);
            break;
        case Phase::kCounterRemoval:
            if (move.action_type != ActionType::kCounterRemoval) {
                throw std::runtime_error("Counter removal phase only allows 'counter_remove'.");
            }
            new_state = ApplyCounterRemovalPhase3(state, move.Position(), quiet);
            break;
        default:
            throw std::runtime_error("Unsupported phase for apply_move.");
    }

    new_state.move_count = state.move_count + 1;
    return new_state;
}

std::pair<std::vector<MoveRecord>, std::vector<ActionCode>> GenerateMovesWithCodes(
    const GameState& state) {
    auto moves = GenerateAllLegalMoves(state);
    auto codes = EncodeActions(moves);
    return {std::move(moves), std::move(codes)};
}

}  // namespace v0
