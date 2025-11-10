#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "v0/game_state.hpp"
#include "v0/rule_engine.hpp"

namespace v0 {

enum class ActionType : int32_t {
    kPlace = 1,
    kMove = 2,
    kMark = 3,
    kCapture = 4,
    kForcedRemoval = 5,
    kCounterRemoval = 6,
    kNoMovesRemoval = 7,
    kProcessRemoval = 8,
};

struct MoveRecord {
    Phase phase{Phase::kPlacement};
    ActionType action_type{ActionType::kPlace};
    Coord primary{-1, -1};   // general position or movement source
    Coord secondary{-1, -1}; // movement destination when applicable

    static MoveRecord Placement(const Coord& pos);
    static MoveRecord Mark(const Coord& pos);
    static MoveRecord Capture(const Coord& pos);
    static MoveRecord ForcedRemoval(const Coord& pos);
    static MoveRecord CounterRemoval(const Coord& pos);
    static MoveRecord NoMovesRemoval(const Coord& pos);
    static MoveRecord ProcessRemoval();
    static MoveRecord Movement(const Coord& from, const Coord& to);

    bool HasPosition() const;
    Coord Position() const;
    bool HasFrom() const;
    Coord From() const;
    bool HasTo() const;
    Coord To() const;
};

struct ActionCode {
    int32_t kind{0};
    int32_t primary{0};
    int32_t secondary{0};
    int32_t extra{0};
};

std::vector<MoveRecord> GenerateAllLegalMoves(const GameState& state);
std::vector<MoveRecord> GenerateForcedRemovalMoves(const GameState& state);
std::vector<MoveRecord> GenerateNoMovesOptions(const GameState& state);
std::vector<MoveRecord> GenerateCounterRemovalMoves(const GameState& state);

std::vector<ActionCode> EncodeActions(const std::vector<MoveRecord>& moves);
ActionCode EncodeAction(const MoveRecord& move);

GameState ApplyMove(const GameState& state, const MoveRecord& move, bool quiet = false);

std::pair<std::vector<MoveRecord>, std::vector<ActionCode>> GenerateMovesWithCodes(
    const GameState& state);

const char* ActionTypeToString(ActionType type);

}  // namespace v0
