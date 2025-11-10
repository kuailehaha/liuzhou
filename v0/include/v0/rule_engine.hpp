#pragma once

#include <utility>
#include <vector>

#include "v0/game_state.hpp"

namespace v0 {

using Coord = std::pair<int, int>;
using Move = std::pair<Coord, Coord>;

std::vector<Coord> GeneratePlacementPositions(const GameState& state);
GameState ApplyPlacementMove(const GameState& state, const Coord& position);

std::vector<Coord> GenerateMarkTargets(const GameState& state);
GameState ApplyMarkSelection(const GameState& state, const Coord& position);

GameState ProcessPhase2Removals(const GameState& state);

std::vector<Move> GenerateMovementMoves(const GameState& state);
bool HasLegalMovementMoves(const GameState& state);
GameState ApplyMovementMove(const GameState& state, const Move& move, bool quiet = false);

std::vector<Coord> GenerateCaptureTargets(const GameState& state);
GameState ApplyCaptureSelection(const GameState& state, const Coord& position, bool quiet = false);

GameState ApplyForcedRemoval(const GameState& state, const Coord& piece_to_remove);
GameState HandleNoMovesPhase3(const GameState& state, const Coord& stucked_player_removes, bool quiet = false);
GameState ApplyCounterRemovalPhase3(
    const GameState& state,
    const Coord& opponent_removes,
    bool quiet = false);

std::vector<Coord> GenerateLegalMovesPhase1(const GameState& state);
GameState ApplyMovePhase1(
    const GameState& state,
    const Coord& move,
    const std::vector<Coord>& mark_positions);

std::vector<Move> GenerateLegalMovesPhase3(const GameState& state);
bool HasLegalMovesPhase3(const GameState& state);
GameState ApplyMovePhase3(
    const GameState& state,
    const Move& move,
    const std::vector<Coord>& capture_positions,
    bool quiet = false);

}  // namespace v0
