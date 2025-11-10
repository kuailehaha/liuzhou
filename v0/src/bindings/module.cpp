#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "v0/game_state.hpp"
#include "v0/rule_engine.hpp"

namespace py = pybind11;

namespace {

bool InBounds(const v0::Coord& coord) {
    return coord.first >= 0 && coord.first < v0::kBoardSize && coord.second >= 0 &&
        coord.second < v0::kBoardSize;
}

std::vector<std::vector<int>> BoardToNested(const v0::GameState& state) {
    std::vector<std::vector<int>> board(v0::kBoardSize, std::vector<int>(v0::kBoardSize));
    for (int r = 0; r < v0::kBoardSize; ++r) {
        for (int c = 0; c < v0::kBoardSize; ++c) {
            board[r][c] = state.BoardAt(r, c);
        }
    }
    return board;
}

void BoardFromNested(v0::GameState& state, const std::vector<std::vector<int>>& board) {
    if (board.size() != static_cast<size_t>(v0::kBoardSize)) {
        throw std::runtime_error("board 必须是 6x6 列表");
    }
    for (size_t r = 0; r < board.size(); ++r) {
        if (board[r].size() != static_cast<size_t>(v0::kBoardSize)) {
            throw std::runtime_error("board 必须是 6x6 列表");
        }
        for (size_t c = 0; c < board[r].size(); ++c) {
            int value = board[r][c];
            if (value < -1 || value > 1) {
                throw std::runtime_error("棋盘值必须在 [-1, 1] 之间");
            }
            state.SetBoard(static_cast<int>(r), static_cast<int>(c), static_cast<int8_t>(value));
        }
    }
}

std::vector<v0::Coord> MarksToVector(const v0::MarkSet& marks) {
    return marks.ToVector();
}

void MarksFromVector(
    v0::GameState& state,
    v0::Player player,
    const std::vector<v0::Coord>& coords) {
    auto& marked = state.Marks(player);
    marked.Clear();
    for (const auto& coord : coords) {
        if (!InBounds(coord)) {
            throw std::runtime_error("标记坐标超出棋盘范围");
        }
        marked.Add(coord);
    }
}
}  // namespace

PYBIND11_MODULE(v0_core, m) {
    m.doc() = "v0 C++ refactor core bindings";

    py::enum_<v0::Phase>(m, "Phase")
        .value("PLACEMENT", v0::Phase::kPlacement)
        .value("MARK_SELECTION", v0::Phase::kMarkSelection)
        .value("REMOVAL", v0::Phase::kRemoval)
        .value("MOVEMENT", v0::Phase::kMovement)
        .value("CAPTURE_SELECTION", v0::Phase::kCaptureSelection)
        .value("FORCED_REMOVAL", v0::Phase::kForcedRemoval)
        .value("COUNTER_REMOVAL", v0::Phase::kCounterRemoval)
        .export_values();

    py::enum_<v0::Player>(m, "Player")
        .value("BLACK", v0::Player::kBlack)
        .value("WHITE", v0::Player::kWhite)
        .export_values();

    py::class_<v0::GameState>(m, "GameState")
        .def(py::init<>())
        .def_property(
            "board",
            [](const v0::GameState& state) { return BoardToNested(state); },
            [](v0::GameState& state, const std::vector<std::vector<int>>& board) {
                BoardFromNested(state, board);
            })
        .def_property(
            "marked_black",
            [](const v0::GameState& state) { return MarksToVector(state.marked_black); },
            [](v0::GameState& state, const std::vector<v0::Coord>& coords) {
                MarksFromVector(state, v0::Player::kBlack, coords);
            })
        .def_property(
            "marked_white",
            [](const v0::GameState& state) { return MarksToVector(state.marked_white); },
            [](v0::GameState& state, const std::vector<v0::Coord>& coords) {
                MarksFromVector(state, v0::Player::kWhite, coords);
            })
        .def_readwrite("phase", &v0::GameState::phase)
        .def_readwrite("current_player", &v0::GameState::current_player)
        .def_readwrite("forced_removals_done", &v0::GameState::forced_removals_done)
        .def_readwrite("move_count", &v0::GameState::move_count)
        .def_readwrite("pending_marks_required", &v0::GameState::pending_marks_required)
        .def_readwrite("pending_marks_remaining", &v0::GameState::pending_marks_remaining)
        .def_readwrite("pending_captures_required", &v0::GameState::pending_captures_required)
        .def_readwrite("pending_captures_remaining", &v0::GameState::pending_captures_remaining)
        .def("copy", &v0::GameState::Copy)
        .def("switch_player", &v0::GameState::SwitchPlayer)
        .def("is_board_full", &v0::GameState::IsBoardFull)
        .def("count_player_pieces", &v0::GameState::CountPlayerPieces, py::arg("player"))
        .def("get_player_pieces", &v0::GameState::GetPlayerPieces, py::arg("player"));

    m.def("generate_placement_positions", &v0::GeneratePlacementPositions, py::arg("state"));
    m.def("apply_placement_move", &v0::ApplyPlacementMove, py::arg("state"), py::arg("position"));

    m.def("generate_mark_targets", &v0::GenerateMarkTargets, py::arg("state"));
    m.def("apply_mark_selection", &v0::ApplyMarkSelection, py::arg("state"), py::arg("position"));

    m.def("process_phase2_removals", &v0::ProcessPhase2Removals, py::arg("state"));

    m.def("generate_movement_moves", &v0::GenerateMovementMoves, py::arg("state"));
    m.def("has_legal_movement_moves", &v0::HasLegalMovementMoves, py::arg("state"));
    m.def(
        "apply_movement_move",
        &v0::ApplyMovementMove,
        py::arg("state"),
        py::arg("move"),
        py::arg("quiet") = false);

    m.def("generate_capture_targets", &v0::GenerateCaptureTargets, py::arg("state"));
    m.def(
        "apply_capture_selection",
        &v0::ApplyCaptureSelection,
        py::arg("state"),
        py::arg("position"),
        py::arg("quiet") = false);

    m.def(
        "apply_forced_removal",
        &v0::ApplyForcedRemoval,
        py::arg("state"),
        py::arg("piece_to_remove"));
    m.def(
        "handle_no_moves_phase3",
        &v0::HandleNoMovesPhase3,
        py::arg("state"),
        py::arg("stucked_player_removes"),
        py::arg("quiet") = false);
    m.def(
        "apply_counter_removal_phase3",
        &v0::ApplyCounterRemovalPhase3,
        py::arg("state"),
        py::arg("opponent_removes"),
        py::arg("quiet") = false);

    m.def("generate_legal_moves_phase1", &v0::GenerateLegalMovesPhase1, py::arg("state"));
    m.def(
        "apply_move_phase1",
        [](const v0::GameState& state, const v0::Coord& move, py::object marks) {
            if (marks.is_none()) {
                static const std::vector<v0::Coord> empty;
                return v0::ApplyMovePhase1(state, move, empty);
            }
            auto mark_vec = marks.cast<std::vector<v0::Coord>>();
            return v0::ApplyMovePhase1(state, move, mark_vec);
        },
        py::arg("state"),
        py::arg("move"),
        py::arg("mark_positions") = py::none());

    m.def("generate_legal_moves_phase3", &v0::GenerateLegalMovesPhase3, py::arg("state"));
    m.def("has_legal_moves_phase3", &v0::HasLegalMovesPhase3, py::arg("state"));
    m.def(
        "apply_move_phase3",
        [](const v0::GameState& state, const v0::Move& move, py::object captures, bool quiet) {
            if (captures.is_none()) {
                static const std::vector<v0::Coord> empty;
                return v0::ApplyMovePhase3(state, move, empty, quiet);
            }
            auto capture_vec = captures.cast<std::vector<v0::Coord>>();
            return v0::ApplyMovePhase3(state, move, capture_vec, quiet);
        },
        py::arg("state"),
        py::arg("move"),
        py::arg("capture_positions") = py::none(),
        py::arg("quiet") = false);

    m.def("version", []() { return std::string("v0-core"); });
}
