#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>   // ← 新增
#include <torch/extension.h>

#include <memory>
#include <string>

#include "v0/fast_legal_mask.hpp"
#include "v0/fast_apply_moves.hpp"
#include "v0/game_state.hpp"
#include "v0/move_generator.hpp"
#include "v0/net_encoding.hpp"
#include "v0/mcts_core.hpp"
#include "v0/project_policy.hpp"
#include "v0/tensor_state_batch.hpp"
#include "v0/rule_engine.hpp"
#include "v0/torchscript_runner.hpp"
#include "v0/inference_engine.hpp"

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

v0::GameState GameStateFromPyLike(const py::object& py_state) {
    if (!py_state || py_state.is_none()) {
        throw std::runtime_error("state 不能为空");
    }

    auto require_attr = [&](const char* name) -> py::object {
        if (!py::hasattr(py_state, name)) {
            throw std::runtime_error(std::string("Python GameState 缺少属性: ") + name);
        }
        return py_state.attr(name);
    };

    auto to_int = [](const py::object& obj, const char* attr_name) -> int {
        py::object value_obj = obj;
        if (py::hasattr(obj, "value")) {
            value_obj = obj.attr("value");
        }
        try {
            return value_obj.cast<int>();
        } catch (const py::cast_error&) {
            throw std::runtime_error(std::string("无法把属性转换为 int: ") + attr_name);
        }
    };

    v0::GameState state;

    py::list board_list = py::list(require_attr("board"));
    BoardFromNested(state, board_list.cast<std::vector<std::vector<int>>>());

    auto parse_phase = [&](const py::object& obj) -> v0::Phase {
        int value = to_int(obj, "phase");
        if (value < static_cast<int>(v0::Phase::kPlacement) ||
            value > static_cast<int>(v0::Phase::kCounterRemoval)) {
            throw std::runtime_error("phase 枚举值超出范围");
        }
        return static_cast<v0::Phase>(value);
    };

    auto parse_player = [&](const py::object& obj) -> v0::Player {
        int value = to_int(obj, "current_player");
        if (value == static_cast<int>(v0::Player::kBlack)) {
            return v0::Player::kBlack;
        }
        if (value == static_cast<int>(v0::Player::kWhite)) {
            return v0::Player::kWhite;
        }
        throw std::runtime_error("current_player 枚举值必须是 1 或 -1");
    };

    state.phase = parse_phase(require_attr("phase"));
    state.current_player = parse_player(require_attr("current_player"));

    auto assign_marks = [&](const char* attr_name, v0::Player player) {
        if (!py::hasattr(py_state, attr_name)) {
            state.Marks(player).Clear();
            return;
        }
        py::object marks_obj = py_state.attr(attr_name);
        if (marks_obj.is_none()) {
            state.Marks(player).Clear();
            return;
        }
        std::vector<v0::Coord> coords = py::list(marks_obj).cast<std::vector<v0::Coord>>();
        MarksFromVector(state, player, coords);
    };

    assign_marks("marked_black", v0::Player::kBlack);
    assign_marks("marked_white", v0::Player::kWhite);

    auto assign_optional_int = [&](const char* attr_name, int32_t& field) {
        if (!py::hasattr(py_state, attr_name)) {
            return;
        }
        py::object attr = py_state.attr(attr_name);
        if (attr.is_none()) {
            return;
        }
        field = attr.cast<int32_t>();
    };

    assign_optional_int("forced_removals_done", state.forced_removals_done);
    assign_optional_int("move_count", state.move_count);
    assign_optional_int("pending_marks_required", state.pending_marks_required);
    assign_optional_int("pending_marks_remaining", state.pending_marks_remaining);
    assign_optional_int("pending_captures_required", state.pending_captures_required);
    assign_optional_int("pending_captures_remaining", state.pending_captures_remaining);

    return state;
}

v0::GameState CoerceGameStateLike(const py::object& state_obj) {
    if (!state_obj || state_obj.is_none()) {
        throw std::runtime_error("state 不能为空");
    }
    if (py::isinstance<v0::GameState>(state_obj)) {
        return state_obj.cast<v0::GameState>();
    }
    return GameStateFromPyLike(state_obj);
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

    py::enum_<v0::ActionType>(m, "ActionType")
        .value("PLACE", v0::ActionType::kPlace)
        .value("MOVE", v0::ActionType::kMove)
        .value("MARK", v0::ActionType::kMark)
        .value("CAPTURE", v0::ActionType::kCapture)
        .value("FORCED_REMOVAL", v0::ActionType::kForcedRemoval)
        .value("COUNTER_REMOVAL", v0::ActionType::kCounterRemoval)
        .value("NO_MOVES_REMOVAL", v0::ActionType::kNoMovesRemoval)
        .value("PROCESS_REMOVAL", v0::ActionType::kProcessRemoval)
        .export_values();

    py::class_<v0::MoveRecord>(m, "MoveRecord")
        .def_property_readonly("phase", [](const v0::MoveRecord& move) { return move.phase; })
        .def_property_readonly(
            "action_type",
            [](const v0::MoveRecord& move) { return move.action_type; })
        .def_property_readonly(
            "action_type_name",
            [](const v0::MoveRecord& move) { return std::string(v0::ActionTypeToString(move.action_type)); })
        .def_property_readonly(
            "position",
            [](const v0::MoveRecord& move) -> py::object {
                if (move.HasPosition()) {
                    return py::cast(move.Position());
                }
                return py::none();
            })
        .def_property_readonly(
            "from_position",
            [](const v0::MoveRecord& move) -> py::object {
                if (move.HasFrom()) {
                    return py::cast(move.From());
                }
                return py::none();
            })
        .def_property_readonly(
            "to_position",
            [](const v0::MoveRecord& move) -> py::object {
                if (move.HasTo()) {
                    return py::cast(move.To());
                }
                return py::none();
            })
        .def(
            "to_dict",
            [](const v0::MoveRecord& move) {
                py::dict d;
                d["phase"] = move.phase;
                d["action_type"] = py::str(v0::ActionTypeToString(move.action_type));
                if (move.action_type == v0::ActionType::kMove) {
                    d["from_position"] = move.From();
                    d["to_position"] = move.To();
                } else if (move.HasPosition()) {
                    d["position"] = move.Position();
                }
                return d;
            })
        .def_static("placement", &v0::MoveRecord::Placement, py::arg("position"))
        .def_static("mark", &v0::MoveRecord::Mark, py::arg("position"))
        .def_static("capture", &v0::MoveRecord::Capture, py::arg("position"))
        .def_static("forced_removal", &v0::MoveRecord::ForcedRemoval, py::arg("position"))
        .def_static("counter_removal", &v0::MoveRecord::CounterRemoval, py::arg("position"))
        .def_static("no_moves_removal", &v0::MoveRecord::NoMovesRemoval, py::arg("position"))
        .def_static("process_removal", &v0::MoveRecord::ProcessRemoval)
        .def_static(
            "movement",
            &v0::MoveRecord::Movement,
            py::arg("from_position"),
            py::arg("to_position"));

    py::class_<v0::ActionCode>(m, "ActionCode")
        .def(py::init<>())
        .def_readwrite("kind", &v0::ActionCode::kind)
        .def_readwrite("primary", &v0::ActionCode::primary)
        .def_readwrite("secondary", &v0::ActionCode::secondary)
        .def_readwrite("extra", &v0::ActionCode::extra)
        .def(
            "to_tuple",
            [](const v0::ActionCode& code) {
                return py::make_tuple(code.kind, code.primary, code.secondary, code.extra);
            });

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

    m.def(
        "generate_all_legal_moves_struct",
        &v0::GenerateAllLegalMoves,
        py::arg("state"));
    m.def(
        "generate_moves_with_codes",
        &v0::GenerateMovesWithCodes,
        py::arg("state"));
    m.def(
        "generate_forced_removal_moves_struct",
        &v0::GenerateForcedRemovalMoves,
        py::arg("state"));
    m.def(
        "generate_no_moves_options_struct",
        &v0::GenerateNoMovesOptions,
        py::arg("state"));
    m.def(
        "generate_counter_removal_moves_struct",
        &v0::GenerateCounterRemovalMoves,
        py::arg("state"));
    m.def("encode_action_codes", &v0::EncodeActions, py::arg("moves"));
    m.def("encode_action_code", &v0::EncodeAction, py::arg("move"));
    m.def(
        "apply_move_struct",
        &v0::ApplyMove,
        py::arg("state"),
        py::arg("move"),
        py::arg("quiet") = false);

    py::class_<v0::TensorStateBatch>(m, "TensorStateBatch")
        .def(py::init<>())
        .def_property_readonly("board", [](const v0::TensorStateBatch& batch) { return batch.board; })
        .def_property_readonly("marks_black", [](const v0::TensorStateBatch& batch) { return batch.marks_black; })
        .def_property_readonly("marks_white", [](const v0::TensorStateBatch& batch) { return batch.marks_white; })
        .def_property_readonly("phase", [](const v0::TensorStateBatch& batch) { return batch.phase; })
        .def_property_readonly("current_player", [](const v0::TensorStateBatch& batch) { return batch.current_player; })
        .def_property_readonly(
            "pending_marks_required",
            [](const v0::TensorStateBatch& batch) { return batch.pending_marks_required; })
        .def_property_readonly(
            "pending_marks_remaining",
            [](const v0::TensorStateBatch& batch) { return batch.pending_marks_remaining; })
        .def_property_readonly(
            "pending_captures_required",
            [](const v0::TensorStateBatch& batch) { return batch.pending_captures_required; })
        .def_property_readonly(
            "pending_captures_remaining",
            [](const v0::TensorStateBatch& batch) { return batch.pending_captures_remaining; })
        .def_property_readonly(
            "forced_removals_done",
            [](const v0::TensorStateBatch& batch) { return batch.forced_removals_done; })
        .def_property_readonly("move_count", [](const v0::TensorStateBatch& batch) { return batch.move_count; })
        .def_property_readonly("mask_alive", [](const v0::TensorStateBatch& batch) { return batch.mask_alive; })
        .def_property_readonly("board_size", [](const v0::TensorStateBatch& batch) { return batch.board_size; })
        .def("device", [](const v0::TensorStateBatch& batch) { return batch.board.device(); })
        .def(
            "to",
            [](const v0::TensorStateBatch& batch, const std::string& device) {
                return batch.To(torch::Device(device));
            },
            py::arg("device"))
        .def("clone", &v0::TensorStateBatch::Clone);

    m.def(
        "tensor_batch_from_game_states",
        [](const std::vector<v0::GameState>& states, const std::string& device) {
            return v0::FromGameStates(states, torch::Device(device));
        },
        py::arg("states"),
        py::arg("device") = std::string("cpu"));
    m.def(
        "tensor_batch_to_game_states",
        &v0::ToGameStates,
        py::arg("batch"));

    py::class_<v0::MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &v0::MCTSConfig::num_simulations)
        .def_readwrite("exploration_weight", &v0::MCTSConfig::exploration_weight)
        .def_readwrite("temperature", &v0::MCTSConfig::temperature)
        .def_readwrite("add_dirichlet_noise", &v0::MCTSConfig::add_dirichlet_noise)
        .def_readwrite("dirichlet_alpha", &v0::MCTSConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &v0::MCTSConfig::dirichlet_epsilon)
        .def_readwrite("batch_size", &v0::MCTSConfig::batch_size)
        .def_readwrite("virtual_loss", &v0::MCTSConfig::virtual_loss)
        .def_readwrite("seed", &v0::MCTSConfig::seed)
        .def_property(
            "device",
            [](const v0::MCTSConfig& cfg) { return cfg.device.str(); },
            [](v0::MCTSConfig& cfg, const std::string& dev) { cfg.device = torch::Device(dev); });

    py::class_<v0::MCTSCore>(m, "MCTSCore")
        .def(py::init<v0::MCTSConfig>())
        .def(
            "set_forward_callback",
            [](v0::MCTSCore& core, py::function fn) {
                py::object fn_keep = fn;
                core.SetForwardCallback([fn_keep](const torch::Tensor& inputs) {
                    py::gil_scoped_acquire gil;
                    py::object result = fn_keep(inputs);
                    py::tuple tup = result.cast<py::tuple>();
                    if (tup.size() != 4) {
                        throw std::runtime_error("forward callback must return a tuple(log_p1, log_p2, log_pmc, value)");
                    }
                    torch::Tensor log_p1 = tup[0].cast<torch::Tensor>();
                    torch::Tensor log_p2 = tup[1].cast<torch::Tensor>();
                    torch::Tensor log_pmc = tup[2].cast<torch::Tensor>();
                    torch::Tensor value = tup[3].cast<torch::Tensor>();
                    return std::make_tuple(log_p1, log_p2, log_pmc, value);
                });
            },
            py::arg("callback"))
        .def(
            "set_torchscript_runner",
            [](v0::MCTSCore& core, std::shared_ptr<v0::TorchScriptRunner> runner) {
                if (!runner) {
                    throw std::runtime_error("TorchScriptRunner is null");
                }
                core.SetForwardCallback([runner](const torch::Tensor& inputs) {
                    return runner->Forward(inputs);
                });
            },
            py::arg("runner"))
        .def(
            "set_inference_engine",
            [](v0::MCTSCore& core, std::shared_ptr<v0::InferenceEngine> engine) {
                if (!engine) {
                    throw std::runtime_error("InferenceEngine is null");
                }
                core.SetForwardCallback([engine](const torch::Tensor& inputs) {
                    return engine->Forward(inputs, inputs.size(0));
                });
            },
            py::arg("engine"))
        .def(
            "set_root_state",
            [](v0::MCTSCore& core, const v0::GameState& state) { core.SetRootState(state); },
            py::arg("state"))
        .def(
            "set_root_state",
            [](v0::MCTSCore& core, py::object state_like) {
                core.SetRootState(CoerceGameStateLike(state_like));
            },
            py::arg("state"))
        .def("reset", &v0::MCTSCore::Reset)
        .def("run_simulations", &v0::MCTSCore::RunSimulations, py::arg("num_simulations"))
        .def(
            "get_policy",
            [](const v0::MCTSCore& core, double temperature) {
                return core.GetPolicy(temperature);
            },
            py::arg("temperature") = 1.0)
        .def(
            "get_root_children_stats",
            [](const v0::MCTSCore& core) {
                py::list result;
                for (const auto& stats : core.GetRootChildrenStats()) {
                    py::dict entry;
                    entry["action_index"] = stats.action_index;
                    entry["prior"] = stats.prior;
                    entry["visit_count"] = stats.visit_count;
                    entry["value_sum"] = stats.value_sum;
                    result.append(entry);
                }
                return result;
            })
        .def("advance_root", &v0::MCTSCore::AdvanceRoot, py::arg("action_index"))
        .def_property_readonly("root_value", &v0::MCTSCore::RootValue)
        .def_property_readonly("root_visit_count", &v0::MCTSCore::RootVisitCount)
        .def_property_readonly("root_state", &v0::MCTSCore::RootState, py::return_value_policy::reference_internal);

    m.def(
        "states_to_model_input",
        &v0::states_to_model_input,
        py::arg("board"),
        py::arg("marks_black"),
        py::arg("marks_white"),
        py::arg("phase"),
        py::arg("current_player"));
    m.def(
        "encode_actions_fast",
        &v0::encode_actions_fast,
        py::arg("board"),
        py::arg("marks_black"),
        py::arg("marks_white"),
        py::arg("phase"),
        py::arg("current_player"),
        py::arg("pending_marks_required"),
        py::arg("pending_marks_remaining"),
        py::arg("pending_captures_required"),
        py::arg("pending_captures_remaining"),
        py::arg("forced_removals_done"),
        py::arg("placement_dim"),
        py::arg("movement_dim"),
        py::arg("selection_dim"),
        py::arg("auxiliary_dim"));
    m.def(
        "batch_apply_moves",
        &v0::batch_apply_moves,
        py::arg("board"),
        py::arg("marks_black"),
        py::arg("marks_white"),
        py::arg("phase"),
        py::arg("current_player"),
        py::arg("pending_marks_required"),
        py::arg("pending_marks_remaining"),
        py::arg("pending_captures_required"),
        py::arg("pending_captures_remaining"),
        py::arg("forced_removals_done"),
        py::arg("move_count"),
        py::arg("action_codes"),
        py::arg("parent_indices"));
    m.def(
        "project_policy_logits_fast",
        &v0::project_policy_logits_fast,
        py::arg("log_p1"),
        py::arg("log_p2"),
        py::arg("log_pmc"),
        py::arg("legal_mask"),
        py::arg("placement_dim"),
        py::arg("movement_dim"),
        py::arg("selection_dim"),
        py::arg("auxiliary_dim"));
    m.def(
        "postprocess_value_head",
        &v0::postprocess_value_head,
        py::arg("raw_values"));
    m.def(
        "apply_temperature_scaling",
        &v0::apply_temperature_scaling,
        py::arg("probs"),
        py::arg("temperature"),
        py::arg("dim") = -1);

    py::class_<v0::InferenceEngine, std::shared_ptr<v0::InferenceEngine>>(m, "InferenceEngine")
        .def(
            py::init<const std::string&, const std::string&, const std::string&, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
            py::arg("path"),
            py::arg("device") = std::string("cuda"),
            py::arg("dtype") = std::string("float16"),
            py::arg("batch_size") = 512,
            py::arg("input_channels") = 11,
            py::arg("height") = v0::kBoardSize,
            py::arg("width") = v0::kBoardSize,
            py::arg("warmup_iters") = 5,
            py::arg("use_inference_mode") = true)
        .def("forward", &v0::InferenceEngine::Forward, py::arg("input"), py::arg("n_valid") = -1)
        .def_property_readonly("device", &v0::InferenceEngine::DeviceString)
        .def_property_readonly("dtype", &v0::InferenceEngine::DTypeString)
        .def_property_readonly("batch_size", &v0::InferenceEngine::BatchSize)
        .def_property_readonly("graph_enabled", &v0::InferenceEngine::GraphEnabled);

    py::class_<v0::TorchScriptRunner, std::shared_ptr<v0::TorchScriptRunner>>(m, "TorchScriptRunner")
        .def(
            py::init<const std::string&, const std::string&, const std::string&, bool>(),
            py::arg("path"),
            py::arg("device") = std::string("cpu"),
            py::arg("dtype") = std::string("auto"),
            py::arg("use_inference_mode") = true)
        .def("forward", &v0::TorchScriptRunner::Forward, py::arg("input"))
        .def_property_readonly("device", &v0::TorchScriptRunner::DeviceString)
        .def_property_readonly("dtype", &v0::TorchScriptRunner::DTypeString);

    m.def("version", []() { return std::string("v0-core"); });
}
