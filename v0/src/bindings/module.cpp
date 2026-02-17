#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>   // ← 新增
#include <torch/extension.h>

#include <limits>

#include <memory>
#include <string>

#include "v0/fast_legal_mask.hpp"
#include "v0/fast_apply_moves.hpp"
#include "v0/game_state.hpp"
#include "v0/move_generator.hpp"
#include "v0/net_encoding.hpp"
#include "v0/mcts_core.hpp"
#include "v0/project_policy.hpp"
#include "v0/root_puct_fused.hpp"
#include "v0/tensor_state_batch.hpp"
#include "v0/rule_engine.hpp"
#include "v0/torchscript_runner.hpp"
#include "v0/inference_engine.hpp"
#include "v0/eval_batcher.hpp"

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RootPuctAllocateVisits(
    const torch::Tensor& priors,
    const torch::Tensor& leaf_values,
    const torch::Tensor& valid_mask,
    int64_t num_simulations,
    double exploration_weight) {
    TORCH_CHECK(priors.dim() == 2, "priors must be 2D [R, A]");
    TORCH_CHECK(leaf_values.dim() == 2, "leaf_values must be 2D [R, A]");
    TORCH_CHECK(valid_mask.dim() == 2, "valid_mask must be 2D [R, A]");
    TORCH_CHECK(priors.sizes() == leaf_values.sizes(), "priors and leaf_values shape mismatch");
    TORCH_CHECK(priors.sizes() == valid_mask.sizes(), "priors and valid_mask shape mismatch");
    TORCH_CHECK(num_simulations > 0, "num_simulations must be positive");
    TORCH_CHECK(priors.is_cuda() == leaf_values.is_cuda(), "priors/leaf_values must be on same device");
    TORCH_CHECK(priors.is_cuda() == valid_mask.is_cuda(), "priors/valid_mask must be on same device");

    if (priors.is_cuda()) {
#if defined(V0_HAS_CUDA_ROOT_PUCT)
        return v0::root_puct_allocate_visits_cuda(
            priors,
            leaf_values,
            valid_mask,
            num_simulations,
            exploration_weight);
#else
        TORCH_CHECK(
            false,
            "root_puct_allocate_visits: requested CUDA tensors but CUDA kernel was not built. "
            "Rebuild with -DBUILD_CUDA_KERNELS=ON.");
#endif
    }

    auto priors_f = priors.to(torch::kFloat32).contiguous();
    auto leaf_f = leaf_values.to(torch::kFloat32).contiguous();
    auto mask_b = valid_mask.to(torch::kBool).contiguous();

    auto visits = torch::zeros_like(priors_f);
    auto value_sum = torch::zeros_like(priors_f);
    auto total_visit = torch::zeros(
        {priors_f.size(0)},
        torch::TensorOptions().dtype(torch::kFloat32).device(priors_f.device()));
    auto neg_inf = -std::numeric_limits<float>::infinity();

    for (int64_t sim = 0; sim < num_simulations; ++sim) {
        auto q = torch::where(
            visits > 0,
            value_sum / visits.clamp_min(1e-8),
            torch::zeros_like(value_sum));
        auto u = static_cast<float>(exploration_weight) * priors_f *
            torch::sqrt(total_visit + 1.0).unsqueeze(1) / (1.0 + visits);
        auto scores = (q + u).masked_fill(mask_b.logical_not(), neg_inf);
        auto selected = std::get<1>(scores.max(1, false));  // [R], int64

        auto selected_col = selected.unsqueeze(1);
        auto one = torch::ones(
            {selected_col.size(0), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(priors_f.device()));
        visits = visits.scatter_add(1, selected_col, one);

        auto selected_leaf = leaf_f.gather(1, selected_col);
        value_sum = value_sum.scatter_add(1, selected_col, selected_leaf);
        total_visit = total_visit + 1.0;
    }

    auto root_values = value_sum.sum(1) / visits.sum(1).clamp_min(1.0);
    return std::make_tuple(visits, value_sum, root_values);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
RootPackSparseActions(
    const torch::Tensor& legal_mask,
    const torch::Tensor& probs,
    const torch::Tensor& metadata) {
    TORCH_CHECK(legal_mask.dim() == 2, "legal_mask must be 2D [B, A]");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2D [B, A]");
    TORCH_CHECK(metadata.dim() == 3, "metadata must be 3D [B, A, 4]");
    TORCH_CHECK(metadata.size(2) == 4, "metadata last dim must be 4");
    TORCH_CHECK(legal_mask.size(0) == probs.size(0) && legal_mask.size(1) == probs.size(1), "legal/probs shape mismatch");
    TORCH_CHECK(metadata.size(0) == legal_mask.size(0) && metadata.size(1) == legal_mask.size(1), "legal/metadata shape mismatch");
    TORCH_CHECK(
        legal_mask.is_cuda() == probs.is_cuda() && legal_mask.is_cuda() == metadata.is_cuda(),
        "legal_mask/probs/metadata must be on the same device type");
    TORCH_CHECK(
        legal_mask.device() == probs.device() && legal_mask.device() == metadata.device(),
        "legal_mask/probs/metadata must be on the same device");

    auto device = legal_mask.device();
    auto legal_mask_b = legal_mask.to(torch::kBool).contiguous();
    auto probs_f = probs.to(torch::kFloat32).contiguous();
    auto metadata_i = metadata.to(torch::kInt32).contiguous();

    const int64_t batch_size = legal_mask_b.size(0);
    const int64_t total_actions = legal_mask_b.size(1);
    auto row_counts = legal_mask_b.sum(1, false, torch::kInt64);
    auto terminal_mask = row_counts.eq(0);
    auto valid_root_indices = torch::nonzero(terminal_mask.logical_not()).view(-1);
    auto counts = row_counts.index_select(0, valid_root_indices).to(torch::kInt64);

    if (valid_root_indices.numel() == 0) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
        auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
        return std::make_tuple(
            terminal_mask,
            valid_root_indices,
            counts,
            torch::empty({0, 0}, bool_opts),
            torch::empty({0, 0}, long_opts),
            torch::empty({0, 0}, float_opts),
            torch::empty({0, 0, 4}, int_opts),
            torch::empty({0}, long_opts),
            torch::empty({0, 4}, int_opts),
            torch::empty({0}, long_opts));
    }

    auto legal_mask_valid = legal_mask_b.index_select(0, valid_root_indices);
    auto probs_valid = probs_f.index_select(0, valid_root_indices);
    auto metadata_valid = metadata_i.index_select(0, valid_root_indices);

    const int64_t num_roots = valid_root_indices.size(0);
    const int64_t max_actions = counts.max().item<int64_t>();

    auto local_pos_all = legal_mask_valid.to(torch::kInt64).cumsum(1) - 1;
    auto nz = torch::nonzero(legal_mask_valid);
    auto row_ids = nz.select(1, 0).to(torch::kInt64);
    auto legal_indices_all = nz.select(1, 1).to(torch::kInt64);
    auto flat_valid_idx = row_ids * total_actions + legal_indices_all;
    auto local_pos = local_pos_all.reshape({-1}).index_select(0, flat_valid_idx);
    auto pack_flat_idx = row_ids * max_actions + local_pos;

    auto valid_mask = torch::zeros(
        {num_roots, max_actions},
        torch::TensorOptions().dtype(torch::kBool).device(device));
    valid_mask.reshape({-1}).index_fill_(0, pack_flat_idx, true);

    auto legal_index_mat = torch::zeros(
        {num_roots, max_actions},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    legal_index_mat.reshape({-1}).index_put_({pack_flat_idx}, legal_indices_all);

    auto priors_mat = torch::zeros(
        {num_roots, max_actions},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto prob_vals = probs_valid.reshape({-1}).index_select(0, flat_valid_idx);
    priors_mat.reshape({-1}).index_put_({pack_flat_idx}, prob_vals);
    priors_mat = priors_mat / priors_mat.sum(1, true).clamp_min(1e-8);

    auto action_code_mat = torch::zeros(
        {num_roots, max_actions, 4},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto action_vals = metadata_valid.reshape({-1, 4}).index_select(0, flat_valid_idx);
    action_code_mat.reshape({-1, 4}).index_put_({pack_flat_idx}, action_vals);

    auto action_codes_all = action_code_mat.reshape({-1, 4}).index_select(0, pack_flat_idx);
    auto parent_local = (
        torch::arange(num_roots, torch::TensorOptions().dtype(torch::kInt64).device(device))
            .view({num_roots, 1})
            .expand({num_roots, max_actions})
            .reshape({-1})
            .index_select(0, pack_flat_idx));
    auto parent_indices_all = valid_root_indices.index_select(0, parent_local);

    return std::make_tuple(
        terminal_mask,
        valid_root_indices,
        counts,
        valid_mask,
        legal_index_mat,
        priors_mat,
        action_code_mat,
        pack_flat_idx,
        action_codes_all,
        parent_indices_all);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RootSparseWriteback(
    const torch::Tensor& legal_index_mat,
    const torch::Tensor& action_code_mat,
    const torch::Tensor& valid_mask,
    const torch::Tensor& legal_policy,
    const torch::Tensor& local_picks,
    const torch::Tensor& valid_root_indices,
    int64_t batch_size,
    int64_t total_action_dim) {
    TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
    TORCH_CHECK(total_action_dim > 0, "total_action_dim must be positive");
    TORCH_CHECK(legal_index_mat.dim() == 2, "legal_index_mat must be [R, M]");
    TORCH_CHECK(valid_mask.dim() == 2, "valid_mask must be [R, M]");
    TORCH_CHECK(legal_policy.dim() == 2, "legal_policy must be [R, M]");
    TORCH_CHECK(action_code_mat.dim() == 3 && action_code_mat.size(2) == 4, "action_code_mat must be [R, M, 4]");
    TORCH_CHECK(local_picks.dim() == 1, "local_picks must be [R]");
    TORCH_CHECK(valid_root_indices.dim() == 1, "valid_root_indices must be [R]");
    TORCH_CHECK(legal_index_mat.size(0) == valid_mask.size(0) && legal_index_mat.size(1) == valid_mask.size(1), "legal_index_mat/valid_mask shape mismatch");
    TORCH_CHECK(legal_index_mat.size(0) == legal_policy.size(0) && legal_index_mat.size(1) == legal_policy.size(1), "legal_index_mat/legal_policy shape mismatch");
    TORCH_CHECK(legal_index_mat.size(0) == action_code_mat.size(0) && legal_index_mat.size(1) == action_code_mat.size(1), "legal_index_mat/action_code_mat shape mismatch");
    TORCH_CHECK(local_picks.size(0) == legal_index_mat.size(0), "local_picks size mismatch");
    TORCH_CHECK(valid_root_indices.size(0) == legal_index_mat.size(0), "valid_root_indices size mismatch");

    auto device = legal_index_mat.device();
    TORCH_CHECK(
        device == action_code_mat.device() &&
            device == valid_mask.device() &&
            device == legal_policy.device() &&
            device == local_picks.device() &&
            device == valid_root_indices.device(),
        "all tensors must be on the same device");

    auto legal_idx = legal_index_mat.to(torch::kInt64).contiguous();
    auto action_codes = action_code_mat.to(torch::kInt32).contiguous();
    auto mask_b = valid_mask.to(torch::kBool).contiguous();
    auto policy = legal_policy.to(torch::kFloat32).contiguous();
    auto picks = local_picks.to(torch::kInt64).contiguous();
    auto roots = valid_root_indices.to(torch::kInt64).contiguous();

    auto policy_dense_valid = torch::zeros(
        {legal_idx.size(0), total_action_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    policy_dense_valid.scatter_add_(1, legal_idx, policy * mask_b.to(torch::kFloat32));

    auto chosen_indices_local = legal_idx.gather(1, picks.view({-1, 1})).view(-1);
    auto chosen_codes_local = action_codes
                                  .gather(1, picks.view({-1, 1, 1}).expand({-1, 1, 4}))
                                  .view({-1, 4});

    auto policy_dense = torch::zeros(
        {batch_size, total_action_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto chosen_action_indices = torch::full(
        {batch_size},
        -1,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto chosen_action_codes = torch::full(
        {batch_size, 4},
        -1,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto chosen_valid_mask = torch::zeros(
        {batch_size},
        torch::TensorOptions().dtype(torch::kBool).device(device));

    policy_dense.index_copy_(0, roots, policy_dense_valid);
    chosen_action_indices.index_copy_(0, roots, chosen_indices_local);
    chosen_action_codes.index_copy_(0, roots, chosen_codes_local);
    chosen_valid_mask.index_fill_(0, roots, true);

    return std::make_tuple(
        policy_dense,
        chosen_action_indices,
        chosen_action_codes,
        chosen_valid_mask);
}

torch::Tensor SoftValueFromBoardBatch(const torch::Tensor& boards, double soft_value_k) {
    TORCH_CHECK(boards.dim() == 3, "boards must be 3D [N, H, W]");
    auto black = boards.eq(1).sum({1, 2}).to(torch::kFloat32);
    auto white = boards.eq(-1).sum({1, 2}).to(torch::kFloat32);
    auto material_delta = (black - white) / static_cast<float>(v0::kCellCount);
    return torch::tanh(material_delta * static_cast<float>(soft_value_k));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SelfPlayStepInplace(
    torch::Tensor board,
    torch::Tensor marks_black,
    torch::Tensor marks_white,
    torch::Tensor phase,
    torch::Tensor current_player,
    torch::Tensor pending_marks_required,
    torch::Tensor pending_marks_remaining,
    torch::Tensor pending_captures_required,
    torch::Tensor pending_captures_remaining,
    torch::Tensor forced_removals_done,
    torch::Tensor move_count,
    torch::Tensor moves_since_capture,
    torch::Tensor plies,
    torch::Tensor done,
    torch::Tensor active_idx,
    torch::Tensor chosen_action_codes,
    torch::Tensor terminal_mask,
    torch::Tensor chosen_valid_mask,
    int64_t max_game_plies,
    double soft_value_k) {
    TORCH_CHECK(max_game_plies > 0, "max_game_plies must be positive");
    TORCH_CHECK(board.dim() == 3, "board must be [N, H, W]");
    TORCH_CHECK(marks_black.dim() == 3, "marks_black must be [N, H, W]");
    TORCH_CHECK(marks_white.dim() == 3, "marks_white must be [N, H, W]");
    TORCH_CHECK(phase.dim() == 1, "phase must be [N]");
    TORCH_CHECK(current_player.dim() == 1, "current_player must be [N]");
    TORCH_CHECK(pending_marks_required.dim() == 1, "pending_marks_required must be [N]");
    TORCH_CHECK(pending_marks_remaining.dim() == 1, "pending_marks_remaining must be [N]");
    TORCH_CHECK(pending_captures_required.dim() == 1, "pending_captures_required must be [N]");
    TORCH_CHECK(pending_captures_remaining.dim() == 1, "pending_captures_remaining must be [N]");
    TORCH_CHECK(forced_removals_done.dim() == 1, "forced_removals_done must be [N]");
    TORCH_CHECK(move_count.dim() == 1, "move_count must be [N]");
    TORCH_CHECK(moves_since_capture.dim() == 1, "moves_since_capture must be [N]");
    TORCH_CHECK(plies.dim() == 1, "plies must be [N]");
    TORCH_CHECK(done.dim() == 1, "done must be [N]");

    auto batch_n = board.size(0);
    TORCH_CHECK(marks_black.size(0) == batch_n, "marks_black batch mismatch");
    TORCH_CHECK(marks_white.size(0) == batch_n, "marks_white batch mismatch");
    TORCH_CHECK(phase.size(0) == batch_n, "phase batch mismatch");
    TORCH_CHECK(current_player.size(0) == batch_n, "current_player batch mismatch");
    TORCH_CHECK(pending_marks_required.size(0) == batch_n, "pending_marks_required batch mismatch");
    TORCH_CHECK(pending_marks_remaining.size(0) == batch_n, "pending_marks_remaining batch mismatch");
    TORCH_CHECK(pending_captures_required.size(0) == batch_n, "pending_captures_required batch mismatch");
    TORCH_CHECK(pending_captures_remaining.size(0) == batch_n, "pending_captures_remaining batch mismatch");
    TORCH_CHECK(forced_removals_done.size(0) == batch_n, "forced_removals_done batch mismatch");
    TORCH_CHECK(move_count.size(0) == batch_n, "move_count batch mismatch");
    TORCH_CHECK(moves_since_capture.size(0) == batch_n, "moves_since_capture batch mismatch");
    TORCH_CHECK(plies.size(0) == batch_n, "plies batch mismatch");
    TORCH_CHECK(done.size(0) == batch_n, "done batch mismatch");

    auto device = board.device();
    auto check_device = [&](const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.device() == device, name, " must be on the same device as board");
    };
    check_device(marks_black, "marks_black");
    check_device(marks_white, "marks_white");
    check_device(phase, "phase");
    check_device(current_player, "current_player");
    check_device(pending_marks_required, "pending_marks_required");
    check_device(pending_marks_remaining, "pending_marks_remaining");
    check_device(pending_captures_required, "pending_captures_required");
    check_device(pending_captures_remaining, "pending_captures_remaining");
    check_device(forced_removals_done, "forced_removals_done");
    check_device(move_count, "move_count");
    check_device(moves_since_capture, "moves_since_capture");
    check_device(plies, "plies");
    check_device(done, "done");

    active_idx = active_idx.to(torch::TensorOptions().dtype(torch::kInt64).device(device)).view(-1);
    chosen_action_codes = chosen_action_codes.to(torch::TensorOptions().dtype(torch::kInt32).device(device));
    terminal_mask = terminal_mask.to(torch::TensorOptions().dtype(torch::kBool).device(device)).view(-1);
    chosen_valid_mask = chosen_valid_mask.to(torch::TensorOptions().dtype(torch::kBool).device(device)).view(-1);

    auto active_n = active_idx.size(0);
    TORCH_CHECK(chosen_action_codes.dim() == 2, "chosen_action_codes must be [A, 4]");
    TORCH_CHECK(chosen_action_codes.size(0) == active_n, "chosen_action_codes batch mismatch");
    TORCH_CHECK(chosen_action_codes.size(1) == 4, "chosen_action_codes second dim must be 4");
    TORCH_CHECK(terminal_mask.size(0) == active_n, "terminal_mask batch mismatch");
    TORCH_CHECK(chosen_valid_mask.size(0) == active_n, "chosen_valid_mask batch mismatch");

    auto slots_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::vector<torch::Tensor> slot_chunks;
    std::vector<torch::Tensor> result_chunks;
    std::vector<torch::Tensor> soft_chunks;

    if (active_n == 0) {
        auto empty_slots = torch::empty({0}, slots_options);
        auto empty_float = torch::empty({0}, float_options);
        return std::make_tuple(empty_slots, empty_float, empty_float.clone());
    }

    auto immediate_done_local = terminal_mask.logical_or(chosen_valid_mask.logical_not());
    auto immediate_local_idx = torch::nonzero(immediate_done_local).view(-1);

    if (immediate_local_idx.numel() > 0) {
        auto immediate_slots = active_idx.index_select(0, immediate_local_idx);
        done.index_fill_(0, immediate_slots, true);

        auto terminal_local = terminal_mask.index_select(0, immediate_local_idx);
        auto player_local = current_player.index_select(0, immediate_slots).to(torch::kFloat32);
        auto result_local = torch::where(terminal_local, -player_local, torch::zeros_like(player_local));

        auto board_local = board.index_select(0, immediate_slots);
        auto soft_local = SoftValueFromBoardBatch(board_local, soft_value_k);

        slot_chunks.push_back(immediate_slots);
        result_chunks.push_back(result_local);
        soft_chunks.push_back(soft_local);
    }

    auto valid_local = immediate_done_local.logical_not();
    auto valid_local_idx = torch::nonzero(valid_local).view(-1);
    if (valid_local_idx.numel() > 0) {
        auto valid_slots = active_idx.index_select(0, valid_local_idx);
        auto valid_action_codes = chosen_action_codes.index_select(0, valid_local_idx);
        auto applied = v0::batch_apply_moves(
            board,
            marks_black,
            marks_white,
            phase,
            current_player,
            pending_marks_required,
            pending_marks_remaining,
            pending_captures_required,
            pending_captures_remaining,
            forced_removals_done,
            move_count,
            moves_since_capture,
            valid_action_codes,
            valid_slots);

        auto next_board = std::get<0>(applied);
        auto next_marks_black = std::get<1>(applied);
        auto next_marks_white = std::get<2>(applied);
        auto next_phase = std::get<3>(applied);
        auto next_current_player = std::get<4>(applied);
        auto next_pending_marks_required = std::get<5>(applied);
        auto next_pending_marks_remaining = std::get<6>(applied);
        auto next_pending_captures_required = std::get<7>(applied);
        auto next_pending_captures_remaining = std::get<8>(applied);
        auto next_forced_removals_done = std::get<9>(applied);
        auto next_move_count = std::get<10>(applied);
        auto next_moves_since_capture = std::get<11>(applied);

        board.index_copy_(0, valid_slots, next_board);
        marks_black.index_copy_(0, valid_slots, next_marks_black.to(torch::kBool));
        marks_white.index_copy_(0, valid_slots, next_marks_white.to(torch::kBool));
        phase.index_copy_(0, valid_slots, next_phase);
        current_player.index_copy_(0, valid_slots, next_current_player);
        pending_marks_required.index_copy_(0, valid_slots, next_pending_marks_required);
        pending_marks_remaining.index_copy_(0, valid_slots, next_pending_marks_remaining);
        pending_captures_required.index_copy_(0, valid_slots, next_pending_captures_required);
        pending_captures_remaining.index_copy_(0, valid_slots, next_pending_captures_remaining);
        forced_removals_done.index_copy_(0, valid_slots, next_forced_removals_done);
        move_count.index_copy_(0, valid_slots, next_move_count);
        moves_since_capture.index_copy_(0, valid_slots, next_moves_since_capture);

        auto ones = torch::ones({valid_slots.size(0)}, slots_options);
        plies.index_add_(0, valid_slots, ones);
        auto updated_plies = plies.index_select(0, valid_slots);

        auto winner_sign = torch::zeros({valid_slots.size(0)}, torch::TensorOptions().dtype(torch::kInt8).device(device));
        auto non_placement = next_phase.ne(static_cast<int64_t>(v0::Phase::kPlacement));
        auto black_count = next_board.eq(1).sum({1, 2});
        auto white_count = next_board.eq(-1).sum({1, 2});
        winner_sign = torch::where(
            non_placement.logical_and(black_count.eq(0)),
            torch::full_like(winner_sign, -1),
            winner_sign);
        winner_sign = torch::where(
            non_placement.logical_and(white_count.eq(0)),
            torch::full_like(winner_sign, 1),
            winner_sign);

        auto draw_limit = next_move_count.ge(v0::kMaxMoveCount).logical_or(
            next_moves_since_capture.ge(v0::kNoCaptureDrawLimit));
        auto hit_max_plies = updated_plies.ge(max_game_plies);
        auto finalize_mask = winner_sign.ne(0).logical_or(draw_limit).logical_or(hit_max_plies);
        auto finalize_local_idx = torch::nonzero(finalize_mask).view(-1);
        if (finalize_local_idx.numel() > 0) {
            auto finalize_slots = valid_slots.index_select(0, finalize_local_idx);
            done.index_fill_(0, finalize_slots, true);

            auto winner_local = winner_sign.index_select(0, finalize_local_idx).to(torch::kFloat32);
            auto result_local = torch::where(
                winner_local.ne(0),
                winner_local,
                torch::zeros_like(winner_local));
            auto board_local = next_board.index_select(0, finalize_local_idx);
            auto soft_local = SoftValueFromBoardBatch(board_local, soft_value_k);

            slot_chunks.push_back(finalize_slots);
            result_chunks.push_back(result_local);
            soft_chunks.push_back(soft_local);
        }
    }

    if (slot_chunks.empty()) {
        auto empty_slots = torch::empty({0}, slots_options);
        auto empty_float = torch::empty({0}, float_options);
        return std::make_tuple(empty_slots, empty_float, empty_float.clone());
    }
    return std::make_tuple(
        torch::cat(slot_chunks, 0),
        torch::cat(result_chunks, 0),
        torch::cat(soft_chunks, 0));
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
        .def_readwrite("max_actions_per_batch", &v0::MCTSConfig::max_actions_per_batch)
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
            "set_eval_batcher",
            [](v0::MCTSCore& core, std::shared_ptr<v0::EvalBatcher> batcher) {
                if (!batcher) {
                    throw std::runtime_error("EvalBatcher is null");
                }
                core.SetForwardCallback([batcher](const torch::Tensor& inputs) {
                    return batcher->Forward(inputs, inputs.size(0));
                });
            },
            py::arg("batcher"))
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
        .def(
            "run_simulations",
            &v0::MCTSCore::RunSimulations,
            py::arg("num_simulations"),
            py::call_guard<py::gil_scoped_release>())
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
        .def("reset_eval_stats", &v0::MCTSCore::ResetEvalStats)
        .def(
            "get_eval_stats",
            [](const v0::MCTSCore& core) {
                const auto stats = core.GetEvalStats();
                py::dict result;
                result["eval_calls"] = stats.eval_calls;
                result["eval_leaves"] = stats.eval_leaves;
                result["full512_calls"] = stats.full512_calls;
                py::list hist;
                for (auto count : stats.hist) {
                    hist.append(count);
                }
                result["hist"] = hist;
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
        py::arg("moves_since_capture"),
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
    m.def(
        "root_puct_allocate_visits",
        &RootPuctAllocateVisits,
        py::arg("priors"),
        py::arg("leaf_values"),
        py::arg("valid_mask"),
        py::arg("num_simulations"),
        py::arg("exploration_weight"));
    m.def(
        "root_pack_sparse_actions",
        &RootPackSparseActions,
        py::arg("legal_mask"),
        py::arg("probs"),
        py::arg("metadata"));
    m.def(
        "root_sparse_writeback",
        &RootSparseWriteback,
        py::arg("legal_index_mat"),
        py::arg("action_code_mat"),
        py::arg("valid_mask"),
        py::arg("legal_policy"),
        py::arg("local_picks"),
        py::arg("valid_root_indices"),
        py::arg("batch_size"),
        py::arg("total_action_dim"));
    m.def(
        "self_play_step_inplace",
        &SelfPlayStepInplace,
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
        py::arg("moves_since_capture"),
        py::arg("plies"),
        py::arg("done"),
        py::arg("active_idx"),
        py::arg("chosen_action_codes"),
        py::arg("terminal_mask"),
        py::arg("chosen_valid_mask"),
        py::arg("max_game_plies"),
        py::arg("soft_value_k"));

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

    py::class_<v0::EvalBatcher, std::shared_ptr<v0::EvalBatcher>>(m, "EvalBatcher")
        .def(
            py::init<std::shared_ptr<v0::InferenceEngine>, int64_t, int64_t, int64_t, int64_t, int64_t>(),
            py::arg("engine"),
            py::arg("batch_size") = 512,
            py::arg("input_channels") = 11,
            py::arg("height") = v0::kBoardSize,
            py::arg("width") = v0::kBoardSize,
            py::arg("timeout_ms") = 2)
        .def("forward", &v0::EvalBatcher::Forward, py::arg("input"), py::arg("n_valid") = -1)
        .def("shutdown", &v0::EvalBatcher::Shutdown)
        .def("reset_eval_stats", &v0::EvalBatcher::ResetStats)
        .def(
            "get_eval_stats",
            [](const v0::EvalBatcher& batcher) {
                const auto stats = batcher.GetStats();
                py::dict result;
                result["eval_calls"] = stats.eval_calls;
                result["eval_leaves"] = stats.eval_leaves;
                result["full512_calls"] = stats.full512_calls;
                py::list hist;
                for (auto count : stats.hist) {
                    hist.append(count);
                }
                result["hist"] = hist;
                return result;
            })
        .def_property_readonly("batch_size", &v0::EvalBatcher::BatchSize)
        .def_property_readonly("timeout_ms", &v0::EvalBatcher::TimeoutMs);

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
